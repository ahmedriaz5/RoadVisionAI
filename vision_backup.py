import cv2
import time
import threading
from collections import Counter
from pathlib import Path
from ultralytics import YOLO
from .config import get_settings
from .schemas import VehicleEvent, Stats

VEHICLE_CLASSES = {1:"bicycle", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}

class TrafficEngine:
    def __init__(self, event_callback=None):
        self.settings = get_settings()
        self.model = YOLO(self.settings.model_name)
        self.event_callback = event_callback
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.running = False
        self.thread = None
        self.total_tracks = set()
        self.active = {}
        self.fps = 0.0
        self.status = "INITIALIZING"
        self.error = None
        self.source = self.settings.source

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _device(self):
        if self.settings.device != "auto":
            return self.settings.device
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _open_source(self):
        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _loop(self):
        cap = self._open_source()
        if cap is None:
            self.status = "SOURCE_ERROR"
            self.error = (
                f"Could not open video source: {self.source!r}. "
                "Put a video at data/videos/test.mp4 or set VIDEO_SOURCE=0 for your webcam."
            )
            self.running = False
            return

        self.status = "RUNNING"
        self.error = None
        last = time.perf_counter()
        frame_counter = 0

        is_file = isinstance(self.source, str) and Path(self.source).suffix.lower() in {".mp4",".avi",".mov",".mkv",".webm"}

        while self.running:
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.status = "STREAM_ENDED"
                break

            now = time.perf_counter()
            frame_counter += 1
            elapsed = now - last
            if elapsed >= 1:
                self.fps = frame_counter / elapsed
                frame_counter = 0
                last = now

            try:
                results = self.model.track(
                    frame, persist=True, conf=self.settings.confidence,
                    iou=self.settings.iou, tracker=self.settings.tracker,
                    device=self._device(), verbose=False
                )
            except Exception as exc:
                self.status = "INFERENCE_ERROR"
                self.error = str(exc)
                time.sleep(1)
                continue

            active = {}
            r = results[0] if results else None
            if r is not None and r.boxes is not None:
                boxes = r.boxes
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                xyxy = boxes.xyxy.cpu().tolist()
                cls = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()

                for i, (box, c, conf) in enumerate(zip(xyxy, cls, confs)):
                    if c not in VEHICLE_CLASSES:
                        continue
                    track_id = ids[i] if i < len(ids) else i
                    label = VEHICLE_CLASSES[c]
                    x1,y1,x2,y2 = map(int, box)
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    active[track_id] = {
                        "track_id": track_id, "vehicle_type": label,
                        "confidence": float(conf), "center_x": cx, "center_y": cy
                    }
                    self.total_tracks.add(track_id)

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label.upper()} #{track_id} {conf:.0%}",
                                (x1,max(25,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                                .55, (0,255,0), 2)

            with self.lock:
                self.active = active

            if self.event_callback:
                for item in active.values():
                    try:
                        self.event_callback(VehicleEvent(**item))
                    except Exception:
                        pass

            counts = Counter(v["vehicle_type"] for v in active.values())
            panel = f"ROADVISION AI | ACTIVE {len(active)} | TOTAL {len(self.total_tracks)} | FPS {self.fps:.1f}"
            cv2.rectangle(frame, (0,0), (frame.shape[1], 38), (20,20,20), -1)
            cv2.putText(frame, panel, (10,25), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 2)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),82])
            if ok:
                with self.lock:
                    self.latest_jpeg = jpg.tobytes()

        cap.release()
        if self.status == "RUNNING":
            self.status = "STOPPED"

    def jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def stats(self):
        with self.lock:
            counts = Counter(v["vehicle_type"] for v in self.active.values())
            return Stats(total_tracks=len(self.total_tracks), active_tracks=len(self.active),
                cars=counts["car"], motorcycles=counts["motorcycle"], buses=counts["bus"],
                trucks=counts["truck"], bicycles=counts["bicycle"], fps=round(self.fps,1))

    def health(self):
        with self.lock:
            return {"status": self.status, "error": self.error, "source": str(self.source),
                    "running": self.running, "active_tracks": len(self.active),
                    "total_tracks": len(self.total_tracks)}
