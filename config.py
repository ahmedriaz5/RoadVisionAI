from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "RoadVision AI"
    host: str = "127.0.0.1"
    port: int = 8000
    video_source: str = "data/videos/test.mp4"
    model_name: str = "yolo11n.engine"
    confidence: float = 0.35
    iou: float = 0.50
    device: str = "auto"
    tracker: str = "bytetrack.yaml"
    db_path: str = "data/roadvision.db"
    max_fps: int = 30
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def source(self):
        raw = self.video_source.strip()
        if raw.isdigit():
            return int(raw)
        p = Path(raw)
        return str((ROOT / p).resolve()) if not p.is_absolute() else str(p)

    @property
    def database(self):
        return str((ROOT / self.db_path).resolve())

@lru_cache
def get_settings():
    return Settings()
