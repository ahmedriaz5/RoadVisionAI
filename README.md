# RoadVision AI

Production-oriented real-time traffic intelligence platform scaffold.

## Current implementation
- YOLO vehicle detection
- Persistent multi-object tracking using Ultralytics tracking
- Vehicle counting and per-track records
- Live annotated video
- FastAPI backend
- WebSocket event stream
- SQLite by default for zero-setup development
- React + TypeScript dashboard scaffold
- Configurable camera/video source
- GPU/CPU auto selection
- Architecture ready for speed calibration, ANPR, vehicle attributes, traffic rules and PostgreSQL/Redis

## Important
Speed in the first release is deliberately marked as an estimate. Accurate km/h requires camera/road calibration. Do not treat uncalibrated pixel velocity as legal-grade speed evidence.

## Quick start (Windows)
1. Install Python 3.11.
2. Create a virtual environment:
   `python -m venv .venv`
3. Activate it:
   `.venv\Scripts\activate`
4. Install:
   `pip install -r apps/backend/requirements.txt`
5. Put a road video at `data/videos/test.mp4`, or change `VIDEO_SOURCE` in `.env`.
6. Start:
   `python -m apps.backend.main`
7. Open:
   `http://127.0.0.1:8000`
8. API docs:
   `http://127.0.0.1:8000/docs`

## Optional frontend
The backend serves a minimal browser dashboard. The `apps/frontend` folder is a React/TypeScript foundation for the full command center.

## GPU
The application automatically selects CUDA when PyTorch reports an NVIDIA GPU. Verify with:
`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`

## Next production modules
- calibrated speed engine using homography
- dedicated license plate detector + OCR
- vehicle color/make/model models
- lane/ROI configuration
- traffic-light and stop-line rules
- PostgreSQL + Redis
- authentication/RBAC
- multi-camera management
- evidence snapshots
- incident/event engine
- advanced analytics
