from pydantic import BaseModel, Field
from typing import Optional

class VehicleEvent(BaseModel):
    track_id: int
    vehicle_type: str
    confidence: float
    center_x: float
    center_y: float
    speed_kmh: Optional[float] = None
    speed_status: str = "UNCALIBRATED"

class Stats(BaseModel):
    total_tracks: int = 0
    active_tracks: int = 0
    cars: int = 0
    motorcycles: int = 0
    buses: int = 0
    trucks: int = 0
    bicycles: int = 0
    fps: float = 0
