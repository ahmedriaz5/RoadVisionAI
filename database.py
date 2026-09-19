import sqlite3
from pathlib import Path
from datetime import datetime

class Database:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(self.path) as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                vehicle_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                center_x REAL NOT NULL,
                center_y REAL NOT NULL
            )
            """)
            con.commit()

    def insert_event(self, event):
        with sqlite3.connect(self.path) as con:
            con.execute(
                """INSERT INTO vehicle_events
                (created_at, track_id, vehicle_type, confidence, center_x, center_y)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (datetime.utcnow().isoformat(), event.track_id, event.vehicle_type,
                 event.confidence, event.center_x, event.center_y)
            )
            con.commit()
