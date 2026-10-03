"""
src/database.py
Database and snapshot management module for the AI Smart Smoke & Fire Detection System.
Implements SQLite storage for confirmed hazard incidents and automatic snapshot frame persistence.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
import sqlite3
import threading
import logging
import uuid
import cv2
import numpy as np

from src import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database connections, schema migrations, incident event logging,
    and snapshot image storage.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        snapshot_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize DatabaseManager.

        Args:
            db_path: Path to SQLite database file (defaults to config.DB_PATH).
            snapshot_dir: Directory where snapshots are saved (defaults to config.SNAPSHOT_DIR).
        """
        self.db_path = Path(db_path or getattr(config, "DB_PATH", "events.db"))
        # If relative path, resolve relative to project BASE_DIR
        if not self.db_path.is_absolute() and hasattr(config, "BASE_DIR"):
            self.db_path = config.BASE_DIR / self.db_path

        self.snapshot_dir = Path(snapshot_dir or getattr(config, "SNAPSHOT_DIR", "snapshots"))
        if not self.snapshot_dir.is_absolute() and hasattr(config, "BASE_DIR"):
            self.snapshot_dir = config.BASE_DIR / self.snapshot_dir

        self._lock = threading.Lock()
        self._ensure_storage_ready()
        self.init_db()

    @classmethod
    def from_config(cls) -> "DatabaseManager":
        """Factory method to initialize DatabaseManager using application configuration."""
        return cls(
            db_path=getattr(config, "DB_PATH", "events.db"),
            snapshot_dir=getattr(config, "SNAPSHOT_DIR", "snapshots")
        )

    def _ensure_storage_ready(self) -> None:
        """Ensure parent directory for database and snapshot folder exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection configured with WAL journal mode and row factory."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        conn.row_factory = sqlite3.Row
        # Enable Write-Ahead Logging for high-concurrency read/write
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init_db(self) -> None:
        """Initialize database schema and required indexes."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        camera TEXT NOT NULL,
                        location TEXT,
                        label TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        severity TEXT NOT NULL,
                        snapshot TEXT
                    );
                """)
                # Optimize query performance for dashboard filters & history
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_label ON events(label);")
                conn.commit()
                logger.debug("Database initialized at %s", self.db_path)

    def save_snapshot(
        self,
        frame: np.ndarray,
        camera: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Save a video frame to the snapshot directory with a standardized filename.

        Example filename:
        snapshots/20261003_132530.jpg or snapshots/20261003_132530_kitchen_1.jpg

        Args:
            frame: OpenCV image array (BGR).
            camera: Optional camera name used for disambiguation.
            timestamp: Optional datetime object (defaults to datetime.now()).

        Returns:
            str: Path to the saved snapshot image (relative or string representation).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: Must be a non-empty numpy.ndarray.")

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        dt = timestamp or datetime.now()
        base_name = dt.strftime("%Y%m%d_%H%M%S")

        # Clean camera name for filesystem safety
        camera_slug = ""
        if camera:
            camera_slug = "_" + "".join(c if c.isalnum() else "_" for c in camera).strip("_")

        candidate_name = f"{base_name}{camera_slug}.jpg"
        target_path = self.snapshot_dir / candidate_name

        # Ensure no accidental overwrite if multiple events fire in the exact same second
        counter = 1
        while target_path.exists():
            candidate_name = f"{base_name}{camera_slug}_{counter}.jpg"
            target_path = self.snapshot_dir / candidate_name
            counter += 1

        success = cv2.imwrite(str(target_path), frame)
        if not success:
            raise IOError(f"Failed to write snapshot image to {target_path}")

        logger.info("Snapshot saved: %s", target_path)

        # Return standardized relative path if within project directory, else absolute string
        try:
            if hasattr(config, "BASE_DIR"):
                return str(target_path.relative_to(config.BASE_DIR)).replace("\\", "/")
        except ValueError:
            pass
        return str(target_path).replace("\\", "/")

    def log_event(
        self,
        event: Optional[Any] = None,
        frame: Optional[np.ndarray] = None,
        camera: Optional[str] = None,
        label: Optional[str] = None,
        confidence: Optional[float] = None,
        severity: Optional[str] = None,
        location: Optional[str] = None,
        snapshot: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Store a confirmed incident event into SQLite.

        Accepts either:
        1. An IncidentEvent instance or dictionary.
        2. Individual keyword arguments.
        3. Optional frame to automatically save and link a snapshot.

        Args:
            event: IncidentEvent object or dict.
            frame: Optional video frame; if provided, snapshot will be saved automatically.
            camera: Camera name.
            label: Detection label ('fire', 'smoke').
            confidence: Confidence score (float).
            severity: Severity level ('LOW', 'MEDIUM', 'HIGH').
            location: Physical location identifier.
            snapshot: Path to an existing snapshot.
            timestamp: Timestamp string (YYYY-MM-DD HH:MM:SS).

        Returns:
            int: The primary key ID of the newly inserted record.
        """
        ev_camera = camera
        ev_location = location
        ev_label = label
        ev_confidence = confidence
        ev_severity = severity
        ev_snapshot = snapshot
        ev_timestamp = timestamp

        # Extract values if an event object or dict is passed
        if event is not None:
            if hasattr(event, "camera"):
                ev_camera = getattr(event, "camera", ev_camera)
                ev_location = getattr(event, "location", ev_location)
                ev_label = getattr(event, "label", ev_label)
                ev_confidence = getattr(event, "confidence", ev_confidence)
                ev_severity = str(getattr(event, "severity", ev_severity))
                ev_snapshot = getattr(event, "snapshot", ev_snapshot)
                ev_timestamp = getattr(event, "timestamp", ev_timestamp)
            elif isinstance(event, dict):
                ev_camera = event.get("camera", ev_camera)
                ev_location = event.get("location", ev_location)
                ev_label = event.get("label", event.get("detection", ev_label))
                ev_confidence = event.get("confidence", ev_confidence)
                ev_severity = str(event.get("severity", ev_severity))
                ev_snapshot = event.get("snapshot", ev_snapshot)
                ev_timestamp = event.get("timestamp", ev_timestamp)

        # Fallbacks for missing values
        if not ev_camera:
            ev_camera = "Unknown Camera"
        if not ev_label:
            ev_label = "fire"
        if ev_confidence is None:
            ev_confidence = 1.0
        if not ev_severity:
            ev_severity = "HIGH"
        if not ev_timestamp:
            ev_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Automatically capture snapshot if frame was supplied and snapshot is not yet saved
        if frame is not None and not ev_snapshot:
            ev_snapshot = self.save_snapshot(frame, camera=ev_camera)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO events (
                        timestamp, camera, location, label, confidence, severity, snapshot
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (
                    ev_timestamp,
                    ev_camera,
                    ev_location,
                    ev_label,
                    float(ev_confidence),
                    str(ev_severity),
                    ev_snapshot
                ))
                conn.commit()
                inserted_id = cursor.lastrowid
                logger.info(
                    "Event #%d logged: [%s] %s (Conf: %.2f, Sev: %s, Snapshot: %s)",
                    inserted_id, ev_camera, ev_label, ev_confidence, ev_severity, ev_snapshot
                )
                return inserted_id

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single event record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE id = ?;", (event_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        camera: Optional[str] = None,
        label: Optional[str] = None,
        severity: Optional[str] = None,
        order_desc: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query events with filtering options and pagination.

        Args:
            limit: Maximum records to return.
            offset: Number of records to skip.
            camera: Optional camera name filter.
            label: Optional detection label filter ('fire', 'smoke').
            severity: Optional severity filter ('LOW', 'MEDIUM', 'HIGH').
            order_desc: Order by ID descending (newest first).

        Returns:
            List of event record dictionaries.
        """
        conditions = []
        params: List[Any] = []

        if camera:
            conditions.append("camera = ?")
            params.append(camera)
        if label:
            conditions.append("LOWER(label) = ?")
            params.append(label.lower())
        if severity:
            conditions.append("UPPER(severity) = ?")
            params.append(severity.upper())

        query = "SELECT * FROM events"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        direction = "DESC" if order_desc else "ASC"
        query += f" ORDER BY id {direction} LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_event_counts(self) -> Dict[str, int]:
        """
        Compute summary counts for the dashboard summary cards.

        Returns:
            Dict containing:
            - total: Total number of logged events
            - fire: Total fire events
            - smoke: Total smoke events
            - high_severity: Total HIGH severity events
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN LOWER(label) = 'fire' THEN 1 ELSE 0 END) as fire,
                    SUM(CASE WHEN LOWER(label) = 'smoke' THEN 1 ELSE 0 END) as smoke,
                    SUM(CASE WHEN UPPER(severity) = 'HIGH' THEN 1 ELSE 0 END) as high_severity
                FROM events;
            """)
            row = cursor.fetchone()
            if row:
                return {
                    "total": row["total"] or 0,
                    "fire": row["fire"] or 0,
                    "smoke": row["smoke"] or 0,
                    "high_severity": row["high_severity"] or 0
                }
            return {"total": 0, "fire": 0, "smoke": 0, "high_severity": 0}

    def get_latest_event(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recently logged event."""
        events = self.get_events(limit=1, order_desc=True)
        return events[0] if events else None

    def clear_events(self) -> int:
        """Delete all event records (used for test teardown and cleanup)."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM events;")
                deleted = cursor.rowcount
                conn.commit()
                return deleted

    def close(self) -> None:
        """Cleanly close any standing resources."""
        pass

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
