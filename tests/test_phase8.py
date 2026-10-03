"""
tests/test_phase8.py
Verification test suite for Phase 8 - Database + Snapshots.
Validates SQLite schema, incident persistence, snapshot image generation,
query filtering, metric aggregation, and thread-safe concurrent writes.
"""

import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.database import DatabaseManager
from src.camera_manager import IncidentEvent
from src import config


def test_database_initialization_and_schema():
    """Verify SQLite database initialization and schema fields."""
    print("[1/6] Testing database initialization and schema structure...")

    temp_dir = Path(tempfile.mkdtemp(prefix="test_db_"))
    try:
        db_file = temp_dir / "test_events.db"
        snap_dir = temp_dir / "snapshots"

        db = DatabaseManager(db_path=db_file, snapshot_dir=snap_dir)
        assert db_file.exists(), "Database file must be created on initialization."

        # Verify table schema columns
        conn = db._get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(events);")
        columns = {row["name"]: row["type"] for row in cursor.fetchall()}
        conn.close()

        print(f"      - Database columns: {list(columns.keys())}")
        required_fields = ["id", "timestamp", "camera", "label", "confidence", "severity", "snapshot"]
        for field in required_fields:
            assert field in columns, f"Missing required column '{field}' in events table."

        print("      -> Schema structure verified successfully.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_snapshot_saving():
    """Verify saving video frames as snapshot files with valid names."""
    print("[2/6] Testing snapshot frame persistence...")

    temp_dir = Path(tempfile.mkdtemp(prefix="test_snap_"))
    try:
        db_file = temp_dir / "events.db"
        snap_dir = temp_dir / "snapshots"
        db = DatabaseManager(db_path=db_file, snapshot_dir=snap_dir)

        # Create a dummy frame (e.g. 480x640 orange fire-like rectangle)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame[100:300, 200:400] = (0, 165, 255)  # Orange in BGR

        snapshot_path_str = db.save_snapshot(dummy_frame, camera="Kitchen Cam")
        snapshot_path = Path(snapshot_path_str)
        if not snapshot_path.is_absolute():
            snapshot_path = BASE_DIR / snapshot_path

        print(f"      - Saved snapshot path: {snapshot_path}")
        assert snapshot_path.exists(), f"Snapshot file does not exist at {snapshot_path}"
        assert snapshot_path.stat().st_size > 0, "Snapshot file must not be empty"

        # Verify saved image can be reloaded by OpenCV
        loaded_img = cv2.imread(str(snapshot_path))
        assert loaded_img is not None, "Saved snapshot could not be loaded by cv2"
        assert loaded_img.shape == (480, 640, 3), f"Unexpected shape {loaded_img.shape}"

        # Clean up created file
        if snapshot_path.exists():
            snapshot_path.unlink()

        print("      -> Snapshot frame persistence verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_event_logging_and_verification():
    """Verify logging an event and checking all fields per specification."""
    print("[3/6] Testing event logging and field verification...")

    temp_dir = Path(tempfile.mkdtemp(prefix="test_log_"))
    try:
        db_file = temp_dir / "events.db"
        snap_dir = temp_dir / "snapshots"
        db = DatabaseManager(db_path=db_file, snapshot_dir=snap_dir)

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # Log event with auto-snapshot
        event_id = db.log_event(
            frame=dummy_frame,
            camera="Kitchen",
            label="fire",
            confidence=0.91,
            severity="HIGH",
            location="Main Kitchen Area",
            timestamp="2026-10-03 13:25:32"
        )

        assert event_id > 0, "Inserted event ID should be positive integer"

        # Retrieve logged event
        record = db.get_event_by_id(event_id)
        assert record is not None, "Event record must exist in database"

        print(f"      - Retrieved record: {dict(record)}")

        # Verification items from specification:
        # - Snapshot exists
        # - Database record exists
        # - Timestamp is correct
        # - Camera is correct
        # - Detection label is correct
        # - Confidence is stored
        # - Severity is stored
        assert record["id"] == event_id
        assert record["timestamp"] == "2026-10-03 13:25:32"
        assert record["camera"] == "Kitchen"
        assert record["label"] == "fire"
        assert abs(record["confidence"] - 0.91) < 1e-4
        assert record["severity"] == "HIGH"
        assert record["snapshot"] is not None

        snap_file = Path(record["snapshot"])
        if not snap_file.is_absolute():
            snap_file = BASE_DIR / snap_file
        assert snap_file.exists(), f"Snapshot path stored in database does not exist: {snap_file}"

        # Clean up snapshot
        if snap_file.exists():
            snap_file.unlink()

        print("      -> All required specification fields verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_incident_event_contract_integration():
    """Verify logging directly from IncidentEvent dataclass."""
    print("[4/6] Testing IncidentEvent dataclass logging integration...")

    temp_dir = Path(tempfile.mkdtemp(prefix="test_incident_"))
    try:
        db_file = temp_dir / "events.db"
        snap_dir = temp_dir / "snapshots"
        db = DatabaseManager(db_path=db_file, snapshot_dir=snap_dir)

        event = IncidentEvent(
            camera="Warehouse Bay 2",
            location="Storage Section B",
            label="smoke",
            confidence=0.84,
            severity="MEDIUM",
            timestamp="2026-10-03 14:15:00",
            snapshot="snapshots/manual_snap.jpg"
        )

        event_id = db.log_event(event=event)
        record = db.get_event_by_id(event_id)

        assert record["camera"] == "Warehouse Bay 2"
        assert record["location"] == "Storage Section B"
        assert record["label"] == "smoke"
        assert abs(record["confidence"] - 0.84) < 1e-4
        assert record["severity"] == "MEDIUM"
        assert record["snapshot"] == "snapshots/manual_snap.jpg"

        print("      -> IncidentEvent contract logging verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_query_filtering_and_counts():
    """Verify query filters, pagination, and dashboard count aggregation."""
    print("[5/6] Testing queries, filters, and count aggregation...")

    temp_dir = Path(tempfile.mkdtemp(prefix="test_query_"))
    try:
        db_file = temp_dir / "events.db"
        snap_dir = temp_dir / "snapshots"
        db = DatabaseManager(db_path=db_file, snapshot_dir=snap_dir)

        # Seed events
        events_data = [
            ("Cam A", "fire", 0.95, "HIGH", "2026-10-03 10:00:00"),
            ("Cam A", "fire", 0.85, "MEDIUM", "2026-10-03 10:05:00"),
            ("Cam B", "smoke", 0.75, "LOW", "2026-10-03 10:10:00"),
            ("Cam B", "smoke", 0.92, "HIGH", "2026-10-03 10:15:00"),
            ("Cam C", "fire", 0.98, "HIGH", "2026-10-03 10:20:00"),
        ]

        for cam, lbl, conf, sev, ts in events_data:
            db.log_event(camera=cam, label=lbl, confidence=conf, severity=sev, timestamp=ts)

        # Total counts test
        counts = db.get_event_counts()
        print(f"      - Aggregated counts: {counts}")
        assert counts["total"] == 5
        assert counts["fire"] == 3
        assert counts["smoke"] == 2
        assert counts["high_severity"] == 3

        # Filtering by camera
        cam_a_events = db.get_events(camera="Cam A")
        assert len(cam_a_events) == 2
        for ev in cam_a_events:
            assert ev["camera"] == "Cam A"

        # Filtering by label
        smoke_events = db.get_events(label="smoke")
        assert len(smoke_events) == 2
        for ev in smoke_events:
            assert ev["label"] == "smoke"

        # Filtering by severity
        high_events = db.get_events(severity="HIGH")
        assert len(high_events) == 3
        for ev in high_events:
            assert ev["severity"] == "HIGH"

        # Test latest event
        latest = db.get_latest_event()
        assert latest is not None
        assert latest["timestamp"] == "2026-10-03 10:20:00"

        # Test pagination
        paged = db.get_events(limit=2, offset=0)
        assert len(paged) == 2

        print("      -> Query filtering and count aggregation verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_concurrent_writes():
    """Verify thread-safety when multiple worker threads insert events simultaneously."""
    print("[6/6] Testing thread-safe concurrent database operations...")

    import concurrent.futures

    temp_dir = Path(tempfile.mkdtemp(prefix="test_concurrent_"))
    try:
        db_file = temp_dir / "events.db"
        snap_dir = temp_dir / "snapshots"
        db = DatabaseManager(db_path=db_file, snapshot_dir=snap_dir)

        def worker_write(idx: int):
            return db.log_event(
                camera=f"ThreadCam_{idx % 3}",
                label="fire" if idx % 2 == 0 else "smoke",
                confidence=0.80 + (idx % 20) * 0.01,
                severity="HIGH" if idx % 2 == 0 else "MEDIUM",
                timestamp=f"2026-10-03 12:00:{idx:02d}"
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_write, i) for i in range(30)]
            results = [f.result() for f in futures]

        assert len(results) == 30
        counts = db.get_event_counts()
        assert counts["total"] == 30
        print(f"      - Successfully wrote {counts['total']} concurrent records without locks or collisions.")

        print("      -> Concurrent operations verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """Run all Phase 8 verification tests."""
    print("=" * 60)
    print("RUNNING PHASE 8 VERIFICATION SUITE (DATABASE + SNAPSHOTS)")
    print("=" * 60)

    test_database_initialization_and_schema()
    test_snapshot_saving()
    test_event_logging_and_verification()
    test_incident_event_contract_integration()
    test_query_filtering_and_counts()
    test_concurrent_writes()

    print("=" * 60)
    print("PHASE 8 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
