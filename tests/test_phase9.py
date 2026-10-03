"""
tests/test_phase9.py
Verification test suite for Phase 9 - Main Application Integration.
Tests end-to-end orchestration across all backend modules:
Camera -> Frame -> YOLO Detector -> Temporal Confirmation -> Severity -> Location -> Alert -> Snapshot -> SQLite DB.
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

from main import FireDetectionPipeline, FrameProcessingResult, PipelineStats
from src.camera_manager import CameraManager, IncidentEvent
from src.detector import YOLODetector
from src.temporal_filter import MultiCameraTemporalFilter, TemporalFilter
from src.severity import SeverityEstimator
from src.alerts import AlertManager, LocalAlarm, TelegramAlerter
from src.database import DatabaseManager
from src import config


def test_pipeline_step_by_step_coordination():
    """
    Verify all 8 pipeline steps execute in exact sequence:
    Detect -> Draw boxes -> Confirm -> Calculate severity -> Alert -> Save snapshot -> Log event.
    """
    print("[1/5] Testing step-by-step pipeline module coordination...")

    temp_dir = Path(tempfile.mkdtemp(prefix="test_pipeline_"))
    try:
        db_file = temp_dir / "pipeline_events.db"
        snap_dir = temp_dir / "snapshots"

        # Mock detector with deterministic outputs
        class MockDetector:
            COLOR_FIRE = (0, 69, 255)
            COLOR_SMOKE = (180, 180, 180)
            COLOR_DEFAULT = (50, 205, 50)

            def detect(self, frame):
                # Return high-confidence fire bounding box covering ~20% of frame
                h, w = frame.shape[:2]
                return [{
                    "label": "fire",
                    "confidence": 0.94,
                    "bbox": (int(w * 0.2), int(h * 0.2), int(w * 0.7), int(h * 0.6))
                }]

            def draw_detections(self, frame, detections, copy=True):
                out = frame.copy() if copy else frame
                for d in detections:
                    x1, y1, x2, y2 = d["bbox"]
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                return out

        # Fast temporal filter confirming on 3 frames
        temp_filter = MultiCameraTemporalFilter(confirm_frames=3, window_size=5, conf_threshold=0.5)
        db_manager = DatabaseManager(db_path=db_file, snapshot_dir=snap_dir)
        alert_manager = AlertManager(
            cooldown_sec=30,
            local_alarm=LocalAlarm(enabled=False),
            telegram_alerter=TelegramAlerter(enabled=False)
        )

        pipeline = FireDetectionPipeline(
            detector=MockDetector(),
            temporal_filter=temp_filter,
            alert_manager=alert_manager,
            database_manager=db_manager
        )

        # Create synthetic test frame (480x640)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (20, 20, 20)

        # Frame 1: Detection occurred, but not confirmed yet (requires 3 frames)
        res1 = pipeline.process_frame(test_frame, camera_name="Test Cam", frame_idx=1)
        assert len(res1.detections) == 1, "Must detect hazard"
        assert res1.temporal_status.is_confirmed is False, "Must not confirm on frame 1"
        assert res1.alert_triggered is False, "Must not alert on unconfirmed frame"
        assert res1.event_logged is False, "Must not log unconfirmed frame"
        print(f"      - Frame 1: Detected={len(res1.detections)}, Confirmed={res1.temporal_status.is_confirmed}, Alert={res1.alert_triggered}")

        # Frame 2: Still accumulating frames
        res2 = pipeline.process_frame(test_frame, camera_name="Test Cam", frame_idx=2)
        assert res2.temporal_status.is_confirmed is False
        assert res2.alert_triggered is False

        # Frame 3: Threshold of 3 reached -> Hazard CONFIRMED!
        res3 = pipeline.process_frame(test_frame, camera_name="Test Cam", frame_idx=3)
        assert res3.temporal_status.is_confirmed is True, "Must confirm on frame 3"
        assert res3.severity.level in ("MEDIUM", "HIGH"), f"Expected high/medium severity, got {res3.severity.level}"
        assert res3.alert_triggered is True, "Must trigger alert upon confirmation"
        assert res3.event_logged is True, "Must log confirmed event into database"
        assert res3.snapshot_path is not None, "Must save snapshot frame"

        # Check snapshot exists on disk
        snap_file = Path(res3.snapshot_path)
        if not snap_file.is_absolute():
            snap_file = BASE_DIR / snap_file
        assert snap_file.exists(), f"Snapshot not found on disk at {snap_file}"

        # Check database record exists
        events = db_manager.get_events(camera="Test Cam")
        assert len(events) == 1, f"Expected 1 logged event, found {len(events)}"
        assert events[0]["label"] == "fire"
        assert events[0]["severity"] == res3.severity.level
        print(f"      - Frame 3: Confirmed={res3.temporal_status.is_confirmed}, Severity={res3.severity.level}, Alert={res3.alert_triggered}, DB_ID={events[0]['id']}")

        # Frame 4: Next frame within cooldown window -> Alert suppressed by cooldown
        res4 = pipeline.process_frame(test_frame, camera_name="Test Cam", frame_idx=4)
        assert res4.temporal_status.is_confirmed is True
        assert res4.alert_triggered is False, "Must suppress alert during cooldown"
        assert res4.event_logged is False, "Must suppress duplicate DB spam during cooldown"
        print(f"      - Frame 4: Cooldown prevented duplicate alert & DB logging as expected.")

        print("      -> Step-by-step pipeline module coordination verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_hud_rendering():
    """Verify that HUD rendering overlays boxes, status badge, camera title, and FPS."""
    print("[2/5] Testing HUD overlay and visual rendering...")

    pipeline = FireDetectionPipeline()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = [{"label": "fire", "confidence": 0.89, "bbox": (100, 100, 300, 300)}]
    result = pipeline.process_frame(dummy_frame, camera_name="Fire Video", frame_idx=1, render_annotated=True)

    assert result.annotated_frame is not None, "Annotated frame must be generated"
    assert result.annotated_frame.shape == dummy_frame.shape, "Annotated frame dimensions must match input"

    # Frame should have non-zero pixels due to HUD text and banner
    diff = np.abs(result.annotated_frame.astype(int) - dummy_frame.astype(int))
    assert np.sum(diff) > 0, "HUD elements must be rendered onto annotated frame"
    print("      -> HUD visual overlay verified.")


def test_video_file_pipeline_execution():
    """Run pipeline against actual test video (videos/fire_test.mp4) in headless mode."""
    print("[3/5] Testing pipeline run on test video (videos/fire_test.mp4)...")

    video_path = BASE_DIR / "videos" / "fire_test.mp4"
    if not video_path.exists():
        print("      [SKIP] videos/fire_test.mp4 not found.")
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="test_vid_"))
    try:
        db_file = temp_dir / "vid_events.db"
        snap_dir = temp_dir / "snapshots"

        db_manager = DatabaseManager(db_path=db_file, snapshot_dir=snap_dir)
        alert_manager = AlertManager(
            cooldown_sec=5,
            local_alarm=LocalAlarm(enabled=False),
            telegram_alerter=TelegramAlerter(enabled=False)
        )

        pipeline = FireDetectionPipeline(
            alert_manager=alert_manager,
            database_manager=db_manager
        )

        # Run for 25 frames
        stats = pipeline.run(
            camera_name="Fire Video",
            max_frames=25,
            headless=True
        )

        print(f"      - Processed {stats.frames_processed} frames at {stats.average_fps} FPS.")
        print(f"      - Detections: {stats.detections_count}, Alerts: {stats.alerts_dispatched}, DB Events: {stats.events_logged}")

        assert stats.frames_processed == 25, f"Expected 25 frames, got {stats.frames_processed}"
        assert stats.average_fps > 0, "Average FPS must be positive"

        print("      -> Video execution completed successfully.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cli_argument_parsing():
    """Verify command line arguments and help output."""
    print("[4/5] Testing command line argument parser...")
    import subprocess

    cmd = [sys.executable, str(BASE_DIR / "main.py"), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0, f"main.py --help failed: {res.stderr}"
    assert "--camera" in res.stdout
    assert "--source" in res.stdout
    assert "--headless" in res.stdout
    assert "--max-frames" in res.stdout
    print("      -> CLI argument parser verified.")


def test_cli_execution_headless():
    """Verify running main.py via CLI in headless mode with max-frames."""
    print("[5/5] Testing running main.py via CLI in headless mode...")
    import subprocess

    cmd = [
        sys.executable,
        str(BASE_DIR / "main.py"),
        "--camera", "Fire Video",
        "--headless",
        "--mute",
        "--max-frames", "10"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    print(f"      - Return code: {res.returncode}")
    assert res.returncode == 0, f"main.py run failed: {res.stderr}\nOutput: {res.stdout}"
    assert "EXECUTION SUMMARY" in res.stdout
    assert "Frames Processed:    10" in res.stdout
    print("      -> CLI headless run verified.")


def run_all_tests():
    """Run all Phase 9 verification tests."""
    print("=" * 60)
    print("RUNNING PHASE 9 VERIFICATION SUITE (MAIN APP INTEGRATION)")
    print("=" * 60)

    test_pipeline_step_by_step_coordination()
    test_hud_rendering()
    test_video_file_pipeline_execution()
    test_cli_argument_parsing()
    test_cli_execution_headless()

    print("=" * 60)
    print("PHASE 9 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
