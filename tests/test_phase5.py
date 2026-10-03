"""
tests/test_phase5.py
Verification test suite for Phase 5 - False Alarm Reduction (Temporal Filtering).
Tests single-frame rejection, multi-frame confirmation, sliding window expiration,
confidence filtering, non-target class filtering, and multi-camera isolation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.temporal_filter import TemporalFilter, MultiCameraTemporalFilter
from src import config


def test_single_frame_no_alert():
    """Verify that a single-frame detection does NOT trigger an alert."""
    print("[1/5] Testing single-frame detection (false alarm suppression)...")
    # Require 5 detections in 10 frames
    t_filter = TemporalFilter(confirm_frames=5, window_size=10, conf_threshold=0.50)

    # Frame 1: Transient detection (e.g. reflection / glint)
    single_detection = [{"label": "fire", "confidence": 0.88, "bbox": (100, 100, 200, 200)}]
    status = t_filter.update(single_detection)

    print(f"      - Frame 1 status: positive_frames={status.positive_frames}/5, confirmed={status.is_confirmed}")
    assert status.is_confirmed is False, "Single-frame detection MUST NOT trigger a confirmed alert"
    assert status.positive_frames == 1

    # Frames 2-5: Clear frames (no detections)
    for i in range(2, 6):
        status = t_filter.update([])
        assert status.is_confirmed is False

    print("      -> Single-frame false alarm suppression passed.")


def test_repeated_detections_confirm():
    """Verify that repeated detections (>= 5 frames) trigger a confirmed alert."""
    print("[2/5] Testing repeated detections (sustained hazard confirmation)...")
    t_filter = TemporalFilter(confirm_frames=5, window_size=10, conf_threshold=0.50)
    fire_detection = [{"label": "fire", "confidence": 0.92, "bbox": (120, 120, 240, 240)}]

    for frame_idx in range(1, 5):
        status = t_filter.update(fire_detection)
        print(f"      - Frame {frame_idx}: positive_frames={status.positive_frames}/5, confirmed={status.is_confirmed}")
        assert status.is_confirmed is False, f"Should not confirm on frame {frame_idx} (requires 5)"

    # Frame 5: 5th detection hits the threshold
    status_5 = t_filter.update(fire_detection)
    print(f"      - Frame 5: positive_frames={status_5.positive_frames}/5, confirmed={status_5.is_confirmed}")
    assert status_5.is_confirmed is True, "5th sustained detection MUST confirm the alert"
    assert "fire" in status_5.confirmed_labels
    assert status_5.max_confidence >= 0.90
    print("      -> Sustained detection confirmation passed.")


def test_sliding_window_recovery():
    """Verify that when hazard disappears, confirmed state decays back to False."""
    print("[3/5] Testing sliding window decay and recovery...")
    t_filter = TemporalFilter(confirm_frames=5, window_size=10, conf_threshold=0.50)
    fire_detection = [{"label": "fire", "confidence": 0.92, "bbox": (100, 100, 200, 200)}]

    # Feed 5 positive frames to confirm
    for _ in range(5):
        t_filter.update(fire_detection)
    assert t_filter.is_currently_confirmed is True

    # Feed clear frames until positive count drops below threshold
    # Window size is 10, initially [1, 1, 1, 1, 1]
    # Adding 6 clear frames pushes out old detections
    for i in range(1, 7):
        status = t_filter.update([])

    print(f"      - After 6 clear frames: positive_frames={status.positive_frames}, confirmed={status.is_confirmed}")
    assert status.is_confirmed is False, "Alert state must clear when hazard is no longer present"
    print("      -> Sliding window decay and recovery passed.")


def test_confidence_and_label_filtering():
    """Verify low-confidence detections and non-target labels are filtered out."""
    print("[4/5] Testing confidence thresholding and target class filtering...")
    t_filter = TemporalFilter(confirm_frames=3, window_size=10, conf_threshold=0.60)

    # 1. Low confidence detection (0.45 < 0.60 threshold)
    low_conf_detection = [{"label": "fire", "confidence": 0.45, "bbox": (100, 100, 200, 200)}]
    for _ in range(5):
        status = t_filter.update(low_conf_detection)
    assert status.positive_frames == 0, "Low-confidence detections must not count towards confirmation"
    assert status.is_confirmed is False

    t_filter.reset()

    # 2. Non-target labels (e.g. person, chair)
    non_target_detection = [{"label": "person", "confidence": 0.95, "bbox": (50, 50, 200, 300)}]
    for _ in range(5):
        status = t_filter.update(non_target_detection)
    assert status.positive_frames == 0, "Non-target classes must not trigger fire/smoke confirmation"
    assert status.is_confirmed is False
    print("      -> Confidence and target label filtering passed.")


def test_multi_camera_isolation():
    """Verify independent temporal confirmation per camera feed."""
    print("[5/5] Testing multi-camera temporal filter isolation...")
    multi_filter = MultiCameraTemporalFilter(confirm_frames=5, window_size=10, conf_threshold=0.50)

    fire_detection = [{"label": "fire", "confidence": 0.91, "bbox": (100, 100, 200, 200)}]

    # Feed 5 detections to "Camera A"
    for _ in range(5):
        status_a = multi_filter.update("Camera A", fire_detection)

    # "Camera B" receives clean frames
    status_b = multi_filter.update("Camera B", [])

    print(f"      - Camera A confirmed: {status_a.is_confirmed}")
    print(f"      - Camera B confirmed: {status_b.is_confirmed}")

    assert status_a.is_confirmed is True, "Camera A should be confirmed"
    assert status_b.is_confirmed is False, "Camera B must remain unconfirmed"
    print("      -> Multi-camera isolation passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 5 VERIFICATION SUITE")
    print("=" * 60)
    test_single_frame_no_alert()
    test_repeated_detections_confirm()
    test_sliding_window_recovery()
    test_confidence_and_label_filtering()
    test_multi_camera_isolation()
    print("=" * 60)
    print("PHASE 5 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)
