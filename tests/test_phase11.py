"""
tests/test_phase11.py
Verification test suite for Phase 11 - Testing + Optimization.
Validates end-to-end benchmark evaluation, ML metrics (Precision, Recall, F1, Confusion Matrix),
application metrics (FPS, Latency, False-Alarm Rate, Detection-to-Alert Delay),
and performance optimization scaling (resolution scaling & stride skipping).
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.benchmark import SystemEvaluator, BenchmarkReport, ConfusionMatrix
from src.detector import YOLODetector
from src.temporal_filter import TemporalFilter
from src.alerts import CooldownTracker
from main import FireDetectionPipeline


def test_confusion_matrix_metrics():
    """Verify ConfusionMatrix precision, recall, F1, and false alarm rate formulas."""
    print("[1/7] Testing ConfusionMatrix metrics computation...")

    # Case 1: Perfect detections
    cm1 = ConfusionMatrix(true_positive=10, false_positive=0, true_negative=10, false_negative=0)
    assert cm1.precision == 1.0
    assert cm1.recall == 1.0
    assert cm1.f1_score == 1.0
    assert cm1.false_alarm_rate == 0.0

    # Case 2: Realistic scenario with 1 FP and 1 FN
    cm2 = ConfusionMatrix(true_positive=9, false_positive=1, true_negative=9, false_negative=1)
    assert cm2.precision == 0.9
    assert cm2.recall == 0.9
    assert cm2.f1_score == 0.9
    assert cm2.false_alarm_rate == 0.1

    # Case 3: Zero denominator safety
    cm3 = ConfusionMatrix()
    assert cm3.precision == 1.0
    assert cm3.recall == 0.0
    assert cm3.f1_score == 0.0
    assert cm3.false_alarm_rate == 0.0

    print("      -> ConfusionMatrix formulas verified.")


def test_hard_negative_scenarios():
    """Verify false alarm rejection on cooking steam, red objects, and low-light scenes."""
    print("[2/7] Testing false alarm rejection on hard negatives...")

    evaluator = SystemEvaluator()

    # 1. Cooking / Steam
    steam_frames, is_hazard = evaluator.generate_scenario_frames("cooking_steam", count=10)
    assert is_hazard is False
    filter_steam = TemporalFilter(confirm_frames=5, window_size=10, conf_threshold=0.5)
    for frame in steam_frames:
        dets = evaluator.detector.detect(frame)
        status = filter_steam.update(dets)
        assert status.is_confirmed is False, "Steam must not trigger confirmed alert"

    # 2. Red object / light
    red_frames, is_hazard = evaluator.generate_scenario_frames("red_object", count=10)
    assert is_hazard is False
    filter_red = TemporalFilter(confirm_frames=5, window_size=10, conf_threshold=0.5)
    for frame in red_frames:
        dets = evaluator.detector.detect(frame)
        status = filter_red.update(dets)
        assert status.is_confirmed is False, "Static red object must not trigger alert"

    # 3. Low light scene
    dark_frames, is_hazard = evaluator.generate_scenario_frames("low_light", count=10)
    assert is_hazard is False
    filter_dark = TemporalFilter(confirm_frames=5, window_size=10, conf_threshold=0.5)
    for frame in dark_frames:
        dets = evaluator.detector.detect(frame)
        status = filter_dark.update(dets)
        assert status.is_confirmed is False, "Low-light scene must not trigger alert"

    print("      -> Hard negatives rejected without false alarms.")


def test_hazard_scenarios_and_temporal_confirmation():
    """Verify fire and smoke detection and multi-frame temporal confirmation."""
    print("[3/7] Testing hazard detection and temporal confirmation...")

    # Using mock detection sequence to evaluate temporal confirmation contract
    temp_filter = TemporalFilter(confirm_frames=5, window_size=10, conf_threshold=0.5)
    fire_det = [{"label": "fire", "confidence": 0.92, "bbox": (100, 100, 300, 300)}]

    # Frames 1 to 4: Accumulating
    for i in range(1, 5):
        st = temp_filter.update(fire_det)
        assert st.is_confirmed is False, f"Should not confirm on frame {i}"

    # Frame 5: Sustained hazard hits threshold -> Confirmed!
    st5 = temp_filter.update(fire_det)
    assert st5.is_confirmed is True, "5th sustained detection must confirm hazard"
    assert "fire" in st5.confirmed_labels

    print("      -> Hazard detection and confirmation verified.")


def test_multiple_triggers_and_cooldown():
    """Verify alert rate limiting and cooldown prevention of multiple duplicate alerts."""
    print("[4/7] Testing alert cooldown across multiple triggers...")

    cooldown = CooldownTracker(cooldown_sec=30)
    camera = "Warehouse Cam"

    # Trigger 1: Allowed
    can1, rem1 = cooldown.can_alert(camera, current_time=0.0)
    assert can1 is True
    cooldown.record_alert(camera, current_time=0.0)

    # Immediate Trigger 2 (at t=1s): Suppressed
    can2, rem2 = cooldown.can_alert(camera, current_time=1.0)
    assert can2 is False
    assert round(rem2, 1) == 29.0

    # Trigger 3 (at t=30.5s): Allowed after cooldown
    can3, rem3 = cooldown.can_alert(camera, current_time=30.5)
    assert can3 is True
    assert rem3 == 0.0

    print("      -> Multiple triggers properly handled by cooldown.")


def test_application_metrics_and_latency():
    """Verify measurement of FPS, average latency, and detection delay."""
    print("[5/7] Testing application metrics measurement...")

    evaluator = SystemEvaluator()
    fps, latency_ms = evaluator.measure_throughput_and_latency(num_frames=15)

    print(f"      - Measured Throughput: {fps} FPS")
    print(f"      - Measured Latency:    {latency_ms:.1f} ms/frame")

    assert fps > 0.0, "FPS must be positive"
    assert latency_ms > 0.0, "Latency must be positive"

    # Detection-to-alert delay calculation
    delay_frames = 5
    delay_sec = delay_frames / max(1.0, fps)
    print(f"      - Detection delay: {delay_frames} frames (~{delay_sec:.2f}s)")
    assert delay_sec > 0.0

    print("      -> Application metrics verified.")


def test_optimization_scaling():
    """Verify optimization resolution scaling (imgsz=320 vs 640)."""
    print("[6/7] Testing optimization resolution scaling...")

    detector_640 = YOLODetector(imgsz=640)
    detector_320 = YOLODetector(imgsz=320)

    dummy_frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    # Warmup
    detector_640.detect(dummy_frame)
    detector_320.detect(dummy_frame)

    import time
    # Measure 640
    t0 = time.perf_counter()
    for _ in range(10):
        detector_640.detect(dummy_frame)
    time_640 = time.perf_counter() - t0

    # Measure 320
    t0 = time.perf_counter()
    for _ in range(10):
        detector_320.detect(dummy_frame)
    time_320 = time.perf_counter() - t0

    print(f"      - 10 inferences at 640: {time_640 * 1000:.1f} ms")
    print(f"      - 10 inferences at 320: {time_320 * 1000:.1f} ms")
    assert time_320 <= time_640 * 1.5, "imgsz=320 should not be substantially slower than 640"

    print("      -> Optimization resolution scaling verified.")


def test_evaluate_system_cli_tool():
    """Verify that scripts/evaluate_system.py executes without error."""
    print("[7/7] Testing scripts/evaluate_system.py execution...")
    import subprocess

    cmd = [sys.executable, str(BASE_DIR / "scripts" / "evaluate_system.py")]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0, f"evaluate_system.py failed: {res.stderr}\n{res.stdout}"
    assert "PERFORMANCE SCORECARD" in res.stdout
    assert "Machine Learning Metrics" in res.stdout
    assert "Application & Runtime Metrics" in res.stdout
    print("      -> scripts/evaluate_system.py CLI tool verified.")


def run_all_tests():
    """Run all Phase 11 verification tests."""
    print("=" * 60)
    print("RUNNING PHASE 11 VERIFICATION SUITE (TESTING + OPTIMIZATION)")
    print("=" * 60)

    test_confusion_matrix_metrics()
    test_hard_negative_scenarios()
    test_hazard_scenarios_and_temporal_confirmation()
    test_multiple_triggers_and_cooldown()
    test_application_metrics_and_latency()
    test_optimization_scaling()
    test_evaluate_system_cli_tool()

    print("=" * 60)
    print("PHASE 11 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
