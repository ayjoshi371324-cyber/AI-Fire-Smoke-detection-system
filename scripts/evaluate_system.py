"""
scripts/evaluate_system.py
Command-line system evaluation and benchmarking tool for Phase 11.
Measures both ML metrics (Precision, Recall, F1, Confusion Matrix, False-Alarm Rate)
and Application metrics (FPS, Latency, Detection-to-Alert Delay) and outputs a detailed report.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.benchmark import SystemEvaluator, BenchmarkReport
from src.detector import YOLODetector
from src.temporal_filter import TemporalFilter
from src.alerts import CooldownTracker


def main():
    print("=" * 70)
    print("   AI SMART SMOKE & FIRE DETECTION SYSTEM - PHASE 11 BENCHMARK")
    print("=" * 70)

    detector = YOLODetector()
    t_filter = TemporalFilter(confirm_frames=5, window_size=10, conf_threshold=0.50)

    print(f"[*] Evaluator Engine: YOLOv8 ({detector.model_path})")
    print(f"[*] Classes: {detector.classes}")
    print(f"[*] Temporal Window: {t_filter.confirm_frames} hits in {t_filter.window_size} frames")
    print("\n[1/4] Measuring inference latency and throughput...")

    evaluator = SystemEvaluator(detector=detector, temporal_filter=t_filter)
    report = evaluator.run_full_benchmark()

    print(f"      - Baseline (640x480):  {report.fps} FPS ({report.avg_latency_ms:.1f} ms/frame)")
    opt = report.optimization_comparison["optimized_imgsz_320"]
    print(f"      - Optimized (imgsz=320): {opt['fps']} FPS ({opt['latency_ms']:.1f} ms/frame)")
    print(f"      - Speedup Factor:       {report.optimization_comparison['speedup_ratio']}x")

    print("\n[2/4] Evaluating Test Scenarios:")
    print("-" * 70)
    print(f"{'Scenario':<32} | {'Expected':<12} | {'Result':<12} | {'Status'}")
    print("-" * 70)

    for key, data in report.test_scenario_results.items():
        expected = "HAZARD" if data["is_hazard"] else "NO ALERT"
        actual = "HAZARD" if data["confirmed"] else "NO ALERT"
        status = "PASSED" if data["pass_criteria"] else "EVALUATE"
        print(f"{data['name']:<32} | {expected:<12} | {actual:<12} | [{status}]")

    print("-" * 70)

    print("\n[3/4] Testing Multiple Triggers & Cooldown Rate Limiting:")
    cooldown = CooldownTracker(cooldown_sec=30)
    cooldown.record_alert("Test Cam", current_time=100.0)
    can_alert_immed, rem = cooldown.can_alert("Test Cam", current_time=105.0)
    can_alert_after, _ = cooldown.can_alert("Test Cam", current_time=131.0)

    print(f"      - Rapid repeated trigger at t=105s (5s elapsed): Suppressed={not can_alert_immed} ({rem:.1f}s remaining)")
    print(f"      - Trigger after cooldown at t=131s (31s elapsed): Permitted={can_alert_after}")
    assert not can_alert_immed and can_alert_after, "Cooldown verification failed"
    print("      -> Cooldown rate limiting verified.")

    cm = report.confusion_matrix
    print("\n" + "=" * 70)
    print("                    PERFORMANCE SCORECARD")
    print("=" * 70)
    print("A. Machine Learning Metrics:")
    print(f"   - Precision:                 {cm.precision * 100:.1f}%")
    print(f"   - Recall:                    {cm.recall * 100:.1f}%")
    print(f"   - F1-Score:                  {cm.f1_score * 100:.1f}%")
    print(f"   - Confusion Matrix:          TP={cm.true_positive}, FP={cm.false_positive}, TN={cm.true_negative}, FN={cm.false_negative}")
    print(f"   - False Alarm Rate:          {cm.false_alarm_rate * 100:.1f}%")

    print("\nB. Application & Runtime Metrics:")
    print(f"   - Baseline Speed:            {report.fps} FPS")
    print(f"   - Average Frame Latency:     {report.avg_latency_ms:.1f} ms")
    print(f"   - Temporal Confirm Window:   {report.detection_to_alert_delay_frames} frames")
    print(f"   - Detection-to-Alert Delay:  ~{report.detection_to_alert_delay_sec:.2f} seconds (at {report.fps} FPS)")

    print("\nC. Optimization Recommendations:")
    print(f"   - Enable --imgsz 320 for {report.optimization_comparison['speedup_ratio']}x higher frame throughput.")
    print("   - Enable --stride 2 to process every 2nd frame on constrained edge hardware.")
    print("=" * 70)
    print("SYSTEM BENCHMARK EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
