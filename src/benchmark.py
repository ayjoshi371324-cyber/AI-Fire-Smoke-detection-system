"""
src/benchmark.py
Benchmarking and system evaluation module for Phase 11.
Measures both ML metrics (Precision, Recall, F1, Confusion Matrix) and
Application metrics (FPS, Latency, False-Alarm Rate, Detection-to-Alert Delay).
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import time
import cv2
import numpy as np
from pathlib import Path

from src import config
from src.detector import YOLODetector
from src.temporal_filter import TemporalFilter
from src.severity import SeverityEstimator


@dataclass
class ConfusionMatrix:
    """Confusion matrix counts for binary or multi-class detection evaluation."""
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return round(self.true_positive / denom, 4) if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return round(self.true_positive / denom, 4) if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * (p * r) / (p + r), 4) if (p + r) > 0 else 0.0

    @property
    def false_alarm_rate(self) -> float:
        """Rate of false alarms on negative samples (FP / (FP + TN))."""
        denom = self.false_positive + self.true_negative
        return round(self.false_positive / denom, 4) if denom > 0 else 0.0


@dataclass
class BenchmarkReport:
    """Consolidated benchmark and system evaluation report."""
    total_frames: int
    fps: float
    avg_latency_ms: float
    confusion_matrix: ConfusionMatrix
    detection_to_alert_delay_frames: int
    detection_to_alert_delay_sec: float
    test_scenario_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimization_comparison: Dict[str, Any] = field(default_factory=dict)


class SystemEvaluator:
    """
    Evaluator executing standard test scenarios and measuring ML + application metrics.
    """

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        temporal_filter: Optional[TemporalFilter] = None
    ):
        self.detector = detector or YOLODetector()
        self.temporal_filter = temporal_filter or TemporalFilter()

    def generate_scenario_frames(self, scenario: str, count: int = 15) -> Tuple[List[np.ndarray], bool]:
        """
        Synthesize realistic scenario frames for benchmarking:
        - 'fire': Simulated flame with high-temperature color gradient
        - 'smoke': Diffuse gray/dark swirling volumetric patches
        - 'cooking_steam': Light semi-transparent diffuse white mist (non-hazard)
        - 'red_object': Solid geometric red/orange shapes (e.g. red mug, car, tail lamp)
        - 'low_light': Underexposed / dark nighttime ambient scene
        - 'distant_fire': Small pinpoint flame (<1% frame area)

        Returns:
            Tuple[List[np.ndarray], is_hazard_scenario: bool]
        """
        frames = []
        h, w = 480, 640

        if scenario == "fire":
            # Sustained flame simulation
            for i in range(count):
                f = np.zeros((h, w, 3), dtype=np.uint8)
                f[:] = (25, 25, 30)
                cx, cy = w // 2, h // 2
                rx = 80 + int(10 * np.sin(i * 0.7))
                ry = 120 + int(15 * np.cos(i * 0.5))
                cv2.ellipse(f, (cx, cy), (rx + 25, ry + 25), 0, 0, 360, (0, 40, 200), -1)
                cv2.ellipse(f, (cx, cy), (rx, ry), 0, 0, 360, (0, 140, 255), -1)
                cv2.ellipse(f, (cx, cy + 20), (rx // 2, ry // 2), 0, 0, 360, (30, 230, 255), -1)
                frames.append(f)
            return frames, True

        elif scenario == "smoke":
            # Gray volumetric smoke plumes
            for i in range(count):
                f = np.zeros((h, w, 3), dtype=np.uint8)
                f[:] = (35, 35, 40)
                for radius in (140, 100, 70):
                    cx = w // 2 + int(15 * np.sin(i * 0.4))
                    cy = h // 2 - int(radius * 0.5)
                    cv2.circle(f, (cx, cy), radius, (130, 130, 130), -1)
                frames.append(f)
            return frames, True

        elif scenario == "cooking_steam":
            # Negative: diffuse white culinary steam over dark kitchen stove
            for i in range(count):
                f = np.full((h, w, 3), (40, 45, 45), dtype=np.uint8)
                # Soft white translucent cloud
                overlay = f.copy()
                cv2.ellipse(overlay, (w // 2, h // 2), (180, 70), 0, 0, 360, (230, 230, 230), -1)
                cv2.addWeighted(overlay, 0.25, f, 0.75, 0, f)
                frames.append(f)
            return frames, False

        elif scenario == "red_object":
            # Negative: solid red rectangle (e.g. red car, fire extinguisher shell, red poster)
            for i in range(count):
                f = np.full((h, w, 3), (120, 120, 120), dtype=np.uint8)
                # Static pure red block (sharp borders, uniform color, no flame texture)
                cv2.rectangle(f, (200, 150), (440, 330), (0, 0, 240), -1)
                frames.append(f)
            return frames, False

        elif scenario == "low_light":
            # Negative: dark night scene with minor noise
            for i in range(count):
                f = np.full((h, w, 3), (12, 10, 10), dtype=np.uint8)
                frames.append(f)
            return frames, False

        elif scenario == "distant_fire":
            # Distant pinpoint hazard
            for i in range(count):
                f = np.full((h, w, 3), (25, 25, 30), dtype=np.uint8)
                cv2.circle(f, (w // 2, h // 2), 12, (0, 140, 255), -1)
                frames.append(f)
            return frames, True

        else:
            # Neutral background
            for i in range(count):
                frames.append(np.full((h, w, 3), (80, 80, 80), dtype=np.uint8))
            return frames, False

    def evaluate_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Run through all test scenarios and record outcomes."""
        scenarios = [
            ("fire", "Fire Scenario (Hazard)"),
            ("smoke", "Smoke Scenario (Hazard)"),
            ("cooking_steam", "Cooking / Steam (Hard Negative)"),
            ("red_object", "Red Object / Light (Color Rejection)"),
            ("low_light", "Low-Light / Night Scene (Evaluation)"),
            ("distant_fire", "Far-Distance / Small Fire (Evaluation)")
        ]

        results = {}
        for sc_key, sc_name in scenarios:
            frames, is_hazard = self.generate_scenario_frames(sc_key, count=10)
            self.temporal_filter.reset()

            total_detections = 0
            confirmed = False
            confidences = []

            for frame in frames:
                dets = self.detector.detect(frame)
                total_detections += len(dets)
                for d in dets:
                    confidences.append(d["confidence"])
                status = self.temporal_filter.update(dets)
                if status.is_confirmed:
                    confirmed = True

            avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

            results[sc_key] = {
                "name": sc_name,
                "is_hazard": is_hazard,
                "total_detections": total_detections,
                "confirmed": confirmed,
                "avg_confidence": avg_conf,
                "pass_criteria": (confirmed if is_hazard else not confirmed)
            }

        return results

    def measure_throughput_and_latency(
        self,
        num_frames: int = 40,
        imgsz: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Measure FPS throughput and per-frame latency (ms).
        """
        dummy_frame = np.full((480, 640, 3), 40, dtype=np.uint8)

        # Warmup (10 runs)
        for _ in range(10):
            self.detector.detect(dummy_frame, imgsz=imgsz)

        latencies = []
        for _ in range(num_frames):
            t0 = time.perf_counter()
            self.detector.detect(dummy_frame, imgsz=imgsz)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        avg_latency = sum(latencies) / len(latencies)
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        return round(fps, 1), round(avg_latency, 2)

    def run_full_benchmark(self) -> BenchmarkReport:
        """Execute full Phase 11 benchmarking pipeline."""
        # 1. Measure Throughput
        fps_baseline, latency_baseline = self.measure_throughput_and_latency(num_frames=30)
        fps_opt, latency_opt = self.measure_throughput_and_latency(num_frames=30, imgsz=320)

        # 2. Evaluate Scenarios
        scenario_results = self.evaluate_scenarios()

        # 3. Calculate Confusion Matrix & Metrics
        cm = ConfusionMatrix()
        for sc in scenario_results.values():
            is_hazard = sc["is_hazard"]
            triggered = sc["confirmed"]
            if is_hazard:
                if triggered:
                    cm.true_positive += 1
                else:
                    cm.false_negative += 1
            else:
                if triggered:
                    cm.false_positive += 1
                else:
                    cm.true_negative += 1

        # 4. Compute Detection-to-Alert Delay
        delay_frames = self.temporal_filter.confirm_frames
        fps_val = max(1.0, fps_baseline)
        delay_sec = round(delay_frames / fps_val, 2)

        return BenchmarkReport(
            total_frames=120,
            fps=fps_baseline,
            avg_latency_ms=latency_baseline,
            confusion_matrix=cm,
            detection_to_alert_delay_frames=delay_frames,
            detection_to_alert_delay_sec=delay_sec,
            test_scenario_results=scenario_results,
            optimization_comparison={
                "baseline": {"resolution": "640x480", "fps": fps_baseline, "latency_ms": latency_baseline},
                "optimized_imgsz_320": {"resolution": "320x320", "fps": fps_opt, "latency_ms": latency_opt},
                "speedup_ratio": round(fps_opt / max(0.1, fps_baseline), 2)
            }
        )
