"""
tests/test_phase4.py
Verification test suite for Phase 4 - Severity Estimation.
Validates severity transitions across varying bounding-box sizes, multi-hazard combinations,
coordinate clamping, and threshold configurations.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from src.severity import SeverityEstimator, estimate_severity, SeverityLevel


def test_severity_none():
    """Verify that empty or non-fire detections return NONE."""
    print("[1/5] Testing NONE severity...")
    frame_shape = (480, 640, 3)

    # Empty detections
    res_empty = estimate_severity([], frame_shape)
    assert res_empty.level == SeverityLevel.NONE.value
    assert res_empty == "NONE"
    assert res_empty.fire_ratio == 0.0
    assert res_empty.smoke_ratio == 0.0

    # Non-fire/smoke detections (e.g. person, chair)
    res_other = estimate_severity([
        {"label": "person", "confidence": 0.89, "bbox": (50, 50, 200, 350)}
    ], frame_shape)
    assert res_other.level == SeverityLevel.NONE.value
    assert res_other == "NONE"
    print("      -> NONE severity validated.")


def test_severity_low():
    """Verify small fire or smoke bounding boxes produce LOW severity."""
    print("[2/5] Testing LOW severity...")
    # Frame area = 640 * 480 = 307,200 px
    frame_shape = (480, 640, 3)

    # Small flame: 60x60 = 3,600 px (~1.17% of frame, > 1% threshold)
    small_fire = [{"label": "fire", "confidence": 0.90, "bbox": (100, 100, 160, 160)}]
    res_fire = estimate_severity(small_fire, frame_shape)
    print(f"      - Small Fire ratio: {res_fire.fire_ratio*100:.2f}%, level: {res_fire.level}")
    assert res_fire.level == SeverityLevel.LOW.value
    assert res_fire == "LOW"

    # Small smoke plume: 70x70 = 4,900 px (~1.59% of frame)
    small_smoke = [{"label": "smoke", "confidence": 0.85, "bbox": (200, 200, 270, 270)}]
    res_smoke = estimate_severity(small_smoke, frame_shape)
    print(f"      - Small Smoke ratio: {res_smoke.smoke_ratio*100:.2f}%, level: {res_smoke.level}")
    assert res_smoke.level == SeverityLevel.LOW.value
    assert res_smoke == "LOW"
    print("      -> LOW severity validated.")


def test_severity_medium():
    """Verify medium fire or extensive smoke produce MEDIUM severity."""
    print("[3/5] Testing MEDIUM severity...")
    frame_shape = (480, 640, 3)

    # Medium fire: 150x150 = 22,500 px (~7.32% of frame, > 5% threshold)
    medium_fire = [{"label": "fire", "confidence": 0.92, "bbox": (100, 100, 250, 250)}]
    res_fire = estimate_severity(medium_fire, frame_shape)
    print(f"      - Medium Fire ratio: {res_fire.fire_ratio*100:.2f}%, level: {res_fire.level}")
    assert res_fire.level == SeverityLevel.MEDIUM.value
    assert res_fire == "MEDIUM"

    # Extensive smoke: 250x250 = 62,500 px (~20.3% of frame)
    heavy_smoke = [{"label": "smoke", "confidence": 0.88, "bbox": (100, 100, 350, 350)}]
    res_smoke = estimate_severity(heavy_smoke, frame_shape)
    print(f"      - Heavy Smoke ratio: {res_smoke.smoke_ratio*100:.2f}%, level: {res_smoke.level}")
    assert res_smoke.level == SeverityLevel.MEDIUM.value
    assert res_smoke == "MEDIUM"
    print("      -> MEDIUM severity validated.")


def test_severity_high():
    """Verify large fire or compound fire + smoke produce HIGH severity."""
    print("[4/5] Testing HIGH severity...")
    frame_shape = (480, 640, 3)

    # Large fire: 250x250 = 62,500 px (~20.3% of frame, > 15% high threshold)
    large_fire = [{"label": "fire", "confidence": 0.96, "bbox": (100, 100, 350, 350)}]
    res_large = estimate_severity(large_fire, frame_shape)
    print(f"      - Large Fire ratio: {res_large.fire_ratio*100:.2f}%, level: {res_large.level}")
    assert res_large.level == SeverityLevel.HIGH.value
    assert res_large == "HIGH"

    # Compound hazard: Medium Fire (6%) + Medium Smoke (6%)
    # Fire: 140x140 = 19,600 (~6.38%)
    # Smoke: 140x140 = 19,600 (~6.38%)
    compound_hazard = [
        {"label": "fire", "confidence": 0.91, "bbox": (50, 50, 190, 190)},
        {"label": "smoke", "confidence": 0.87, "bbox": (250, 50, 390, 190)}
    ]
    res_compound = estimate_severity(compound_hazard, frame_shape)
    print(f"      - Compound hazard: fire {res_compound.fire_ratio*100:.2f}%, smoke {res_compound.smoke_ratio*100:.2f}%, level: {res_compound.level}")
    assert res_compound.level == SeverityLevel.HIGH.value
    assert res_compound == "HIGH"
    print("      -> HIGH severity validated.")


def test_edge_cases_and_clamping():
    """Verify coordinate clamping, custom thresholds, and color mappings."""
    print("[5/5] Testing boundary clamping, color mappings, and custom thresholds...")
    frame_shape = (480, 640)

    # Bounding box exceeding frame boundaries
    out_of_bounds = [
        {"label": "fire", "confidence": 0.95, "bbox": (-50, -50, 700, 550)}
    ]
    res_oob = estimate_severity(out_of_bounds, frame_shape)
    # Fully covering clamped area should equal 1.0 (100%)
    assert res_oob.fire_ratio == 1.0
    assert res_oob.level == SeverityLevel.HIGH.value

    # Color mapping checks
    estimator = SeverityEstimator()
    assert estimator.get_color("HIGH") == (0, 0, 255)
    assert estimator.get_color("MEDIUM") == (0, 140, 255)
    assert estimator.get_color("LOW") == (0, 215, 255)
    assert estimator.get_color("NONE") == (128, 128, 128)

    # Custom threshold instantiation
    custom_estimator = SeverityEstimator(low_threshold=0.02, medium_threshold=0.10, high_threshold=0.30)
    res_custom = custom_estimator.estimate([{"label": "fire", "bbox": (100, 100, 150, 150)}], (480, 640))
    # 50x50 = 2500 / 307200 = 0.8% < 2% custom threshold -> LOW (since fire_ratio > 0)
    assert res_custom.level == SeverityLevel.LOW.value
    print("      -> Boundary clamping and custom thresholds validated.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 4 VERIFICATION SUITE")
    print("=" * 60)
    test_severity_none()
    test_severity_low()
    test_severity_medium()
    test_severity_high()
    test_edge_cases_and_clamping()
    print("=" * 60)
    print("PHASE 4 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)
