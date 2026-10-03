"""
tests/test_phase10.py
Verification test suite for Phase 10 - Custom Model Integration.
Validates loading of models/best.pt, custom class labels (Fire/Smoke),
standardized detection output contracts, false-positive resistance on hard negatives,
and full pipeline integration with the custom-trained YOLO model.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.detector import YOLODetector
from src.camera_manager import CameraManager
from main import FireDetectionPipeline
from src import config


def test_custom_model_loading_and_metadata():
    """Verify that models/best.pt is loaded as the primary custom model."""
    print("[1/6] Testing models/best.pt loading and class metadata...")

    best_pt_path = BASE_DIR / "models" / "best.pt"
    assert best_pt_path.exists(), f"Expected custom model file at {best_pt_path}"

    detector = YOLODetector(model_path=str(best_pt_path))
    info = detector.get_info()

    print(f"      - Loaded model path: {info['model_path']}")
    print(f"      - Is custom model:   {info['is_custom_model']}")
    print(f"      - Is fire model:     {info['is_fire_model']}")
    print(f"      - Classes count:     {info['num_classes']}")
    print(f"      - Class names:       {info['class_names']}")

    assert info["is_custom_model"] is True, "Must identify as custom model"
    assert info["is_fire_model"] is True, "Must identify as fire/smoke detection model"
    assert info["num_classes"] == 2, f"Expected 2 classes, got {info['num_classes']}"

    # Verify Fire and Smoke classes are present
    classes_lower = [c.lower() for c in info["class_names"]]
    assert "fire" in classes_lower, f"'fire' class not found in {info['class_names']}"
    assert "smoke" in classes_lower, f"'smoke' class not found in {info['class_names']}"

    print("      -> Custom model loading and metadata verified.")


def test_standardized_detection_output_contract():
    """Verify that custom model preserves the exact standardized detection contract."""
    print("[2/6] Testing detector standardized output contract...")

    detector = YOLODetector()

    # Create dummy test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)

    detections = detector.detect(frame, conf=0.1)
    assert isinstance(detections, list), "Detector must return a list"

    # Test contract conformance on any detections
    for det in detections:
        assert "label" in det, "Detection missing 'label' key"
        assert "confidence" in det, "Detection missing 'confidence' key"
        assert "bbox" in det, "Detection missing 'bbox' key"

        assert isinstance(det["label"], str)
        assert isinstance(det["confidence"], float)
        assert 0.0 <= det["confidence"] <= 1.0

        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 <= 640 and 0 <= x2 <= 640
        assert 0 <= y1 <= 480 and 0 <= y2 <= 480

    print("      -> Standardized output contract verified.")


def test_hard_negatives_and_low_light_scenes():
    """Test custom model against non-fire scenes (dark scenes, pure colors, blank frames)."""
    print("[3/6] Testing false alarm resistance on hard negatives...")

    detector = YOLODetector(conf_threshold=0.5)

    # 1. Pitch black / low-light scene
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    black_dets = detector.detect(black_frame)
    assert len(black_dets) == 0, f"Expected 0 detections on blank black frame, got {len(black_dets)}"

    # 2. Pure white / overexposed scene
    white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    white_dets = detector.detect(white_frame)
    assert len(white_dets) == 0, f"Expected 0 detections on white frame, got {len(white_dets)}"

    # 3. Ambient indoor lighting (gray / brown office tone)
    ambient_frame = np.full((480, 640, 3), (120, 110, 100), dtype=np.uint8)
    ambient_dets = detector.detect(ambient_frame)
    assert len(ambient_dets) == 0, f"Expected 0 detections on ambient gray frame, got {len(ambient_dets)}"

    print("      -> False alarm resistance on hard negatives verified.")


def test_drawing_and_visual_annotations():
    """Verify detector draw_detections renders correctly on images."""
    print("[4/6] Testing bounding box and label rendering with custom model...")

    detector = YOLODetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    synthetic_detections = [
        {"label": "fire", "confidence": 0.94, "bbox": (50, 50, 200, 200)},
        {"label": "smoke", "confidence": 0.82, "bbox": (250, 100, 450, 350)}
    ]

    annotated = detector.draw_detections(frame, synthetic_detections, copy=True)
    assert annotated is not None
    assert annotated.shape == frame.shape

    # Frame should have non-zero pixels due to drawn boxes and labels
    assert np.any(annotated > 0), "Rendered frame must contain drawn annotations"

    print("      -> Annotation rendering verified.")


def test_fallback_resilience():
    """Verify that detector falls back gracefully when a nonexistent path is given."""
    print("[5/6] Testing model fallback mechanism when custom weights missing...")

    # Point to nonexistent path
    nonexistent = BASE_DIR / "models" / "nonexistent_model_xyz.pt"
    detector = YOLODetector(model_path=str(nonexistent))
    info = detector.get_info()

    assert detector.model is not None, "Detector should load fallback model"
    print(f"      - Graceful fallback model loaded: {info['model_path']}")
    print("      -> Fallback resilience verified.")


def test_pipeline_integration_with_custom_model():
    """Run full FireDetectionPipeline with the custom trained model on video."""
    print("[6/6] Testing FireDetectionPipeline integration with custom model...")

    pipeline = FireDetectionPipeline(
        detector=YOLODetector()
    )
    assert pipeline.detector.is_custom_model is True

    # Process 15 frames from Fire Video in headless mode
    stats = pipeline.run("Fire Video", max_frames=15, headless=True)

    print(f"      - Pipeline processed: {stats.frames_processed} frames at {stats.average_fps} FPS")
    assert stats.frames_processed == 15
    assert stats.average_fps > 0

    print("      -> Pipeline integration with custom model verified.")


def run_all_tests():
    """Run all Phase 10 verification tests."""
    print("=" * 60)
    print("RUNNING PHASE 10 VERIFICATION SUITE (CUSTOM MODEL INTEGRATION)")
    print("=" * 60)

    test_custom_model_loading_and_metadata()
    test_standardized_detection_output_contract()
    test_hard_negatives_and_low_light_scenes()
    test_drawing_and_visual_annotations()
    test_fallback_resilience()
    test_pipeline_integration_with_custom_model()

    print("=" * 60)
    print("PHASE 10 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
