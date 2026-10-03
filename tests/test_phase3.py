"""
tests/test_phase3.py
Verification test suite for Phase 3 - YOLO Detection Interface.
Verifies model loading, standardized detection output schema, rendering, and video frame inference.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import cv2
from src.detector import YOLODetector
from src.camera_manager import VideoSource


def test_detector_initialization():
    """Verify detector loads temporary/fallback model gracefully."""
    print("[1/5] Testing detector initialization and model loading...")
    detector = YOLODetector()
    info = detector.get_info()
    print(f"      - Loaded model: {info['model_path']}")
    print(f"      - Is custom model: {info['is_custom_model']}")
    print(f"      - Available classes: {info['num_classes']}")
    print(f"      - Confidence threshold: {info['conf_threshold']}")

    assert info["num_classes"] > 0, "Model should have classes loaded"
    assert detector.model is not None, "YOLO model instance must exist"
    print("      -> Detector initialization passed.")


def test_inference_and_schema():
    """Verify detection output contract [label, confidence, bbox]."""
    print("[2/5] Testing inference and standardized output contract...")
    detector = YOLODetector(conf_threshold=0.25)

    # Create a synthetic image with a distinct object (e.g. cup/bottle or simulated rectangle)
    test_image = np.full((480, 640, 3), 200, dtype=np.uint8)
    # Draw a colored shape
    cv2.circle(test_image, (320, 240), 80, (0, 0, 255), -1)

    detections = detector.detect(test_image)
    print(f"      - Detections returned: {len(detections)}")
    assert isinstance(detections, list), "Detections must be returned as a list"

    # If any detections returned, verify every item strictly matches the schema
    for det in detections:
        assert "label" in det and isinstance(det["label"], str)
        assert "confidence" in det and isinstance(det["confidence"], float)
        assert "bbox" in det and isinstance(det["bbox"], tuple) and len(det["bbox"]) == 4
        x1, y1, x2, y2 = det["bbox"]
        assert all(isinstance(v, int) for v in (x1, y1, x2, y2))
        assert 0 <= det["confidence"] <= 1.0

    print("      -> Output format strictly matches [{'label': str, 'confidence': float, 'bbox': (x1, y1, x2, y2)}].")


def test_drawing_annotations():
    """Verify drawing detections on frame."""
    print("[3/5] Testing bounding box and badge rendering...")
    detector = YOLODetector()
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    sample_detections = [
        {"label": "fire", "confidence": 0.94, "bbox": (100, 100, 250, 300)},
        {"label": "smoke", "confidence": 0.88, "bbox": (300, 80, 500, 260)}
    ]

    annotated = detector.draw_detections(frame, sample_detections, copy=True)
    assert annotated is not None
    assert annotated.shape == frame.shape
    # Ensure drawing produced changes on the canvas
    assert not np.array_equal(annotated, frame), "Annotated frame must have rendered bounding boxes"

    # Save verification snapshot
    snapshots_dir = BASE_DIR / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    out_path = snapshots_dir / "phase3_test_render.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"      - Saved test annotated render to {out_path.name}")
    print("      -> Drawing annotations passed.")


def test_video_feed_inference():
    """Verify running detector directly on video frames from VideoSource."""
    print("[4/5] Testing detector on VideoSource frames (fire_test.mp4)...")
    detector = YOLODetector(conf_threshold=0.30)
    video_path = str(BASE_DIR / "videos" / "fire_test.mp4")

    with VideoSource(source=video_path, name="Fire Video") as source:
        frames_processed = 0
        for frame in source:
            detections = detector.detect(frame)
            annotated = detector.draw_detections(frame, detections)
            frames_processed += 1
            if frames_processed >= 5:
                break

    print(f"      - Successfully processed {frames_processed} video frames through detector pipeline.")
    print("      -> Video feed inference passed.")


def test_empty_and_corrupt_frame_handling():
    """Verify detector handles empty/corrupt frames safely without raising unhandled exceptions."""
    print("[5/5] Testing resilience against empty/corrupted frames...")
    detector = YOLODetector()

    assert detector.detect(None) == [], "None frame should return empty list"
    assert detector.detect(np.array([])) == [], "Empty numpy array should return empty list"

    # 1x1 dummy frame
    tiny_frame = np.zeros((1, 1, 3), dtype=np.uint8)
    res = detector.detect(tiny_frame)
    assert isinstance(res, list)

    print("      -> Frame resilience testing passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 3 VERIFICATION SUITE")
    print("=" * 60)
    test_detector_initialization()
    test_inference_and_schema()
    test_drawing_annotations()
    test_video_feed_inference()
    test_empty_and_corrupt_frame_handling()
    print("=" * 60)
    print("PHASE 3 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)
