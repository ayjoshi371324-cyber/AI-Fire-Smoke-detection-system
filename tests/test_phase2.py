"""
tests/test_phase2.py
Verification test suite for Phase 2 - Camera / Video Input.
Tests webcam capture, fire test video, smoke test video, error handling, and CameraManager.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from src.camera_manager import VideoSource, CameraManager
from src import config


def test_fire_video():
    """Verify frame reading, properties, and looping on fire test video."""
    print("[1/5] Testing Fire Video source...")
    video_path = str(BASE_DIR / "videos" / "fire_test.mp4")
    source = VideoSource(source=video_path, name="Fire Video", loop=True)

    assert source.open(), "Failed to open fire test video"
    props = source.get_properties()
    assert props["is_opened"] is True
    assert props["width"] == 640
    assert props["height"] == 480
    assert props["fps"] > 0
    print(f"      - Properties: {props['width']}x{props['height']} @ {props['fps']} FPS, {props['total_frames']} frames")

    # Read several frames
    for i in range(5):
        success, frame = source.read()
        assert success is True, f"Failed to read frame {i+1}"
        assert isinstance(frame, np.ndarray), "Frame must be a numpy ndarray"
        assert frame.shape == (480, 640, 3), f"Unexpected frame shape: {frame.shape}"

    print(f"      - Successfully read {source.frame_count} frames")
    source.release()
    assert not source._is_opened
    print("      -> Fire Video test passed.")


def test_smoke_video():
    """Verify frame reading on smoke test video."""
    print("[2/5] Testing Smoke Video source...")
    video_path = str(BASE_DIR / "videos" / "smoke_test.mp4")
    with VideoSource(source=video_path, name="Smoke Video") as source:
        props = source.get_properties()
        assert props["is_opened"] is True
        print(f"      - Properties: {props['width']}x{props['height']} @ {props['fps']} FPS")

        success, frame = source.read()
        assert success is True, "Failed to read first smoke frame"
        assert frame.shape == (480, 640, 3)

    print("      -> Smoke Video test passed (context manager release verified).")


def test_webcam():
    """Verify physical/virtual webcam capture."""
    print("[3/5] Testing Webcam 0 source...")
    source = VideoSource(source=0, name="Webcam 0")
    opened = source.open()
    if opened:
        props = source.get_properties()
        print(f"      - Webcam opened: {props['width']}x{props['height']} @ {props['fps']} FPS")
        success, frame = source.read()
        if success and frame is not None:
            assert isinstance(frame, np.ndarray)
            print(f"      - Successfully captured live frame with shape {frame.shape}")
        else:
            print("      - Warning: Webcam opened but frame grab timed out/empty (acceptable in headless/CI).")
        source.release()
        print("      -> Webcam test passed.")
    else:
        print("      - Notice: Webcam 0 could not be opened on this hardware. Fallback handled safely.")
        print("      -> Webcam error handling passed.")


def test_error_handling():
    """Verify error handling on non-existent or invalid video sources."""
    print("[4/5] Testing error handling on invalid sources...")
    # 1. Non-existent file
    invalid_file = VideoSource(source="videos/non_existent_file_xyz.mp4", name="Ghost File")
    assert not invalid_file.open(), "Non-existent file should return False on open"
    success, frame = invalid_file.read()
    assert success is False and frame is None, "Read on invalid source must return (False, None)"
    invalid_file.release()

    # 2. Invalid webcam index
    invalid_cam = VideoSource(source=999, name="Ghost Cam")
    assert not invalid_cam.open(), "Invalid webcam index 999 should return False on open"
    success, frame = invalid_cam.read()
    assert success is False and frame is None
    invalid_cam.release()
    print("      -> Error handling on invalid sources passed.")


def test_camera_manager():
    """Verify CameraManager coordinating multiple sources from config."""
    print("[5/5] Testing CameraManager multi-source coordination...")
    manager = CameraManager.from_config()
    configured_cameras = manager.list_cameras()
    print(f"      - Configured cameras: {list(configured_cameras.keys())}")
    assert "Fire Video" in configured_cameras
    assert "Smoke Video" in configured_cameras

    # Read from Fire Video via manager
    success, frame = manager.read_frame("Fire Video")
    assert success is True
    assert frame is not None and frame.shape == (480, 640, 3)
    print("      - Read frame from 'Fire Video' via CameraManager")

    # Read from Smoke Video via manager
    success, frame = manager.read_frame("Smoke Video")
    assert success is True
    assert frame is not None and frame.shape == (480, 640, 3)
    print("      - Read frame from 'Smoke Video' via CameraManager")

    # Clean release
    manager.release_all()
    print("      -> CameraManager multi-source test passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 2 VERIFICATION SUITE")
    print("=" * 60)
    test_fire_video()
    test_smoke_video()
    test_webcam()
    test_error_handling()
    test_camera_manager()
    print("=" * 60)
    print("PHASE 2 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)
