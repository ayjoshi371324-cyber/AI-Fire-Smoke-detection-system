"""
tests/test_phase6.py
Verification test suite for Phase 6 - Location / Camera Management.
Validates camera-to-location mapping, IncidentEvent data contracts,
and multi-camera location isolation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.camera_manager import CameraManager, VideoSource, IncidentEvent
from src import config


def test_location_configuration_resolution():
    """Verify camera to location resolution from configuration."""
    print("[1/5] Testing camera-to-location mapping resolution...")
    manager = CameraManager.from_config()
    cameras_info = manager.list_cameras()

    print(f"      - Configured cameras and locations:")
    for name, info in cameras_info.items():
        print(f"        * '{name}' -> location: '{info['location']}', source: {info['source']}")
        assert "location" in info and len(info["location"]) > 0

    assert manager.get_location("Fire Video") == "Server Room Rack B"
    assert manager.get_location("Smoke Video") == "Chemical Storage Warehouse"
    assert manager.get_location("Webcam 0") == "Building A - Reception"

    # Unmapped camera fallback to camera name
    fallback_loc = manager.get_location("Unmapped Sensor 9")
    assert fallback_loc == "Unmapped Sensor 9"
    print("      -> Location resolution verified.")


def test_incident_event_contract():
    """Verify IncidentEvent contains both camera and location attributes."""
    print("[2/5] Testing IncidentEvent contract for camera and location tracking...")
    manager = CameraManager.from_config()

    event = manager.create_incident_event(
        camera_name="Fire Video",
        label="fire",
        confidence=0.94,
        severity="HIGH",
        bbox=(100, 100, 300, 300)
    )

    assert hasattr(event, "camera") and event.camera == "Fire Video"
    assert hasattr(event, "location") and event.location == "Server Room Rack B"
    assert hasattr(event, "label") and event.label == "fire"
    assert hasattr(event, "confidence") and event.confidence == 0.94
    assert hasattr(event, "severity") and event.severity == "HIGH"
    assert hasattr(event, "timestamp") and len(event.timestamp) > 0

    data = event.to_dict()
    assert data["camera"] == "Fire Video"
    assert data["location"] == "Server Room Rack B"
    assert data["severity"] == "HIGH"

    print(f"      - Generated event: {data}")
    print("      -> IncidentEvent contract verified.")


def test_video_source_location_metadata():
    """Verify VideoSource exposes location in properties."""
    print("[3/5] Testing VideoSource location property binding...")
    source = VideoSource(
        source="videos/fire_test.mp4",
        name="Fire Video",
        location="Server Room Rack B"
    )
    props = source.get_properties()
    assert "location" in props
    assert props["location"] == "Server Room Rack B"
    assert props["name"] == "Fire Video"
    source.release()
    print("      -> VideoSource location binding verified.")


def test_dynamic_camera_registration():
    """Verify dynamically registering cameras with custom locations."""
    print("[4/5] Testing dynamic camera registration with custom location...")
    manager = CameraManager()
    manager.register_camera(
        name="Kitchen Cam",
        source="videos/fire_test.mp4",
        location="Main Floor Kitchen"
    )

    assert "Kitchen Cam" in manager.cameras_config
    assert manager.get_location("Kitchen Cam") == "Main Floor Kitchen"

    event = manager.create_incident_event(
        camera_name="Kitchen Cam",
        label="smoke",
        confidence=0.88,
        severity="MEDIUM"
    )
    assert event.camera == "Kitchen Cam"
    assert event.location == "Main Floor Kitchen"
    print("      -> Dynamic camera registration verified.")


def test_multi_camera_location_isolation():
    """Verify distinct locations are preserved across multiple events."""
    print("[5/5] Testing multi-camera location isolation across simultaneous events...")
    manager = CameraManager.from_config()

    event_fire = manager.create_incident_event("Fire Video", "fire", 0.95, "HIGH")
    event_smoke = manager.create_incident_event("Smoke Video", "smoke", 0.85, "LOW")

    assert event_fire.location != event_smoke.location
    assert event_fire.location == "Server Room Rack B"
    assert event_smoke.location == "Chemical Storage Warehouse"
    print(f"      - Event 1: [{event_fire.camera}] at '{event_fire.location}'")
    print(f"      - Event 2: [{event_smoke.camera}] at '{event_smoke.location}'")
    print("      -> Multi-camera location isolation verified.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 6 VERIFICATION SUITE")
    print("=" * 60)
    test_location_configuration_resolution()
    test_incident_event_contract()
    test_video_source_location_metadata()
    test_dynamic_camera_registration()
    test_multi_camera_location_isolation()
    print("=" * 60)
    print("PHASE 6 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)
