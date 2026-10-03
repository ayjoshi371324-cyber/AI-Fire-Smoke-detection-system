"""
tests/test_phase13.py
Verification test suite for Phase 13 - Final Polish + Documentation.
Validates master directory structure conformance, code cleanliness and compilation,
module imports, and comprehensive documentation in README.md covering all 12 required sections.
"""

import sys
import os
import py_compile
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_readme_comprehensive_sections():
    """Verify README.md exists and contains all 12 mandatory specification sections."""
    print("[1/5] Testing README.md documentation completeness...")
    readme_path = BASE_DIR / "README.md"
    assert readme_path.exists(), "README.md file is missing."

    content = readme_path.read_text(encoding="utf-8")
    assert len(content) > 1000, "README.md content is too brief."

    required_topics = [
        ("Project overview", "1. Project Overview"),
        ("Features", "2. Features"),
        ("Tech stack", "3. Technology Stack"),
        ("Installation", "5. Installation & Setup"),
        ("Dataset information", "6. Dataset Information"),
        ("Model training instructions", "7. Model Training Instructions"),
        ("How to add best.pt", "8. How to Add `best.pt`"),
        ("How to run detection", "9. How to Run Detection Pipeline"),
        ("How to run dashboard", "10. How to Run the Dashboard"),
        ("Configuration", "Configure Environment Variables"),
        ("Testing results", "11. Testing & Benchmark Results"),
        ("Known limitations", "12. Known Limitations")
    ]

    for topic_desc, phrase in required_topics:
        assert phrase.lower() in content.lower(), f"Missing required topic '{topic_desc}' in README.md"
        print(f"      - Verified section: '{topic_desc}'")

    print("      -> README.md verified with all 12 mandatory sections.")


def test_master_project_structure():
    """Verify project structure conforms strictly to Section 3 & 17 specifications."""
    print("[2/5] Testing master project structure conformance...")

    required_folders = ["models", "data", "videos", "snapshots", "assets", "src", "tests", "scripts"]
    for folder in required_folders:
        folder_path = BASE_DIR / folder
        assert folder_path.is_dir(), f"Required folder '{folder}' does not exist."
        print(f"      - Directory verified: '{folder}/'")

    required_files = [
        "src/__init__.py",
        "src/config.py",
        "src/camera_manager.py",
        "src/detector.py",
        "src/severity.py",
        "src/temporal_filter.py",
        "src/alerts.py",
        "src/database.py",
        "main.py",
        "dashboard.py",
        "requirements.txt",
        ".env",
        ".gitignore",
        "README.md"
    ]

    for file_rel in required_files:
        fp = BASE_DIR / file_rel
        assert fp.is_file(), f"Required file '{file_rel}' does not exist."
        print(f"      - File verified: '{file_rel}'")

    print("      -> Master project structure verified.")


def test_code_cleanliness_and_compilation():
    """Verify that all application and source Python modules compile without syntax errors."""
    print("[3/5] Testing Python source code compilation across the project...")

    python_files = list((BASE_DIR / "src").glob("*.py")) + [
        BASE_DIR / "main.py",
        BASE_DIR / "dashboard.py"
    ] + list((BASE_DIR / "scripts").glob("*.py"))

    for py_file in python_files:
        compiled = py_compile.compile(str(py_file), doraise=True)
        assert compiled is not None, f"Failed to compile {py_file}"
        print(f"      - Bytecode compiled: {py_file.name}")

    print("      -> All Python modules compiled cleanly.")


def test_core_package_exports():
    """Verify that all components can be imported cleanly without broken dependencies."""
    print("[4/5] Testing modular imports and component resolution...")

    from src import config
    from src.camera_manager import CameraManager, VideoSource, IncidentEvent
    from src.detector import YOLODetector
    from src.severity import SeverityEstimator, estimate_severity
    from src.temporal_filter import TemporalFilter, MultiCameraTemporalFilter
    from src.alerts import AlertManager, LocalAlarm, TelegramAlerter
    from src.database import DatabaseManager
    from src.benchmark import SystemEvaluator
    from main import FireDetectionPipeline

    # Instantiate instances to verify dependency injection
    cam_mgr = CameraManager.from_config()
    db_mgr = DatabaseManager.from_config()
    alert_mgr = AlertManager.from_config()
    pipeline = FireDetectionPipeline(
        camera_manager=cam_mgr,
        alert_manager=alert_mgr,
        database_manager=db_mgr
    )
    assert pipeline is not None
    print("      -> All core modular components resolved and instantiated.")


def test_end_to_end_system_sanity():
    """Verify an end-to-end execution pass of the full integrated system."""
    print("[5/5] Testing end-to-end system health sanity check...")

    from main import FireDetectionPipeline
    import numpy as np

    pipeline = FireDetectionPipeline()
    test_frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    # Process frame
    res = pipeline.process_frame(test_frame, camera_name="Fire Video", frame_idx=1)
    assert res is not None
    assert res.annotated_frame is not None
    assert res.severity is not None
    assert res.temporal_status is not None
    print(f"      - End-to-end frame processed cleanly: {res.camera_name} (FPS: {res.fps:.1f})")

    print("      -> End-to-end system health sanity verified.")


def run_all_tests():
    """Run all Phase 13 verification tests."""
    print("=" * 60)
    print("RUNNING PHASE 13 VERIFICATION SUITE (POLISH + DOCUMENTATION)")
    print("=" * 60)

    test_readme_comprehensive_sections()
    test_master_project_structure()
    test_code_cleanliness_and_compilation()
    test_core_package_exports()
    test_end_to_end_system_sanity()

    print("=" * 60)
    print("PHASE 13 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
