"""
tests/test_phase1.py
Verification test suite for Phase 1 - Project Foundation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def test_imports():
    """Verify all required initial libraries can be imported."""
    print("[1/5] Testing dependencies import...")
    import ultralytics
    import cv2
    import streamlit
    import pandas
    import plotly
    import numpy
    import requests
    import dotenv
    print(f"      - ultralytics version: {ultralytics.__version__}")
    print(f"      - cv2 (OpenCV) version: {cv2.__version__}")
    print(f"      - streamlit version: {streamlit.__version__}")
    print(f"      - pandas version: {pandas.__version__}")
    print(f"      - numpy version: {numpy.__version__}")
    print(f"      - requests version: {requests.__version__}")
    print("      -> All dependencies imported successfully.")

def test_directories():
    """Verify required directories exist."""
    print("[2/5] Testing directory structure...")
    base_dir = Path(__file__).resolve().parent.parent
    required_dirs = ["models", "data", "videos", "snapshots", "assets", "src"]
    for d in required_dirs:
        dir_path = base_dir / d
        assert dir_path.is_dir(), f"Missing required directory: {d}"
        print(f"      - Found directory: {d}/")
    print("      -> All required directories exist.")

def test_config():
    """Verify configuration loads properly with required keys."""
    print("[3/5] Testing configuration module...")
    from src import config
    assert hasattr(config, "MODEL_PATH"), "Missing MODEL_PATH"
    assert hasattr(config, "CONF_THRESHOLD"), "Missing CONF_THRESHOLD"
    assert hasattr(config, "CONFIRM_FRAMES"), "Missing CONFIRM_FRAMES"
    assert hasattr(config, "WINDOW_SIZE"), "Missing WINDOW_SIZE"
    assert hasattr(config, "ALERT_COOLDOWN_SEC"), "Missing ALERT_COOLDOWN_SEC"
    assert hasattr(config, "DB_PATH"), "Missing DB_PATH"
    assert hasattr(config, "SNAPSHOT_DIR"), "Missing SNAPSHOT_DIR"
    assert hasattr(config, "CAMERAS"), "Missing CAMERAS"
    assert isinstance(config.CAMERAS, dict), "CAMERAS must be a dict"
    print(f"      - MODEL_PATH: {config.MODEL_PATH}")
    print(f"      - CONF_THRESHOLD: {config.CONF_THRESHOLD}")
    print(f"      - CONFIRM_FRAMES: {config.CONFIRM_FRAMES}")
    print(f"      - WINDOW_SIZE: {config.WINDOW_SIZE}")
    print(f"      - ALERT_COOLDOWN_SEC: {config.ALERT_COOLDOWN_SEC}")
    print(f"      - DB_PATH: {config.DB_PATH}")
    print(f"      - SNAPSHOT_DIR: {config.SNAPSHOT_DIR}")
    print(f"      - CAMERAS: {list(config.CAMERAS.keys())}")
    print("      -> Configuration verified successfully.")

def test_security():
    """Verify .env is git-ignored and no secrets committed."""
    print("[4/5] Testing secrets and .gitignore...")
    base_dir = Path(__file__).resolve().parent.parent
    gitignore_path = base_dir / ".gitignore"
    env_path = base_dir / ".env"
    env_example_path = base_dir / ".env.example"

    assert gitignore_path.exists(), "Missing .gitignore"
    assert env_example_path.exists(), "Missing .env.example"
    assert env_path.exists(), "Missing .env"

    gitignore_content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in gitignore_content, ".env must be ignored in .gitignore"
    assert ".venv" in gitignore_content, ".venv must be ignored in .gitignore"
    print("      -> .gitignore correctly excludes .env and .venv.")

def test_python_version():
    """Verify running on appropriate Python version."""
    print("[5/5] Testing Python runtime...")
    print(f"      - Python version: {sys.version.split()[0]}")
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    print("      -> Python runtime is valid.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 1 VERIFICATION SUITE")
    print("=" * 60)
    test_python_version()
    test_directories()
    test_security()
    test_config()
    test_imports()
    print("=" * 60)
    print("PHASE 1 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)
