"""
src/config.py
Configuration module for the AI Smart Smoke & Fire Detection System.
Loads environment variables and provides centralized configuration.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load .env if present
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Model Settings
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
FALLBACK_MODEL_PATH = os.getenv("FALLBACK_MODEL_PATH", "models/yolov8n.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.50"))

# Temporal Filter Settings (False Alarm Reduction)
CONFIRM_FRAMES = int(os.getenv("CONFIRM_FRAMES", "5"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))

# Severity Estimation Thresholds (fraction of frame area)
SEVERITY_LOW_THRESHOLD = float(os.getenv("SEVERITY_LOW_THRESHOLD", "0.01"))      # 1% of frame
SEVERITY_MEDIUM_THRESHOLD = float(os.getenv("SEVERITY_MEDIUM_THRESHOLD", "0.05"))  # 5% of frame
SEVERITY_HIGH_THRESHOLD = float(os.getenv("SEVERITY_HIGH_THRESHOLD", "0.15"))     # 15% of frame

# Alert Settings
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "30"))
LOCAL_ALARM_ENABLED = os.getenv("LOCAL_ALARM_ENABLED", "True").lower() in ("true", "1", "yes")
TELEGRAM_ALERT_ENABLED = os.getenv("TELEGRAM_ALERT_ENABLED", "False").lower() in ("true", "1", "yes")

# Telegram Credentials
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Database & Storage Settings
DB_PATH = os.getenv("DB_PATH", "events.db")
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
ASSETS_DIR = "assets"
ALARM_SOUND_PATH = os.path.join(ASSETS_DIR, "alarm.wav")

# Camera Sources Configuration
# Supports webcam indices (e.g. 0), local video paths, or RTSP streams
CAMERAS = {
    "Webcam 0": 0,
    "Fire Video": "videos/fire_test.mp4",
    "Smoke Video": "videos/smoke_test.mp4"
}

# Physical Location Metadata for Cameras
CAMERA_LOCATIONS = {
    "Webcam 0": "Building A - Reception",
    "Fire Video": "Server Room Rack B",
    "Smoke Video": "Chemical Storage Warehouse"
}

def get_camera_location(camera_name: str) -> str:
    """Return configured location for a camera, falling back to camera name."""
    return CAMERA_LOCATIONS.get(camera_name, camera_name)

# Ensure runtime directories exist
def ensure_directories():
    """Ensure runtime folders exist upon initialization."""
    for folder in [SNAPSHOT_DIR, "models", "data", "videos", "assets"]:
        target_path = BASE_DIR / folder
        target_path.mkdir(parents=True, exist_ok=True)

ensure_directories()
