# AI Smart Smoke & Fire Detection System

An AI-driven real-time smoke and fire detection platform utilizing computer vision (YOLO), multi-frame temporal confirmation, visual severity estimation, multi-channel alerting (local audible alarm & Telegram notifications), SQLite event logging, and an interactive Streamlit monitoring dashboard.

---

## Master Architecture & Roadmap

- **Phase 1: Project Foundation** *(Current)*
- **Phase 2: Camera / Video Input Pipeline**
- **Phase 3: YOLO Detection Interface**
- **Phase 4: Severity Estimation**
- **Phase 5: Temporal Confirmation & False Alarm Filtering**
- **Phase 6: Multi-Camera & Location Management**
- **Phase 7: Alert Systems (Local Windows Alarm & Telegram API)**
- **Phase 8: SQLite Database & Snapshot Logging**
- **Phase 9: Main Application Pipeline Integration**
- **Phase 10: Custom YOLO Model Integration (`models/best.pt`)**
- **Phase 11: System Testing & Performance Optimization**
- **Phase 12: Streamlit Monitoring Dashboard**
- **Phase 13: Final Polish & Comprehensive Documentation**

---

## Directory Layout

```text
fire-detection/
│
├── models/             # YOLO model weights (models/best.pt, yolov8n.pt fallback)
├── data/               # Reference datasets and hard-negative test samples
├── videos/             # Test video clips (fire, smoke, cooking, steam, false positives)
├── snapshots/          # Automated capture of confirmed fire/smoke incidents
├── assets/             # Assets such as alarm sounds (assets/alarm.wav)
│
├── src/                # Core modular business logic
│   ├── __init__.py     # Package initialization
│   ├── config.py       # Centralized settings & environment loader
│   ├── camera_manager.py
│   ├── detector.py
│   ├── severity.py
│   ├── temporal_filter.py
│   ├── alerts.py
│   └── database.py
│
├── main.py             # Backend orchestrator pipeline
├── dashboard.py        # Streamlit monitoring dashboard
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variable template
├── .env                # Local secrets (git ignored)
├── .gitignore          # Repository exclusions
└── README.md
```

---

## Getting Started (Phase 1 Foundation)

### 1. Environment Setup

Create and activate a virtual environment (Python 3.12 recommended):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Settings

Copy `.env.example` to `.env` and configure your local settings:

```powershell
Copy-Item .env.example .env
```

### 4. Verify Phase 1 Installation

Run the configuration verification test:

```powershell
python -m src.config
```
