# AI Smart Smoke & Fire Detection System

## Antigravity Implementation Plan

> **Purpose:** This file is the master implementation instruction for
> Antigravity.\
> Follow the phases in order. Do not skip a phase. Do not build all
> features at once.

------------------------------------------------------------------------

# 1. PROJECT GOAL

Build an AI-based Smart Smoke & Fire Detection System that:

-   Detects **fire** and **smoke** from camera/video feeds.
-   Uses YOLO for object detection.
-   Shows bounding boxes and confidence.
-   Confirms detections across multiple frames to reduce false alarms.
-   Estimates severity as **NONE / LOW / MEDIUM / HIGH**.
-   Identifies the camera/location where detection occurred.
-   Triggers local and Telegram alerts.
-   Stores confirmed events and snapshots in SQLite.
-   Provides a professional monitoring dashboard/frontend.
-   Supports a custom-trained YOLO model through `models/best.pt`.

The custom ML model will be trained **separately**, preferably in Google
Colab.

------------------------------------------------------------------------

# 2. MOST IMPORTANT DEVELOPMENT RULE

## Implement phases in this exact order

``` text
PHASE 1  → Project Foundation
PHASE 2  → Camera / Video Input
PHASE 3  → YOLO Detection Interface
PHASE 4  → Severity Estimation
PHASE 5  → False Alarm Reduction
PHASE 6  → Location / Camera Management
PHASE 7  → Alerts
PHASE 8  → Database + Snapshots
PHASE 9  → Main Application Integration
PHASE 10 → Custom Model Integration
PHASE 11 → Testing + Optimization
PHASE 12 → Frontend / Dashboard
PHASE 13 → Final Polish + Documentation
```

## ML training runs separately

``` text
Google Colab
    ↓
Dataset preparation
    ↓
YOLO training
    ↓
Evaluation
    ↓
best.pt
    ↓
models/best.pt
    ↓
Antigravity project
```

**Do not wait for model training to finish before developing the
application.**

Until the custom model is ready, use a temporary/pretrained YOLO model
only for testing the application pipeline.

------------------------------------------------------------------------

# 3. MASTER PROJECT STRUCTURE

Create and maintain this structure:

``` text
fire-detection/
│
├── models/
│   └── best.pt
│
├── data/
│
├── videos/
│
├── snapshots/
│
├── assets/
│   └── alarm.wav
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── camera_manager.py
│   ├── detector.py
│   ├── severity.py
│   ├── temporal_filter.py
│   ├── alerts.py
│   └── database.py
│
├── main.py
├── dashboard.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

Do not put all logic inside `main.py`.

------------------------------------------------------------------------

# 4. PHASE 1 --- PROJECT FOUNDATION

## IMPLEMENT THIS FIRST

### Goal

Create the basic project structure and development environment.

### Tasks

1.  Inspect the existing project before changing anything.
2.  Preserve existing working code.
3.  Create the required folders.
4.  Create the Python virtual environment if required.
5.  Install required dependencies.
6.  Create `requirements.txt`.
7.  Create `.gitignore`.
8.  Create `.env` for secrets.
9.  Create `src/config.py`.
10. Make sure the application can start without errors.

### Initial dependencies

``` text
ultralytics
opencv-python
streamlit
pandas
plotly
numpy
requests
python-dotenv
```

Use additional libraries only when actually required.

### Initial configuration

`src/config.py` should contain configurable values such as:

``` python
MODEL_PATH = "models/best.pt"

CONF_THRESHOLD = 0.50

CONFIRM_FRAMES = 5
WINDOW_SIZE = 10

ALERT_COOLDOWN_SEC = 30

DB_PATH = "events.db"

SNAPSHOT_DIR = "snapshots"
```

Keep configuration separate from application logic.

### Verification

Before moving to Phase 2:

-   Python environment works.
-   Dependencies install.
-   Project imports successfully.
-   Folder structure exists.
-   Configuration loads without errors.
-   No secrets are committed.

------------------------------------------------------------------------

# 5. PHASE 2 --- CAMERA / VIDEO INPUT

## Goal

Create a reusable camera/video input layer.

### File

``` text
src/camera_manager.py
```

### Must support

-   Webcam.
-   Local video file.
-   Future RTSP/CCTV source.

Example configuration:

``` python
CAMERAS = {
    "Lab 1": 0,
    "Test Video": "videos/fire_test.mp4"
}
```

### Camera manager responsibilities

-   Open source.
-   Read frames.
-   Detect read/open errors.
-   Provide frames to the detection pipeline.
-   Release resources safely.

Do not put YOLO inference inside the camera manager.

### Verification

Test:

1.  Webcam.
2.  Fire video.
3.  Smoke video.

Do not proceed until frame capture works reliably.

------------------------------------------------------------------------

# 6. PHASE 3 --- YOLO DETECTION INTERFACE

## Goal

Create a clean detector module that can later switch from the temporary
model to the custom model without changing the rest of the application.

### File

``` text
src/detector.py
```

### Responsibilities

-   Load YOLO model.
-   Accept a frame.
-   Run inference.
-   Extract detections.
-   Return a consistent data structure.

Expected output:

``` python
[
    {
        "label": "fire",
        "confidence": 0.91,
        "bbox": (x1, y1, x2, y2)
    }
]
```

### Important

The rest of the application must depend on this detection format, not
directly on YOLO internals.

This makes model replacement easy.

### Temporary model

If `models/best.pt` does not exist yet, allow a temporary/pretrained
model for development.

Do not modify the application architecture when `best.pt` becomes
available.

### Verification

Display:

-   Label.
-   Confidence.
-   Bounding box.

Confirm the detector works with a test video/image.

------------------------------------------------------------------------

# 7. PHASE 4 --- SEVERITY ESTIMATION

## Goal

Estimate the severity of detected fire/smoke.

### File

``` text
src/severity.py
```

### Output

``` text
NONE
LOW
MEDIUM
HIGH
```

### Inputs

-   Detection list.
-   Frame dimensions.

### Basic approach

Calculate:

``` text
fire area / frame area
smoke area / frame area
```

Use configurable thresholds.

Example logic:

``` text
No fire/smoke
    → NONE

Small detected region
    → LOW

Medium detected region
    → MEDIUM

Large fire region or significant fire + smoke
    → HIGH
```

Do not claim this is a real-world fire-risk measurement. It is a
project-level visual severity estimate.

### Verification

Test different bounding-box sizes and confirm the severity changes
correctly.

------------------------------------------------------------------------

# 8. PHASE 5 --- FALSE ALARM REDUCTION

## Goal

Prevent a single incorrect detection from triggering an alert.

### File

``` text
src/temporal_filter.py
```

### Layer 1 --- Confidence

Use:

``` text
CONF_THRESHOLD
```

### Layer 2 --- Temporal confirmation

Example:

``` text
5 detections within the last 10 frames
→ confirmed detection
```

### Layer 3 --- Training data

The separately trained model should include hard-negative examples:

-   Steam.
-   Cooking smoke.
-   Dust.
-   Fog.
-   Red/orange objects.
-   Bright lights.

### Verification

Test:

``` text
One-frame detection
→ No immediate alert
```

and:

``` text
Repeated detections
→ Confirmed detection
```

------------------------------------------------------------------------

# 9. PHASE 6 --- LOCATION / CAMERA MANAGEMENT

## Goal

Identify which camera/location produced an event.

Example:

``` python
CAMERAS = {
    "Lab 1": 0,
    "Kitchen": "videos/kitchen.mp4",
    "Warehouse": "videos/warehouse.mp4"
}
```

Every confirmed event should contain:

``` text
camera
location
```

For the initial version, camera name is sufficient as the physical
location identifier.

Optional future feature:

-   Divide frame into a 3×3 grid.
-   Report top-left, center, bottom-right, etc.

Do not implement optional features until the core system works.

------------------------------------------------------------------------

# 10. PHASE 7 --- ALERT SYSTEM

## Goal

Trigger an alert only after a detection is confirmed.

### File

``` text
src/alerts.py
```

### Local alert

Support a Windows-compatible alarm/buzzer.

### Telegram alert

Send a message containing:

``` text
FIRE ALERT

Camera: Kitchen
Detection: Fire
Severity: HIGH
Confidence: 91%
Time: 13:25:32
```

### Optional

Twilio SMS can be added later.

Do not make SMS a blocker for the MVP.

### Security

Never hardcode:

-   Telegram bot token.
-   Chat ID if treated as secret.
-   Twilio credentials.
-   API keys.

Use `.env`.

### Cooldown

Prevent alert spam:

``` text
ALERT_COOLDOWN_SEC = 30
```

### Verification

Test:

-   Local alarm.
-   Telegram.
-   Cooldown.
-   Duplicate-alert prevention.

------------------------------------------------------------------------

# 11. PHASE 8 --- DATABASE + SNAPSHOTS

## Goal

Store confirmed events.

### File

``` text
src/database.py
```

### Database

Use SQLite.

Database:

``` text
events.db
```

### Event fields

``` text
id
timestamp
camera
label
confidence
severity
snapshot
```

### Snapshot behavior

When an alert is confirmed:

1.  Save the frame.
2.  Generate a unique filename.
3.  Store the snapshot path in SQLite.

Example:

``` text
snapshots/
└── 20261003_132530.jpg
```

### Verification

Trigger a test event and verify:

-   Snapshot exists.
-   Database record exists.
-   Timestamp is correct.
-   Camera is correct.
-   Detection label is correct.
-   Confidence is stored.
-   Severity is stored.

------------------------------------------------------------------------

# 12. PHASE 9 --- MAIN APPLICATION INTEGRATION

## Goal

Connect all backend modules.

### File

``` text
main.py
```

### Pipeline

``` text
Camera
   ↓
Frame
   ↓
YOLO Detector
   ↓
Fire / Smoke Detection
   ↓
Temporal Confirmation
   ↓
Severity
   ↓
Location
   ↓
Alert
   ↓
Snapshot
   ↓
Database
```

### `main.py` responsibilities

`main.py` should coordinate modules, not contain all their internal
logic.

It should:

-   Start camera.
-   Get frames.
-   Call detector.
-   Apply temporal filtering.
-   Calculate severity.
-   Draw detections.
-   Trigger alerts.
-   Save snapshots.
-   Log events.

### Verification

Run the complete backend pipeline with a test video.

The system must:

-   Detect.
-   Draw boxes.
-   Confirm.
-   Calculate severity.
-   Alert.
-   Save snapshot.
-   Log event.

------------------------------------------------------------------------

# 13. PHASE 10 --- CUSTOM MODEL INTEGRATION

## Goal

Replace the temporary model with the custom-trained model.

### Model source

The model is trained separately in Google Colab.

Expected file:

``` text
models/best.pt
```

### Important

Do not retrain inside the application.

Do not rewrite the detector.

Only change the configured model path if necessary:

``` text
models/best.pt
```

### Verification

Test the custom model with:

-   Fire videos.
-   Smoke videos.
-   Steam.
-   Cooking.
-   Red objects/lights.
-   Low-light scenes.
-   Distant fire.
-   Different camera angles.

Confirm that the same detector output format still works.

------------------------------------------------------------------------

# 14. PHASE 11 --- TESTING + OPTIMIZATION

## Goal

Measure the complete system.

### Test cases

  Test                Expected
  ------------------- ----------------
  Fire video          Fire detected
  Smoke video         Smoke detected
  Cooking/steam       No false alert
  Red object/light    No false alert
  Low light           Evaluate
  Far-distance fire   Evaluate
  Multiple triggers   Cooldown works

### ML metrics

Record:

-   Precision.
-   Recall.
-   F1.
-   mAP@0.5.
-   Confusion matrix.

### Application metrics

Record:

-   FPS.
-   False-alarm rate.
-   Detection-to-alert delay.

### Optimization

If FPS is low:

-   Reduce image size.
-   Process fewer frames.
-   Consider a smaller YOLO model.

If false positives are high:

-   Tune confidence.
-   Tune temporal confirmation.
-   Add hard-negative training images.

If small fires are missed:

-   Improve training data.
-   Consider larger training image size.
-   Evaluate a larger YOLO model.

Do not optimize without measuring the effect.

------------------------------------------------------------------------

# 15. PHASE 12 --- FRONTEND / DASHBOARD

## IMPORTANT

Only start this phase after the backend detection pipeline is stable.

### Technology

Use:

``` text
Streamlit
Pandas
Plotly
```

### File

``` text
dashboard.py
```

## Dashboard layout

### Header

``` text
AI Smart Smoke & Fire Detection System
```

### System status

Show:

``` text
System: ONLINE
Camera: CONNECTED
AI Model: LOADED
```

### Live monitoring

Show:

-   Current camera/video.
-   Bounding boxes.
-   Fire/smoke labels.
-   Confidence.
-   Severity.
-   Location.

### Summary cards

Show:

``` text
Total Events
Fire Events
Smoke Events
High Severity Events
```

### Charts

Include:

-   Fire vs Smoke.
-   Severity distribution.
-   Events by camera.
-   Events over time.

### Event history

Display:

``` text
Timestamp
Camera
Label
Confidence
Severity
Snapshot
```

### Latest alert

Example:

``` text
HIGH SEVERITY

Location: Kitchen
Detection: Fire
Confidence: 94%
Time: 13:25
```

### Latest snapshot

Display the latest event image.

### Filters

Add where practical:

-   Camera.
-   Detection type.
-   Severity.
-   Date/time.

### UI requirements

The dashboard should be:

-   Clean.
-   Professional.
-   Easy to understand.
-   Suitable for a college project demonstration.
-   Not overloaded with unnecessary components.

Do not sacrifice backend stability for visual effects.

------------------------------------------------------------------------

# 16. PHASE 13 --- FINAL POLISH + DOCUMENTATION

After all functionality works:

## Code

-   Remove unnecessary code.
-   Fix warnings/errors.
-   Improve naming.
-   Add useful comments.
-   Keep modules clean.

## UI

Improve:

-   Spacing.
-   Typography.
-   Cards.
-   Status indicators.
-   Charts.
-   Empty states.
-   Error states.

## Documentation

Update:

``` text
README.md
```

Include:

1.  Project overview.
2.  Features.
3.  Tech stack.
4.  Installation.
5.  Dataset information.
6.  Model training instructions.
7.  How to add `best.pt`.
8.  How to run detection.
9.  How to run dashboard.
10. Configuration.
11. Testing results.
12. Known limitations.

------------------------------------------------------------------------

# 17. FINAL PROJECT STRUCTURE

The final project should approximately look like:

``` text
fire-detection/
│
├── models/
│   └── best.pt
│
├── data/
│
├── videos/
│
├── snapshots/
│
├── assets/
│   └── alarm.wav
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── camera_manager.py
│   ├── detector.py
│   ├── severity.py
│   ├── temporal_filter.py
│   ├── alerts.py
│   └── database.py
│
├── main.py
├── dashboard.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

# 18. DEFINITION OF DONE

The project is complete only when this complete flow works:

``` text
Camera
   ↓
OpenCV
   ↓
Custom YOLO Model
   ↓
Fire / Smoke Detection
   ↓
Temporal Confirmation
   ↓
Severity
   ↓
Camera / Location
   ↓
Alert
   ↓
Snapshot
   ↓
SQLite
   ↓
Dashboard
```

The dashboard must show the resulting event information.

------------------------------------------------------------------------

# 19. ANTIGRAVITY OPERATING RULES

Always follow these rules:

1.  **Implement only one phase at a time.**
2.  **Start with Phase 1.**
3.  Before implementing a phase, inspect the existing relevant files.
4.  Do not delete working functionality.
5.  Do not rewrite unrelated files.
6.  After implementation, test the phase.
7.  Fix errors before moving forward.
8.  Keep the application modular.
9.  Do not train the ML model inside the application.
10. Treat `best.pt` as an external model artifact.
11. Use a temporary model when necessary for early development.
12. Do not hardcode credentials.
13. Use `.env` for secrets.
14. Do not add unnecessary dependencies.
15. Do not build the frontend before the backend pipeline is stable.
16. Prefer a working MVP over unnecessary advanced features.
17. Keep configuration separate from business logic.
18. Keep `main.py` as an orchestrator.
19. Preserve a consistent detector output format.
20. At the end of every phase, report:
    -   What was implemented.
    -   Files created/modified.
    -   Tests performed.
    -   Test results.
    -   Any remaining issue.
    -   Whether the phase is ready to be marked complete.

------------------------------------------------------------------------

# 20. FIRST ACTION FOR ANTIGRAVITY

## START WITH PHASE 1 ONLY

Do NOT immediately implement all phases.

First:

``` text
Inspect existing project
        ↓
Create/repair project structure
        ↓
Set up dependencies
        ↓
Create configuration
        ↓
Create .gitignore/.env handling
        ↓
Verify project starts
```

After Phase 1 is verified, proceed to Phase 2.

The ML model training can happen separately in parallel.

**Current implementation priority:**

``` text
FIRST  → PHASE 1
NEXT   → PHASE 2
THEN   → PHASE 3
THEN   → PHASE 4
...
LAST   → PHASE 13
```

Do not skip directly to the frontend.
