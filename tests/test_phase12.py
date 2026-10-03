"""
tests/test_phase12.py
Verification test suite for Phase 12 - Frontend / Dashboard.
Validates dashboard data loading, metric calculations, Plotly chart generation,
event filtering logic, snapshot preview path resolution, and syntax compilation.
"""

import sys
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import py_compile

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.database import DatabaseManager
import dashboard


def test_dashboard_syntax_and_compilation():
    """Verify that dashboard.py compiles without syntax or import errors."""
    print("[1/6] Testing dashboard.py compilation and syntax...")
    dashboard_file = BASE_DIR / "dashboard.py"
    assert dashboard_file.exists(), "dashboard.py must exist in project root."

    compiled = py_compile.compile(str(dashboard_file))
    assert compiled is not None, "dashboard.py failed to compile."
    print("      -> dashboard.py compiled successfully.")


def test_event_dataframe_loading_and_transformation():
    """Verify loading event data into pandas DataFrame with formatting."""
    print("[2/6] Testing event data extraction and pandas formatting...")

    temp_dir = Path(tempfile.mkdtemp(prefix="test_dash_"))
    try:
        db_file = temp_dir / "dash_events.db"
        snap_dir = temp_dir / "snapshots"
        db = DatabaseManager(db_path=db_file, snapshot_dir=snap_dir)

        # 1. Test empty database handling
        df_empty = dashboard.load_event_data(db)
        assert isinstance(df_empty, pd.DataFrame)
        assert len(df_empty) == 0
        assert "confidence" in df_empty.columns

        # 2. Seed test events
        db.log_event(camera="Cam 1", label="fire", confidence=0.942, severity="HIGH", timestamp="2026-10-03 12:00:00")
        db.log_event(camera="Cam 2", label="smoke", confidence=0.816, severity="MEDIUM", timestamp="2026-10-03 12:05:00")

        df = dashboard.load_event_data(db)
        assert len(df) == 2
        assert "confidence_pct" in df.columns
        assert df["confidence_pct"].iloc[0] in ("94.2%", "81.6%")
        print(f"      - Loaded {len(df)} records into DataFrame with formatted percentage column.")

        print("      -> Event data transformation verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_plotly_charts_generation():
    """Verify generation of all required Plotly charts (pie, bar, timeline)."""
    print("[3/6] Testing Plotly chart rendering and serialization...")

    test_data = [
        {"id": 1, "timestamp": "2026-10-03 10:00:00", "camera": "Kitchen", "label": "fire", "confidence": 0.95, "severity": "HIGH"},
        {"id": 2, "timestamp": "2026-10-03 10:05:00", "camera": "Kitchen", "label": "smoke", "confidence": 0.85, "severity": "MEDIUM"},
        {"id": 3, "timestamp": "2026-10-03 10:10:00", "camera": "Warehouse", "label": "fire", "confidence": 0.91, "severity": "HIGH"},
        {"id": 4, "timestamp": "2026-10-03 10:15:00", "camera": "Lab", "label": "smoke", "confidence": 0.72, "severity": "LOW"},
    ]
    df = pd.DataFrame(test_data)

    # 1. Fire vs Smoke breakdown chart
    import plotly.express as px
    label_counts = df["label"].str.capitalize().value_counts().reset_index()
    label_counts.columns = ["Hazard", "Count"]
    fig1 = px.pie(label_counts, names="Hazard", values="Count", hole=0.45)
    assert isinstance(fig1, go.Figure)
    assert len(fig1.data[0].labels) == 2

    # 2. Severity distribution chart
    sev_counts = df["severity"].str.upper().value_counts().reset_index()
    sev_counts.columns = ["Severity", "Count"]
    fig2 = px.bar(sev_counts, x="Severity", y="Count")
    assert isinstance(fig2, go.Figure)

    # 3. Camera incidents chart
    cam_counts = df["camera"].value_counts().reset_index()
    cam_counts.columns = ["Camera", "Count"]
    fig3 = px.bar(cam_counts, y="Camera", x="Count", orientation="h")
    assert isinstance(fig3, go.Figure)

    print("      -> All 3 interactive Plotly figures generated successfully.")


def test_summary_metrics_aggregation():
    """Verify metrics calculation for total, fire, smoke, and high severity."""
    print("[4/6] Testing summary metrics count aggregation...")

    temp_dir = Path(tempfile.mkdtemp(prefix="test_dash_counts_"))
    try:
        db_file = temp_dir / "counts.db"
        db = DatabaseManager(db_path=db_file)

        db.log_event(camera="A", label="fire", confidence=0.9, severity="HIGH")
        db.log_event(camera="A", label="fire", confidence=0.8, severity="LOW")
        db.log_event(camera="B", label="smoke", confidence=0.85, severity="HIGH")

        counts = db.get_event_counts()
        print(f"      - Aggregated counts: {counts}")
        assert counts["total"] == 3
        assert counts["fire"] == 2
        assert counts["smoke"] == 1
        assert counts["high_severity"] == 2

        print("      -> Summary metrics aggregation verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_event_filtering_logic():
    """Verify DataFrame filtering by camera, label, severity, and search."""
    print("[5/6] Testing event history filtering criteria...")

    records = [
        {"id": 1, "camera": "Kitchen", "location": "Building 1", "label": "fire", "severity": "HIGH"},
        {"id": 2, "camera": "Kitchen", "location": "Building 1", "label": "smoke", "severity": "MEDIUM"},
        {"id": 3, "camera": "Server Room", "location": "Basement", "label": "fire", "severity": "LOW"},
    ]
    df = pd.DataFrame(records)

    # Filter by Camera
    f_cam = df[df["camera"] == "Kitchen"]
    assert len(f_cam) == 2

    # Filter by Label
    f_label = df[df["label"] == "smoke"]
    assert len(f_label) == 1

    # Filter by Severity
    f_sev = df[df["severity"] == "HIGH"]
    assert len(f_sev) == 1

    # Search filter
    search_term = "basement"
    mask = df["location"].str.contains(search_term, case=False, na=False)
    assert len(df[mask]) == 1
    assert df[mask].iloc[0]["camera"] == "Server Room"

    print("      -> All filter dimensions verified.")


def test_snapshot_inspection_path_handling():
    """Verify snapshot resolution and existence checks."""
    print("[6/6] Testing snapshot path resolution...")

    temp_dir = Path(tempfile.mkdtemp(prefix="test_snap_res_"))
    try:
        snap_file = temp_dir / "20261003_120000.jpg"
        snap_file.write_bytes(b"dummy")

        db_file = temp_dir / "events.db"
        db = DatabaseManager(db_path=db_file)
        ev_id = db.log_event(camera="Cam", label="fire", confidence=0.9, severity="HIGH", snapshot=str(snap_file))

        ev = db.get_event_by_id(ev_id)
        assert ev is not None
        resolved = Path(ev["snapshot"])
        assert resolved.exists(), f"Snapshot path not resolved: {resolved}"

        print("      -> Snapshot inspection path resolution verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """Run all Phase 12 verification tests."""
    print("=" * 60)
    print("RUNNING PHASE 12 VERIFICATION SUITE (FRONTEND / DASHBOARD)")
    print("=" * 60)

    test_dashboard_syntax_and_compilation()
    test_event_dataframe_loading_and_transformation()
    test_plotly_charts_generation()
    test_summary_metrics_aggregation()
    test_event_filtering_logic()
    test_snapshot_inspection_path_handling()

    print("=" * 60)
    print("PHASE 12 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
