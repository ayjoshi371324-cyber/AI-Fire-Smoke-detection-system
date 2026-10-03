"""
dashboard.py
Interactive Monitoring Dashboard for the AI Smart Smoke & Fire Detection System.
Built with Streamlit, Plotly, and Pandas.
Provides live camera stream monitoring, real-time metrics, analytics charts,
event log filtering, and snapshot inspection.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config
from src.camera_manager import CameraManager
from src.detector import YOLODetector
from src.temporal_filter import MultiCameraTemporalFilter
from src.severity import SeverityEstimator
from src.alerts import AlertManager
from src.database import DatabaseManager
from main import FireDetectionPipeline

# Set Page Config
st.set_page_config(
    page_title="AI Smart Smoke & Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: #1a1c24;
        border-radius: 12px;
        padding: 18px 22px;
        border-left: 5px solid #FF4500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9ca3af;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .badge-high {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #f59e0b;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .status-online {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_database_manager() -> DatabaseManager:
    """Initialize single DatabaseManager instance."""
    return DatabaseManager.from_config()


@st.cache_resource
def get_camera_manager() -> CameraManager:
    """Initialize CameraManager instance."""
    return CameraManager.from_config()


@st.cache_resource
def get_detector() -> YOLODetector:
    """Initialize shared YOLO detector."""
    return YOLODetector()


def load_event_data(db_mgr: DatabaseManager) -> pd.DataFrame:
    """Fetch all events from SQLite and return as formatted pandas DataFrame."""
    events = db_mgr.get_events(limit=500, order_desc=True)
    if not events:
        return pd.DataFrame(columns=[
            "id", "timestamp", "camera", "location", "label", "confidence", "severity", "snapshot"
        ])
    df = pd.DataFrame(events)
    if "confidence" in df.columns:
        df["confidence_pct"] = (df["confidence"] * 100).round(1).astype(str) + "%"
    return df


def render_header(detector: YOLODetector, active_camera: str):
    """Render top header and system status strip."""
    col_title, col_status = st.columns([3, 2])
    with col_title:
        st.title("🔥 AI Smart Smoke & Fire Detection System")
        st.caption("Real-Time Multi-Camera Hazard Detection & Intelligent Monitoring Dashboard")

    with col_status:
        st.markdown("<div style='text-align: right; padding-top: 15px;'>", unsafe_allow_html=True)
        st.markdown(
            """
            <span class='status-pill status-online'>● System: ONLINE</span>
            <span class='status-pill status-online'>● Camera: CONNECTED</span>
            <span class='status-pill status-online'>● AI Model: LOADED</span>
            """,
            unsafe_allow_html=True
        )
        model_name = Path(detector.model_path).name
        st.caption(f"Active Model: `{model_name}` | Source: `{active_camera}`")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()


def render_summary_cards(db_mgr: DatabaseManager):
    """Render top summary metrics cards."""
    counts = db_mgr.get_event_counts()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3b82f6;">
            <div class="metric-title">Total Incidents</div>
            <div class="metric-value">{counts['total']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ef4444;">
            <div class="metric-title">Fire Incidents</div>
            <div class="metric-value">{counts['fire']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #9ca3af;">
            <div class="metric-title">Smoke Incidents</div>
            <div class="metric-value">{counts['smoke']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #dc2626;">
            <div class="metric-title">High Severity Alerts</div>
            <div class="metric-value">{counts['high_severity']}</div>
        </div>
        """, unsafe_allow_html=True)


def render_live_monitoring(
    camera_name: str,
    cam_mgr: CameraManager,
    detector: YOLODetector,
    db_mgr: DatabaseManager,
    pipeline: FireDetectionPipeline
):
    """Render the live camera stream section with detection overlay and HUD."""
    st.subheader(f"📹 Live Stream Monitoring — {camera_name}")

    col_feed, col_telemetry = st.columns([3, 1])

    with col_telemetry:
        st.markdown("### Telemetry")
        location = cam_mgr.get_location(camera_name)
        st.info(f"**Physical Location:**\n{location}")

        stream_placeholder = st.empty()
        severity_pill = st.empty()
        confidence_metric = st.empty()
        cooldown_status = st.empty()

        # Stream Controls
        run_stream = st.toggle("Enable Live AI Inference Feed", value=False, key="stream_toggle")

    with col_feed:
        frame_placeholder = st.empty()
        status_text = st.empty()

        if run_stream:
            try:
                source = cam_mgr.get_source(camera_name, loop=True)
                if not source.open():
                    st.error(f"Failed to open video source for {camera_name}")
                    return

                status_text.success("● AI Pipeline Streaming Active")
                frame_count = 0

                # Stream processing loop
                while st.session_state.get("stream_toggle", False):
                    ret, frame = source.read()
                    if not ret or frame is None:
                        break

                    frame_count += 1
                    res = pipeline.process_frame(
                        frame=frame,
                        camera_name=camera_name,
                        frame_idx=frame_count,
                        render_annotated=True
                    )

                    # Convert BGR to RGB for Streamlit rendering
                    if res.annotated_frame is not None:
                        rgb_frame = cv2.cvtColor(res.annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    # Update telemetry
                    sev_level = res.severity.level
                    badge_class = "badge-high" if sev_level == "HIGH" else ("badge-medium" if sev_level == "MEDIUM" else "badge-low")
                    severity_pill.markdown(f"**Visual Severity:** <span class='{badge_class}'>{sev_level}</span>", unsafe_allow_html=True)

                    if res.temporal_status.max_confidence > 0:
                        confidence_metric.metric("Max Detection Confidence", f"{res.temporal_status.max_confidence*100:.1f}%")

                    rem_cd = pipeline.alert_manager.get_remaining_cooldown(camera_name)
                    if rem_cd > 0:
                        cooldown_status.warning(f"Cooldown Active: {rem_cd:.1f}s")
                    else:
                        cooldown_status.info("Alert System: Ready")

                    # Smooth frame pacing
                    time.sleep(0.03)

            except Exception as exc:
                st.error(f"Streaming error: {exc}")
            finally:
                source.release()
                status_text.info("Streaming stopped.")
        else:
            # Display representative still frame or standby placeholder
            source = cam_mgr.get_source(camera_name)
            if source.open():
                ret, frame = source.read()
                if ret and frame is not None:
                    # Run single frame inference for preview
                    res = pipeline.process_frame(frame, camera_name, render_annotated=True)
                    if res.annotated_frame is not None:
                        rgb_frame = cv2.cvtColor(res.annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                source.release()
            else:
                frame_placeholder.info("Click 'Enable Live AI Inference Feed' in the sidebar or toggle above to start streaming.")


def render_latest_alert_and_snapshot(db_mgr: DatabaseManager):
    """Render latest alert details card and most recent snapshot image."""
    st.subheader("🚨 Latest Incident Alert & Snapshot")

    latest = db_mgr.get_latest_event()
    if not latest:
        st.info("No confirmed hazard events logged yet. The system is clean.")
        return

    col_alert_info, col_alert_img = st.columns([1, 1])

    with col_alert_info:
        sev = latest["severity"]
        badge_class = "badge-high" if sev == "HIGH" else ("badge-medium" if sev == "MEDIUM" else "badge-low")

        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {'#ef4444' if sev == 'HIGH' else '#f59e0b'};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="{badge_class}">{latest['severity']} SEVERITY</span>
                <span style="color: #9ca3af; font-size: 0.9rem;">{latest['timestamp']}</span>
            </div>
            <div style="margin-top: 15px;">
                <p><strong>Camera:</strong> {latest['camera']}</p>
                <p><strong>Location:</strong> {latest.get('location') or latest['camera']}</p>
                <p><strong>Detection:</strong> {latest['label'].capitalize()}</p>
                <p><strong>Confidence:</strong> {latest['confidence']*100:.1f}%</p>
                <p><strong>Event ID:</strong> #{latest['id']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_alert_img:
        snap_path_str = latest.get("snapshot")
        if snap_path_str:
            snap_path = Path(snap_path_str)
            if not snap_path.is_absolute():
                snap_path = BASE_DIR / snap_path

            if snap_path.exists():
                st.image(str(snap_path), caption=f"Confirmed Event #{latest['id']} Snapshot ({latest['camera']})", use_container_width=True)
            else:
                st.warning(f"Snapshot file not found at {snap_path}")
        else:
            st.info("No snapshot attached to this incident.")


def render_analytics_charts(df: pd.DataFrame):
    """Render interactive Plotly analytics charts."""
    st.subheader("📊 Hazard Analytics & Distribution")

    if df.empty:
        st.info("Insufficient event history to render charts. Run the detection pipeline to generate data.")
        return

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:
        # Chart 1: Fire vs Smoke Donut Chart
        label_counts = df["label"].str.capitalize().value_counts().reset_index()
        label_counts.columns = ["Hazard", "Count"]

        colors = {"Fire": "#ef4444", "Smoke": "#9ca3af"}
        fig1 = px.pie(
            label_counts,
            names="Hazard",
            values="Count",
            title="Fire vs Smoke Breakdown",
            hole=0.45,
            color="Hazard",
            color_discrete_map=colors
        )
        fig1.update_layout(template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        # Chart 2: Severity Distribution Bar Chart
        sev_counts = df["severity"].str.upper().value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]

        sev_colors = {"HIGH": "#dc2626", "MEDIUM": "#f59e0b", "LOW": "#10b981", "NONE": "#6b7280"}
        fig2 = px.bar(
            sev_counts,
            x="Severity",
            y="Count",
            title="Severity Distribution",
            color="Severity",
            color_discrete_map=sev_colors,
            text_auto=True
        )
        fig2.update_layout(template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        # Chart 3: Incidents by Camera
        cam_counts = df["camera"].value_counts().reset_index()
        cam_counts.columns = ["Camera", "Count"]
        fig3 = px.bar(
            cam_counts,
            y="Camera",
            x="Count",
            orientation="h",
            title="Incidents Logged by Camera",
            color="Camera",
            text_auto=True
        )
        fig3.update_layout(template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Chart 4: Events Timeline Over Time
        try:
            df["dt"] = pd.to_datetime(df["timestamp"])
            timeline = df.set_index("dt").resample("1min").size().reset_index(name="Events")
            fig4 = px.line(
                timeline,
                x="dt",
                y="Events",
                title="Events Timeline Over Time",
                markers=True
            )
            fig4.update_layout(template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig4, use_container_width=True)
        except Exception:
            # Fallback simple index plot
            fig4 = px.scatter(df, x="id", y="confidence", color="severity", title="Confidence Distribution by Event ID")
            fig4.update_layout(template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig4, use_container_width=True)


def render_event_history_and_filters(df: pd.DataFrame, db_mgr: DatabaseManager):
    """Render searchable event history table with filters and snapshot viewer."""
    st.subheader("📜 Confirmed Event History & Logs")

    if df.empty:
        st.info("No recorded incidents found in database.")
        return

    # Filter Controls
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        cameras = ["All"] + sorted(list(df["camera"].dropna().unique()))
        selected_cam = st.selectbox("Filter Camera", cameras)

    with col_f2:
        labels = ["All"] + sorted(list(df["label"].dropna().str.capitalize().unique()))
        selected_label = st.selectbox("Filter Hazard Type", labels)

    with col_f3:
        severities = ["All"] + sorted(list(df["severity"].dropna().str.upper().unique()))
        selected_sev = st.selectbox("Filter Severity", severities)

    with col_f4:
        search_query = st.text_input("Search Location / Camera", value="")

    # Apply filters
    filtered_df = df.copy()
    if selected_cam != "All":
        filtered_df = filtered_df[filtered_df["camera"] == selected_cam]
    if selected_label != "All":
        filtered_df = filtered_df[filtered_df["label"].str.lower() == selected_label.lower()]
    if selected_sev != "All":
        filtered_df = filtered_df[filtered_df["severity"].str.upper() == selected_sev.upper()]
    if search_query:
        mask = (
            filtered_df["camera"].str.contains(search_query, case=False, na=False) |
            filtered_df["location"].str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    st.write(f"Showing **{len(filtered_df)}** of {len(df)} total logged events:")

    # Display Table
    cols_to_display = ["id", "timestamp", "camera", "location", "label", "confidence_pct", "severity", "snapshot"]
    available_cols = [c for c in cols_to_display if c in filtered_df.columns]
    st.dataframe(
        filtered_df[available_cols],
        use_container_width=True,
        hide_index=True
    )

    # Snapshot Preview on Selection
    with st.expander("🖼️ Inspect Event Snapshot by ID", expanded=False):
        inspect_id = st.number_input("Enter Event ID to inspect snapshot:", min_value=1, step=1, value=int(df["id"].iloc[0]))
        selected_event = db_mgr.get_event_by_id(int(inspect_id))
        if selected_event:
            st.json(selected_event)
            snap_file = selected_event.get("snapshot")
            if snap_file:
                p = Path(snap_file)
                if not p.is_absolute():
                    p = BASE_DIR / p
                if p.exists():
                    st.image(str(p), caption=f"Event #{inspect_id} Snapshot", use_container_width=True)
                else:
                    st.warning(f"File not found on disk at {p}")
        else:
            st.warning(f"No event record found with ID #{inspect_id}")


def render_sidebar(cam_mgr: CameraManager, db_mgr: DatabaseManager, pipeline: FireDetectionPipeline) -> str:
    """Render sidebar navigation, camera selector, and quick actions."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/fire-element.png", width=64)
        st.title("Control Center")

        cameras = list(cam_mgr.cameras_config.keys())
        active_camera = st.selectbox("Active Camera Feed", cameras, index=0)

        st.divider()
        st.subheader("⚙️ System Controls")

        # Thresholds summary
        st.write(f"**Confidence Threshold:** `{config.CONF_THRESHOLD}`")
        st.write(f"**Temporal Confirm Window:** `{config.CONFIRM_FRAMES}` frames")
        st.write(f"**Alert Cooldown:** `{config.ALERT_COOLDOWN_SEC}s`")

        st.divider()
        st.subheader("⚡ Quick Actions")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

        # Simulate test event button for demonstration
        if st.button("🚨 Trigger Test Fire Alert", use_container_width=True):
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_frame[150:350, 200:440] = (0, 100, 255)  # Orange flame region

            snap_path = db_mgr.save_snapshot(dummy_frame, camera=active_camera)
            event_id = db_mgr.log_event(
                camera=active_camera,
                location=cam_mgr.get_location(active_camera),
                label="fire",
                confidence=0.95,
                severity="HIGH",
                snapshot=snap_path
            )
            st.success(f"Demonstration incident #{event_id} logged successfully!")
            st.rerun()

        if st.button("🗑️ Clear Event History", use_container_width=True):
            count = db_mgr.clear_events()
            st.warning(f"Cleared {count} events from database.")
            st.rerun()

        st.divider()
        st.caption("AI Smart Smoke & Fire Detection System | v1.0.0")

    return active_camera


def main():
    """Main application orchestrator."""
    db_mgr = get_database_manager()
    cam_mgr = get_camera_manager()
    detector = get_detector()

    # Pipeline orchestrator
    pipeline = FireDetectionPipeline(
        camera_manager=cam_mgr,
        detector=detector,
        database_manager=db_mgr
    )

    # Sidebar
    active_camera = render_sidebar(cam_mgr, db_mgr, pipeline)

    # Top Header
    render_header(detector, active_camera)

    # Summary Metrics Cards
    render_summary_cards(db_mgr)

    # Tabs for clean presentation
    tab_live, tab_analytics, tab_history = st.tabs([
        "📹 Live Monitoring & Stream",
        "📊 Hazard Analytics & Charts",
        "📜 Incident History & Snapshots"
    ])

    with tab_live:
        render_live_monitoring(active_camera, cam_mgr, detector, db_mgr, pipeline)
        st.divider()
        render_latest_alert_and_snapshot(db_mgr)

    # Load data for analytics and history tabs
    df_events = load_event_data(db_mgr)

    with tab_analytics:
        render_analytics_charts(df_events)

    with tab_history:
        render_event_history_and_filters(df_events, db_mgr)


if __name__ == "__main__":
    main()
