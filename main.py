"""
main.py
Main application orchestrator for the AI Smart Smoke & Fire Detection System.
Coordinates the camera feed, YOLO detector, temporal filter, severity estimator,
alert dispatcher, snapshot generator, and SQLite database logging.
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config
from src.camera_manager import CameraManager, VideoSource, IncidentEvent
from src.detector import YOLODetector
from src.temporal_filter import MultiCameraTemporalFilter, TemporalFilter, TemporalStatus
from src.severity import SeverityEstimator, SeverityResult, estimate_severity
from src.alerts import AlertManager, AlertResult
from src.database import DatabaseManager

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("FireDetectionApp")


@dataclass
class FrameProcessingResult:
    """Outcome and metadata for a single processed video frame."""
    frame_number: int
    camera_name: str
    location: str
    detections: List[Dict[str, Any]]
    temporal_status: TemporalStatus
    severity: SeverityResult
    alert_triggered: bool
    event_logged: bool
    snapshot_path: Optional[str] = None
    annotated_frame: Optional[np.ndarray] = None
    fps: float = 0.0


@dataclass
class PipelineStats:
    """Summary execution metrics of the detection pipeline."""
    frames_processed: int = 0
    detections_count: int = 0
    confirmed_incidents: int = 0
    alerts_dispatched: int = 0
    events_logged: int = 0
    snapshots_saved: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time > 0 else time.time()
        return max(0.001, end - self.start_time)

    @property
    def average_fps(self) -> float:
        return round(self.frames_processed / self.elapsed_seconds, 1)


class FireDetectionPipeline:
    """
    Core orchestrator coordinating all detection components into a unified pipeline:
    Camera -> Frame -> YOLO -> Temporal Confirmation -> Severity -> Location -> Alert -> Snapshot -> DB
    """

    def __init__(
        self,
        camera_manager: Optional[CameraManager] = None,
        detector: Optional[YOLODetector] = None,
        temporal_filter: Optional[MultiCameraTemporalFilter] = None,
        severity_estimator: Optional[SeverityEstimator] = None,
        alert_manager: Optional[AlertManager] = None,
        database_manager: Optional[DatabaseManager] = None
    ):
        """Initialize all pipeline components, with fallbacks to defaults/config."""
        self.camera_manager = camera_manager or CameraManager.from_config()
        self.detector = detector or YOLODetector()
        self.temporal_filter = temporal_filter or MultiCameraTemporalFilter()
        self.severity_estimator = severity_estimator or SeverityEstimator()
        self.alert_manager = alert_manager or AlertManager.from_config()
        self.database_manager = database_manager or DatabaseManager.from_config()

        self.stats = PipelineStats()
        self._prev_frame_time = time.time()
        self._fps = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        camera_name: str,
        frame_idx: int = 0,
        render_annotated: bool = True
    ) -> FrameProcessingResult:
        """
        Process a single image frame through the complete 8-step pipeline:
        1. YOLO Detection
        2. Multi-frame Temporal Confirmation (False Alarm Suppression)
        3. Severity Estimation (Visual Area Coverage)
        4. Location Resolution
        5. Incident Event Construction
        6. Cooldown Evaluation & Alert Dispatch
        7. Snapshot Persistence
        8. SQLite Database Logging
        """
        # Calculate dynamic FPS
        now = time.time()
        time_diff = now - self._prev_frame_time
        if time_diff > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / time_diff)
        self._prev_frame_time = now

        self.stats.frames_processed += 1
        location = self.camera_manager.get_location(camera_name)

        # 1. YOLO Inference
        detections = self.detector.detect(frame)
        self.stats.detections_count += len(detections)

        # 2. Temporal Confirmation (Multi-frame analysis)
        temporal_status = self.temporal_filter.update(camera_name, detections)

        # 3. Severity Estimation
        severity = self.severity_estimator.estimate(detections, frame)

        # 4 & 5. Incident Confirmation & Alert Decision
        alert_triggered = False
        event_logged = False
        snapshot_saved_path = None

        if temporal_status.is_confirmed and severity.level != "NONE":
            self.stats.confirmed_incidents += 1

            # Determine dominant label & maximum confidence
            dominant_label = (
                temporal_status.confirmed_labels[0]
                if temporal_status.confirmed_labels
                else (detections[0]["label"] if detections else "fire")
            )
            confidence = temporal_status.max_confidence or (
                detections[0]["confidence"] if detections else 1.0
            )

            # Check if alert cooldown permits firing
            can_alert, remaining_cooldown = self.alert_manager.cooldown_tracker.can_alert(camera_name)

            if can_alert:
                # 6. Save Snapshot Frame before alert dispatch
                try:
                    snapshot_saved_path = self.database_manager.save_snapshot(
                        frame=frame,
                        camera=camera_name
                    )
                    self.stats.snapshots_saved += 1
                except Exception as exc:
                    logger.error("Failed to save snapshot: %s", exc)

                # 7. Create IncidentEvent
                event = self.camera_manager.create_incident_event(
                    camera_name=camera_name,
                    label=dominant_label,
                    confidence=confidence,
                    severity=severity.level,
                    snapshot=snapshot_saved_path,
                    bbox=detections[0].get("bbox") if detections else None
                )

                # 8. Trigger Alert (Local Alarm + Telegram)
                alert_res = self.alert_manager.trigger_alert(
                    event=event,
                    photo_path=snapshot_saved_path
                )
                alert_triggered = alert_res.triggered
                if alert_triggered:
                    self.stats.alerts_dispatched += 1

                # 9. Store Incident Record in SQLite Database
                try:
                    db_id = self.database_manager.log_event(
                        event=event,
                        snapshot=snapshot_saved_path
                    )
                    event_logged = db_id is not None
                    if event_logged:
                        self.stats.events_logged += 1
                except Exception as exc:
                    logger.error("Failed to log incident event to database: %s", exc)

        # Render visualizations and informational HUD
        annotated_frame = None
        if render_annotated:
            annotated_frame = self.render_hud(
                frame=frame,
                detections=detections,
                camera_name=camera_name,
                location=location,
                temporal_status=temporal_status,
                severity=severity,
                fps=self._fps
            )

        return FrameProcessingResult(
            frame_number=frame_idx,
            camera_name=camera_name,
            location=location,
            detections=detections,
            temporal_status=temporal_status,
            severity=severity,
            alert_triggered=alert_triggered,
            event_logged=event_logged,
            snapshot_path=snapshot_saved_path,
            annotated_frame=annotated_frame,
            fps=self._fps
        )

    def render_hud(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        camera_name: str,
        location: str,
        temporal_status: TemporalStatus,
        severity: SeverityResult,
        fps: float
    ) -> np.ndarray:
        """Draw bounding boxes and a sleek monitoring HUD over the frame."""
        # 1. Draw detection bounding boxes & badges
        output = self.detector.draw_detections(frame, detections, copy=True)
        h, w = output.shape[:2]

        # 2. Draw Top Semi-Transparent Status Banner
        header_height = 42
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

        # 3. Camera & Location Info
        title_text = f"CAM: {camera_name} | LOC: {location}"
        cv2.putText(
            output, title_text, (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

        # 4. FPS counter
        fps_text = f"{fps:.1f} FPS"
        cv2.putText(
            output, fps_text, (w - 85, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
        )

        # 5. Draw Severity & Confirmation Status Badge (Bottom Right)
        badge_color = SeverityEstimator.get_color(severity.level)
        if temporal_status.is_confirmed:
            status_text = f"CONFIRMED [{severity.level}]"
            badge_bg = badge_color
            text_color = (255, 255, 255)
        elif temporal_status.positive_frames > 0:
            status_text = f"ANALYZING: {temporal_status.positive_frames}/{temporal_status.threshold_frames}"
            badge_bg = (50, 50, 50)
            text_color = (0, 215, 255)
        else:
            status_text = "MONITORING: NORMAL"
            badge_bg = (30, 80, 30)
            text_color = (220, 255, 220)

        # Draw status pill in bottom corner
        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        pad = 8
        bx1 = w - tw - pad * 2 - 12
        by1 = h - th - pad * 2 - 12
        bx2 = w - 12
        by2 = h - 12

        cv2.rectangle(output, (bx1, by1), (bx2, by2), badge_bg, -1)
        cv2.rectangle(output, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        cv2.putText(
            output, status_text, (bx1 + pad, by2 - pad - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA
        )

        # 6. Cooldown indicator
        remaining = self.alert_manager.get_remaining_cooldown(camera_name)
        if remaining > 0:
            cooldown_text = f"Cooldown: {remaining:.0f}s"
            cv2.putText(
                output, cooldown_text, (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA
            )

        return output

    def run(
        self,
        camera_name: str,
        max_frames: Optional[int] = None,
        headless: bool = False,
        loop: bool = False,
        stride: int = 1
    ) -> PipelineStats:
        """
        Run the detection pipeline continuously on the designated camera source.

        Args:
            camera_name: Name of configured camera to process.
            max_frames: Optional maximum frames to process before stopping.
            headless: If True, disables cv2.imshow GUI display.
            loop: If True and source is a video file, loops playback upon completion.

        Returns:
            PipelineStats detailing execution metrics.
        """
        source = self.camera_manager.get_source(camera_name, loop=loop)
        opened = source.open()
        if not opened:
            logger.error("Unable to open camera source for '%s'", camera_name)
            return self.stats

        logger.info("Pipeline started for '%s' (%s). Headless=%s", camera_name, source.location, headless)
        window_title = f"AI Fire Detection - {camera_name}"

        frame_idx = 0
        self.stats.start_time = time.time()

        try:
            while True:
                ret, frame = source.read()
                if not ret or frame is None:
                    logger.info("End of video stream reached for '%s'.", camera_name)
                    break

                frame_idx += 1
                if stride > 1 and (frame_idx % stride != 0):
                    continue

                result = self.process_frame(
                    frame=frame,
                    camera_name=camera_name,
                    frame_idx=frame_idx,
                    render_annotated=not headless
                )

                if result.alert_triggered:
                    logger.warning(
                        "ALERT TRIGGERED [%s]: Severity=%s, Conf=%.2f, Snapshot=%s",
                        camera_name, result.severity.level,
                        result.temporal_status.max_confidence, result.snapshot_path
                    )

                if not headless and result.annotated_frame is not None:
                    cv2.imshow(window_title, result.annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):  # 'q' or ESC
                        logger.info("User requested exit via keyboard.")
                        break
                    elif key == ord('m'):  # Toggle mute
                        self.alert_manager.stop_alarm()
                        logger.info("Alarm stopped / muted.")
                    elif key == ord('r'):  # Reset filter
                        self.temporal_filter.reset(camera_name)
                        self.alert_manager.reset_cooldown(camera_name)
                        logger.info("Reset temporal filter and cooldown.")

                if max_frames and frame_idx >= max_frames:
                    logger.info("Reached maximum requested frames (%d).", max_frames)
                    break

        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user interrupt (Ctrl+C).")
        finally:
            self.stats.end_time = time.time()
            source.release()
            if not headless:
                cv2.destroyAllWindows()
            self.alert_manager.stop_alarm()

        logger.info(
            "Pipeline completed: %d frames processed at %.1f FPS. Alerts: %d, DB Events: %d, Snapshots: %d",
            self.stats.frames_processed, self.stats.average_fps,
            self.stats.alerts_dispatched, self.stats.events_logged,
            self.stats.snapshots_saved
        )
        return self.stats


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Smart Smoke & Fire Detection System - Main Application Pipeline"
    )
    parser.add_argument(
        "-c", "--camera",
        type=str,
        default="Fire Video",
        help="Configured camera name to stream (default: 'Fire Video')."
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        default=None,
        help="Optional direct video file or webcam index override (e.g. 'videos/fire_test.mp4' or '0')."
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Physical location identifier for the source."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="YOLO confidence threshold (0.0 - 1.0)."
    )
    parser.add_argument(
        "--headless", "--no-gui",
        action="store_true",
        help="Run without displaying GUI window (headless/server mode)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before stopping."
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop video playback indefinitely."
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Mute local audible sound alarm."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Optional inference image resolution (e.g. 320 or 640)."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (skip frames for speed optimization)."
    )
    return parser.parse_args()


def main() -> None:
    """Application entrypoint."""
    args = parse_arguments()

    print("=" * 65)
    print("   AI SMART SMOKE & FIRE DETECTION SYSTEM - PIPELINE RUNNER")
    print("=" * 65)

    # Initialize Camera Manager
    cam_mgr = CameraManager.from_config()

    # Handle direct source override if provided
    active_camera_name = args.camera
    if args.source:
        active_camera_name = "Custom Source"
        # Parse int if webcam index
        src_val = int(args.source) if args.source.isdigit() else args.source
        cam_mgr.register_camera(
            name=active_camera_name,
            source=src_val,
            location=args.location or "Custom Location"
        )

    # Initialize Detector
    detector = YOLODetector(conf_threshold=args.conf, imgsz=args.imgsz)
    det_info = detector.get_info()
    print(f"[*] Detector Model: {det_info['model_path']} (Custom: {det_info['is_custom_model']})")
    print(f"[*] Confidence Threshold: {detector.conf_threshold}")
    if args.imgsz:
        print(f"[*] Optimized Resolution (imgsz): {args.imgsz}")
    if args.stride > 1:
        print(f"[*] Frame Stride / Skip: Every {args.stride} frames")

    # Initialize Alert Manager
    alert_mgr = AlertManager.from_config()
    if args.mute:
        alert_mgr.local_alarm.enabled = False
        print("[*] Local sound alarm: MUTED")

    # Initialize Database Manager
    db_mgr = DatabaseManager.from_config()
    print(f"[*] SQLite Database: {db_mgr.db_path}")
    print(f"[*] Snapshots Directory: {db_mgr.snapshot_dir}")

    # Build Pipeline
    pipeline = FireDetectionPipeline(
        camera_manager=cam_mgr,
        detector=detector,
        alert_manager=alert_mgr,
        database_manager=db_mgr
    )

    print(f"[*] Starting Pipeline on Camera: '{active_camera_name}' (Location: '{cam_mgr.get_location(active_camera_name)}')")
    print("=" * 65)

    stats = pipeline.run(
        camera_name=active_camera_name,
        max_frames=args.max_frames,
        headless=args.headless,
        loop=args.loop,
        stride=args.stride
    )

    print("\n" + "=" * 65)
    print("                  EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Frames Processed:    {stats.frames_processed}")
    print(f"Average Speed:       {stats.average_fps} FPS")
    print(f"Detections Found:    {stats.detections_count}")
    print(f"Confirmed Hazards:   {stats.confirmed_incidents}")
    print(f"Alerts Dispatched:   {stats.alerts_dispatched}")
    print(f"Events Logged (DB):  {stats.events_logged}")
    print(f"Snapshots Saved:     {stats.snapshots_saved}")
    print("=" * 65)


if __name__ == "__main__":
    main()
