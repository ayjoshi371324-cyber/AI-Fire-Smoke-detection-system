"""
src/temporal_filter.py
False alarm reduction layer implementing multi-frame temporal confirmation.
Requires sustained detections across consecutive video frames within a sliding
window before an incident is marked as confirmed.
"""

from typing import List, Dict, Any, Optional, Set
from collections import deque
from dataclasses import dataclass, field
import logging

from src import config

logger = logging.getLogger(__name__)


@dataclass
class TemporalStatus:
    """Status summary returned by TemporalFilter after evaluating a frame."""
    is_confirmed: bool
    positive_frames: int
    window_size: int
    threshold_frames: int
    confirmed_labels: List[str] = field(default_factory=list)
    max_confidence: float = 0.0
    qualifying_detections: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def confirmation_ratio(self) -> float:
        """Ratio of positive frames to total window size."""
        return round(self.positive_frames / max(1, self.window_size), 2)

    def __bool__(self) -> bool:
        return self.is_confirmed


class TemporalFilter:
    """
    Sliding window temporal filter that accumulates frame detection history.
    Prevents fleeting false alarms (e.g. transient reflections, camera glitch)
    from triggering alerts by requiring N positive frames out of the last M frames.
    """

    def __init__(
        self,
        confirm_frames: Optional[int] = None,
        window_size: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        target_labels: Optional[Set[str]] = None
    ):
        """
        Initialize temporal filter.

        Args:
            confirm_frames: Number of positive frames needed within window to confirm (default from config).
            window_size: Number of past frames tracked in sliding window (default from config).
            conf_threshold: Minimum confidence required for a detection to count (default from config).
            target_labels: Set of target class names (e.g. {"fire", "smoke"}). If None, defaults to {"fire", "smoke"}.
        """
        self.confirm_frames = confirm_frames if confirm_frames is not None else config.CONFIRM_FRAMES
        self.window_size = window_size if window_size is not None else config.WINDOW_SIZE
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.CONF_THRESHOLD
        self.target_labels = {lbl.lower() for lbl in (target_labels or {"fire", "smoke"})}

        # Sliding window storing boolean presence of target detections per frame
        self._window: deque = deque(maxlen=self.window_size)
        # History of labels seen per frame in the window
        self._labels_window: deque = deque(maxlen=self.window_size)

    def filter_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections by confidence threshold and target labels.

        Args:
            detections: Raw detection list from detector.

        Returns:
            List of qualifying detections meeting confidence and label requirements.
        """
        qualifying = []
        for det in detections:
            conf = det.get("confidence", 0.0)
            label = det.get("label", "").lower()

            if conf >= self.conf_threshold:
                # If target labels defined, ensure label matches or is part of name
                if not self.target_labels or any(t in label for t in self.target_labels):
                    qualifying.append(det)

        return qualifying

    def update(self, detections: List[Dict[str, Any]]) -> TemporalStatus:
        """
        Ingest current frame detections, update sliding window, and evaluate confirmation status.

        Args:
            detections: List of detections for the current frame.

        Returns:
            TemporalStatus detailing whether detection is confirmed, frame counts, and qualifying items.
        """
        qualifying = self.filter_detections(detections)
        has_positive = len(qualifying) > 0

        # Record frame outcome in sliding windows
        self._window.append(has_positive)
        current_labels = {d["label"].lower() for d in qualifying}
        self._labels_window.append(current_labels)

        # Count positive frames in current sliding window
        positive_frames = sum(1 for is_pos in self._window if is_pos)
        is_confirmed = positive_frames >= self.confirm_frames

        # Aggregate confirmed labels across positive frames in window
        aggregated_labels: Set[str] = set()
        for labels in self._labels_window:
            aggregated_labels.update(labels)

        max_confidence = (
            max((d.get("confidence", 0.0) for d in qualifying), default=0.0)
            if qualifying
            else 0.0
        )

        return TemporalStatus(
            is_confirmed=is_confirmed,
            positive_frames=positive_frames,
            window_size=self.window_size,
            threshold_frames=self.confirm_frames,
            confirmed_labels=sorted(list(aggregated_labels)),
            max_confidence=max_confidence,
            qualifying_detections=qualifying
        )

    def reset(self) -> None:
        """Clear all historical frame history from the sliding window."""
        self._window.clear()
        self._labels_window.clear()

    @property
    def is_currently_confirmed(self) -> bool:
        """Quick check if currently in confirmed alert state."""
        return sum(1 for is_pos in self._window if is_pos) >= self.confirm_frames


class MultiCameraTemporalFilter:
    """
    Manages isolated TemporalFilter instances for multiple camera feeds,
    preventing frame counts from different sources from interleaving.
    """

    def __init__(
        self,
        confirm_frames: Optional[int] = None,
        window_size: Optional[int] = None,
        conf_threshold: Optional[float] = None
    ):
        self.confirm_frames = confirm_frames
        self.window_size = window_size
        self.conf_threshold = conf_threshold
        self._filters: Dict[str, TemporalFilter] = {}

    def get_filter(self, camera_name: str) -> TemporalFilter:
        """Get or initialize a filter dedicated to the given camera."""
        if camera_name not in self._filters:
            self._filters[camera_name] = TemporalFilter(
                confirm_frames=self.confirm_frames,
                window_size=self.window_size,
                conf_threshold=self.conf_threshold
            )
        return self._filters[camera_name]

    def update(self, camera_name: str, detections: List[Dict[str, Any]]) -> TemporalStatus:
        """Update detection state for a specific camera."""
        camera_filter = self.get_filter(camera_name)
        return camera_filter.update(detections)

    def reset(self, camera_name: Optional[str] = None) -> None:
        """Reset a specific camera's filter or all camera filters."""
        if camera_name:
            if camera_name in self._filters:
                self._filters[camera_name].reset()
        else:
            for f in self._filters.values():
                f.reset()
