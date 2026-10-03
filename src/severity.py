"""
src/severity.py
Severity estimation module for the AI Smart Smoke & Fire Detection System.
Calculates visual severity (NONE, LOW, MEDIUM, HIGH) based on the relative
area of fire and smoke detections against the total frame dimensions.
"""

from typing import List, Dict, Any, Tuple, Union, Optional
from dataclasses import dataclass
import enum
import logging
import numpy as np

from src import config

logger = logging.getLogger(__name__)


class SeverityLevel(str, enum.Enum):
    """Visual severity categories."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    def __str__(self) -> str:
        return self.value


@dataclass
class SeverityResult:
    """Detailed severity assessment result."""
    level: str  # "NONE", "LOW", "MEDIUM", "HIGH"
    fire_ratio: float
    smoke_ratio: float
    total_ratio: float
    fire_count: int
    smoke_count: int
    summary: str

    def __str__(self) -> str:
        return self.level

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.level == other
        if isinstance(other, SeverityLevel):
            return self.level == other.value
        if isinstance(other, SeverityResult):
            return self.level == other.level
        return False


class SeverityEstimator:
    """
    Estimates visual severity based on bounding box coverage relative to frame area.
    """

    def __init__(
        self,
        low_threshold: Optional[float] = None,
        medium_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None
    ):
        """
        Initialize severity estimator with configurable threshold ratios.

        Args:
            low_threshold: Frame area ratio threshold for LOW severity (default from config).
            medium_threshold: Frame area ratio threshold for MEDIUM severity (default from config).
            high_threshold: Frame area ratio threshold for HIGH severity (default from config).
        """
        self.low_threshold = (
            low_threshold if low_threshold is not None else config.SEVERITY_LOW_THRESHOLD
        )
        self.medium_threshold = (
            medium_threshold if medium_threshold is not None else config.SEVERITY_MEDIUM_THRESHOLD
        )
        self.high_threshold = (
            high_threshold if high_threshold is not None else config.SEVERITY_HIGH_THRESHOLD
        )

    @staticmethod
    def _extract_dimensions(
        frame_shape: Union[Tuple[int, ...], np.ndarray]
    ) -> Tuple[int, int]:
        """
        Extract (width, height) from various frame representations.

        Accepts:
            - (height, width) or (height, width, channels)
            - numpy ndarray
        """
        if isinstance(frame_shape, np.ndarray):
            height, width = frame_shape.shape[:2]
            return width, height

        if isinstance(frame_shape, (tuple, list)):
            if len(frame_shape) >= 2:
                # In OpenCV/numpy conventions, shape is (height, width, ...)
                height, width = frame_shape[0], frame_shape[1]
                return width, height

        raise ValueError(f"Invalid frame dimensions or shape: {frame_shape}")

    def estimate(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Union[Tuple[int, ...], np.ndarray]
    ) -> SeverityResult:
        """
        Calculate visual severity level based on detection bounding boxes and frame area.

        Args:
            detections: List of detection dictionaries containing 'label' and 'bbox'.
            frame_shape: Frame dimensions as (height, width, ...) or numpy ndarray.

        Returns:
            SeverityResult containing severity level string and calculated ratios.
        """
        width, height = self._extract_dimensions(frame_shape)
        frame_area = max(1, width * height)

        fire_area = 0
        smoke_area = 0
        fire_count = 0
        smoke_count = 0

        for det in detections:
            label = det.get("label", "").lower()
            bbox = det.get("bbox", (0, 0, 0, 0))
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            # Clamp coordinates to frame boundaries
            clamped_x1 = max(0, min(width, x1))
            clamped_y1 = max(0, min(height, y1))
            clamped_x2 = max(0, min(width, x2))
            clamped_y2 = max(0, min(height, y2))

            box_w = max(0, clamped_x2 - clamped_x1)
            box_h = max(0, clamped_y2 - clamped_y1)
            area = box_w * box_h

            if "fire" in label:
                fire_area += area
                fire_count += 1
            elif "smoke" in label:
                smoke_area += area
                smoke_count += 1

        fire_ratio = round(fire_area / frame_area, 4)
        smoke_ratio = round(smoke_area / frame_area, 4)
        total_ratio = round(min(1.0, (fire_area + smoke_area) / frame_area), 4)

        # Classification logic based on area and detection type:
        # 1. No fire or smoke -> NONE
        if fire_count == 0 and smoke_count == 0:
            level = SeverityLevel.NONE.value
            summary = "No fire or smoke detected"

        # 2. Large fire region OR significant fire + smoke -> HIGH
        elif (fire_ratio >= self.high_threshold) or (
            fire_ratio >= self.medium_threshold and smoke_ratio >= self.medium_threshold
        ):
            level = SeverityLevel.HIGH.value
            summary = f"Critical hazard: Large fire region ({fire_ratio*100:.1f}% coverage)"

        # 3. Medium detected region OR extensive smoke -> MEDIUM
        elif (fire_ratio >= self.medium_threshold) or (smoke_ratio >= self.high_threshold):
            level = SeverityLevel.MEDIUM.value
            summary = f"Moderate hazard: Fire ({fire_ratio*100:.1f}%), Smoke ({smoke_ratio*100:.1f}%)"

        # 4. Small detected region -> LOW
        elif (fire_ratio > 0) or (smoke_ratio > 0):
            level = SeverityLevel.LOW.value
            summary = f"Minor hazard: Fire ({fire_ratio*100:.1f}%), Smoke ({smoke_ratio*100:.1f}%)"

        else:
            level = SeverityLevel.NONE.value
            summary = "No visual hazard identified"

        return SeverityResult(
            level=level,
            fire_ratio=fire_ratio,
            smoke_ratio=smoke_ratio,
            total_ratio=total_ratio,
            fire_count=fire_count,
            smoke_count=smoke_count,
            summary=summary
        )

    @staticmethod
    def get_color(level: Union[str, SeverityLevel]) -> Tuple[int, int, int]:
        """
        Get BGR color corresponding to severity level.

        Returns:
            Tuple[int, int, int] (BGR format for OpenCV rendering).
        """
        key = str(level).upper()
        if key == SeverityLevel.HIGH.value:
            return (0, 0, 255)      # Red
        if key == SeverityLevel.MEDIUM.value:
            return (0, 140, 255)    # Orange
        if key == SeverityLevel.LOW.value:
            return (0, 215, 255)    # Amber/Yellow
        return (128, 128, 128)      # Gray for NONE


# Convenient functional entry point
_default_estimator = SeverityEstimator()

def estimate_severity(
    detections: List[Dict[str, Any]],
    frame_shape: Union[Tuple[int, ...], np.ndarray]
) -> SeverityResult:
    """
    Convenience function to estimate severity using default configured thresholds.
    """
    return _default_estimator.estimate(detections, frame_shape)
