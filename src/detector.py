"""
src/detector.py
Modular YOLO detection interface for the Fire Detection System.
Provides consistent detection outputs, decouples YOLO internals from downstream modules,
and seamlessly switches between development/fallback models and custom-trained models.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from pathlib import Path
import os
import cv2
import numpy as np
from ultralytics import YOLO

from src import config

# Configure logger
logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Standardized YOLO Detection Interface.
    Encapsulates model loading, inference, and formatting so the rest of the
    system relies exclusively on a clean, decoupled data contract:
    [
        {
            "label": "fire",
            "confidence": 0.91,
            "bbox": (x1, y1, x2, y2)
        }
    ]
    """

    # Distinct color schemes (BGR)
    COLOR_FIRE = (0, 69, 255)       # Red-Orange for Fire
    COLOR_SMOKE = (180, 180, 180)   # Slate Gray for Smoke
    COLOR_DEFAULT = (50, 205, 50)   # Emerald Green for generic/test objects

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None
    ):
        """
        Initialize detector with specified model or default from configuration.

        Args:
            model_path: Optional path to YOLO weights (.pt). If None, uses config.MODEL_PATH.
            conf_threshold: Minimum detection confidence (0.0 - 1.0).
            device: Computation device ('cpu', 'cuda', '0', etc.). Auto-detected if None.
            imgsz: Optional inference image dimension (e.g. 320 or 640) for speed optimization.
        """
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.CONF_THRESHOLD
        self.device = device  # None lets Ultralytics select best available device
        self.imgsz = imgsz

        # Determine model path with graceful fallback
        target_path = Path(model_path) if model_path else Path(config.MODEL_PATH)
        fallback_path = Path(config.FALLBACK_MODEL_PATH)

        if target_path.exists():
            if target_path.is_dir() and (target_path / "data.pkl").exists():
                import zipfile
                packaged_path = target_path.parent / (target_path.stem + "_packed.pt")
                if not packaged_path.exists() or packaged_path.stat().st_mtime < target_path.stat().st_mtime:
                    with zipfile.ZipFile(packaged_path, "w", compression=zipfile.ZIP_STORED) as zf:
                        for root, _, files in os.walk(target_path):
                            for file in files:
                                full_file = Path(root) / file
                                rel = full_file.relative_to(target_path).as_posix()
                                zf.write(full_file, f"archive/{rel}")
                target_path = packaged_path
            self.model_path = str(target_path)
            self.is_custom_model = True
            logger.info(f"Loading primary YOLO model from '{self.model_path}'")
        elif fallback_path.exists():
            self.model_path = str(fallback_path)
            self.is_custom_model = False
            logger.warning(
                f"Primary model '{target_path}' not found. "
                f"Falling back to temporary model '{self.model_path}' for development."
            )
        else:
            # If neither exists locally, fallback to default ultralytics yolov8n
            self.model_path = "yolov8n.pt"
            self.is_custom_model = False
            logger.warning(f"Using default temporary model '{self.model_path}'.")

        # Load YOLO model instance
        self.model = YOLO(self.model_path)
        self.classes: Dict[int, str] = self.model.names
        self.is_fire_model = any("fire" in name.lower() or "smoke" in name.lower() for name in self.classes.values())

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a single frame and return standardized detections.

        Args:
            frame: Input image/frame as numpy array (BGR).
            conf: Optional confidence override for this inference call.
            imgsz: Optional resolution override for inference.

        Returns:
            List[Dict[str, Any]]: Standardized list of detections:
            [
                {
                    "label": "fire",
                    "confidence": 0.91,
                    "bbox": (x1, y1, x2, y2)
                }
            ]
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Empty or invalid frame passed to detector.")
            return []

        threshold = conf if conf is not None else self.conf_threshold
        active_imgsz = imgsz or self.imgsz

        # Run Ultralytics YOLO inference
        predict_kwargs = {
            "source": frame,
            "conf": threshold,
            "verbose": False,
            "device": self.device
        }
        if active_imgsz:
            predict_kwargs["imgsz"] = active_imgsz

        results = self.model.predict(**predict_kwargs)

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        first_result = results[0]
        if first_result.boxes is None or len(first_result.boxes) == 0:
            return detections

        for box in first_result.boxes:
            # Extract coordinates as (x1, y1, x2, y2)
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            confidence = round(float(box.conf[0]), 4)
            class_id = int(box.cls[0])
            raw_label = self.classes.get(class_id, str(class_id)).lower()

            detections.append({
                "label": raw_label,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        copy: bool = True
    ) -> np.ndarray:
        """
        Render bounding boxes, labels, and confidence tags on a frame.

        Args:
            frame: OpenCV BGR image frame.
            detections: Standardized list of detections.
            copy: If True, draws on a copy of the frame to keep original pristine.

        Returns:
            Rendered image with visual detections.
        """
        output_frame = frame.copy() if copy else frame

        for det in detections:
            label = det.get("label", "unknown")
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", (0, 0, 0, 0))
            x1, y1, x2, y2 = bbox

            # Choose accent color
            if "fire" in label:
                color = self.COLOR_FIRE
            elif "smoke" in label:
                color = self.COLOR_SMOKE
            else:
                color = self.COLOR_DEFAULT

            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

            # Prepare text badge
            text = f"{label.upper()} {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_height - 8)
            badge_y2 = y1
            badge_x2 = min(output_frame.shape[1], x1 + text_width + 10)

            # Draw filled badge header
            cv2.rectangle(output_frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Draw badge text in white or dark depending on color
            text_color = (0, 0, 0) if color == self.COLOR_SMOKE else (255, 255, 255)
            cv2.putText(
                output_frame,
                text,
                (x1 + 5, badge_y2 - 4),
                font,
                font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA
            )

        return output_frame

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic details of the active detector."""
        return {
            "model_path": self.model_path,
            "is_custom_model": self.is_custom_model,
            "is_fire_model": self.is_fire_model,
            "conf_threshold": self.conf_threshold,
            "num_classes": len(self.classes),
            "class_names": list(self.classes.values())[:10]  # First 10 classes
        }
