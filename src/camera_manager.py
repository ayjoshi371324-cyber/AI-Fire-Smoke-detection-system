"""
src/camera_manager.py
Reusable camera and video input abstraction layer for the Fire Detection System.
Supports webcams, local video files, and remote RTSP/CCTV streams.
Handles error detection, resource management, and frame provision.
"""

from typing import Optional, Tuple, Dict, Any, Generator, Union
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)


class VideoSource:
    """
    Manages a single video source (webcam, video file, or network stream).
    Provides safe opening, reading, looping, and release of video capture resources.
    """

    def __init__(
        self,
        source: Union[int, str, Path],
        name: str = "Camera",
        loop: bool = False
    ):
        """
        Initialize video source.

        Args:
            source: Webcam device index (int/str), video file path, or RTSP/HTTP URL.
            name: Human-readable identifier for this video feed.
            loop: If True and the source is a video file, restart when reaching the end.
        """
        self.name = name
        self.loop = loop
        self._raw_source = source
        self.source = self._normalize_source(source)
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False
        self.frame_count: int = 0

    @staticmethod
    def _normalize_source(source: Union[int, str, Path]) -> Union[int, str]:
        """Convert string digits to int for webcam indices, or string paths."""
        if isinstance(source, int):
            return source
        if isinstance(source, Path):
            return str(source)
        if isinstance(source, str):
            # If string is all digits (e.g. "0"), treat as camera index
            if source.strip().isdigit():
                return int(source.strip())
            return source.strip()
        return source

    def is_file_source(self) -> bool:
        """Check if source represents a local file."""
        return isinstance(self.source, str) and not self.is_network_source()

    def is_network_source(self) -> bool:
        """Check if source is an RTSP or HTTP network stream."""
        if isinstance(self.source, str):
            lower = self.source.lower()
            return lower.startswith("rtsp://") or lower.startswith("http://") or lower.startswith("https://")
        return False

    def is_webcam(self) -> bool:
        """Check if source is a physical or virtual webcam index."""
        return isinstance(self.source, int)

    def open(self) -> bool:
        """
        Open the video capture source.

        Returns:
            bool: True if source opened successfully, False otherwise.
        """
        if self._is_opened and self.cap is not None and self.cap.isOpened():
            return True

        logger.info(f"Opening video source '{self.name}' ({self.source})...")

        # For Windows webcams, cv2.CAP_DSHOW can provide faster/more stable capture
        if self.is_webcam():
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source '{self.name}' ({self.source})")
            self._is_opened = False
            return False

        self._is_opened = True
        self.frame_count = 0
        logger.info(f"Successfully opened video source '{self.name}'")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video source.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - (True, frame) if a frame was read successfully.
                - (False, None) if frame read failed or end of stream reached.
        """
        if not self._is_opened or self.cap is None:
            if not self.open():
                return False, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            # Handle video looping if enabled
            if self.loop and self.is_file_source():
                logger.debug(f"Source '{self.name}' reached EOF, looping back to start...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame_count += 1
                    return True, frame

            logger.warning(f"Could not read frame from source '{self.name}'")
            return False, None

        self.frame_count += 1
        return True, frame

    def get_properties(self) -> Dict[str, Any]:
        """
        Retrieve properties of the video capture source.

        Returns:
            Dictionary containing width, height, fps, total_frames, and backend info.
        """
        if not self._is_opened or self.cap is None:
            return {
                "name": self.name,
                "source": self.source,
                "is_opened": False,
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "total_frames": 0,
                "current_frame": 0
            }

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        return {
            "name": self.name,
            "source": self.source,
            "is_opened": self.cap.isOpened(),
            "width": width,
            "height": height,
            "fps": fps if fps > 0 else 30.0,
            "total_frames": total_frames,
            "current_frame": current_frame,
            "frames_read": self.frame_count
        }

    def release(self) -> None:
        """Safely release the video capture resource."""
        if self.cap is not None:
            logger.info(f"Releasing video capture resource for '{self.name}'")
            self.cap.release()
            self.cap = None
        self._is_opened = False

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Generator yielding frames sequentially until exhausted."""
        try:
            while True:
                success, frame = self.read()
                if not success or frame is None:
                    break
                yield frame
        finally:
            self.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __repr__(self) -> str:
        return f"<VideoSource name='{self.name}' source='{self.source}' opened={self._is_opened}>"


class CameraManager:
    """
    Coordinates one or multiple video sources defined in configuration or registered at runtime.
    """

    def __init__(self, cameras_config: Optional[Dict[str, Any]] = None):
        """
        Initialize CameraManager with camera configuration mapping.

        Args:
            cameras_config: Dictionary mapping camera names to their sources.
        """
        self.cameras_config: Dict[str, Any] = cameras_config or {}
        self._active_sources: Dict[str, VideoSource] = {}

    @classmethod
    def from_config(cls) -> "CameraManager":
        """Instantiate CameraManager using src.config.CAMERAS."""
        from src import config
        return cls(cameras_config=config.CAMERAS)

    def register_camera(self, name: str, source: Union[int, str, Path], loop: bool = False) -> None:
        """Register a new camera source configuration."""
        self.cameras_config[name] = source
        # If an active source was registered with this name, release it
        if name in self._active_sources:
            self._active_sources[name].release()
            del self._active_sources[name]

    def get_source(self, name: str, loop: bool = False) -> VideoSource:
        """
        Get or initialize a VideoSource instance by camera name.

        Args:
            name: Camera identifier key.
            loop: Whether video files should loop automatically.

        Returns:
            VideoSource instance.
        """
        if name not in self.cameras_config:
            raise KeyError(f"Camera '{name}' not found in configuration. Available: {list(self.cameras_config.keys())}")

        if name not in self._active_sources or not self._active_sources[name]._is_opened:
            source_target = self.cameras_config[name]
            self._active_sources[name] = VideoSource(source=source_target, name=name, loop=loop)

        return self._active_sources[name]

    def open_camera(self, name: str, loop: bool = False) -> bool:
        """Open a configured camera by name."""
        source = self.get_source(name, loop=loop)
        return source.open()

    def read_frame(self, name: str) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the specified camera."""
        source = self.get_source(name)
        return source.read()

    def list_cameras(self) -> Dict[str, Any]:
        """List all configured camera names and their source targets."""
        return dict(self.cameras_config)

    def release_camera(self, name: str) -> None:
        """Release a specific camera."""
        if name in self._active_sources:
            self._active_sources[name].release()
            del self._active_sources[name]

    def release_all(self) -> None:
        """Release all active camera sources."""
        for name, source in list(self._active_sources.items()):
            source.release()
        self._active_sources.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_all()
