"""
src/alerts.py
Alert notification system for the AI Smart Smoke & Fire Detection System.
Provides local audio alarm (Windows-compatible sound/buzzer) and remote Telegram notifications.
Includes configurable cooldown tracking to prevent alert fatigue and spam.
"""

from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
import datetime
import time
import logging
import threading
from pathlib import Path
import os
import requests

from src import config

logger = logging.getLogger(__name__)

# Try importing winsound on Windows platforms
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    winsound = None
    WINSOUND_AVAILABLE = False


@dataclass
class AlertResult:
    """Detailed outcome of an alert dispatch attempt."""
    triggered: bool
    cooldown_active: bool
    local_alarm_played: bool
    telegram_sent: bool
    message: str
    camera: str
    severity: str
    remaining_cooldown: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "triggered": self.triggered,
            "cooldown_active": self.cooldown_active,
            "local_alarm_played": self.local_alarm_played,
            "telegram_sent": self.telegram_sent,
            "message": self.message,
            "camera": self.camera,
            "severity": self.severity,
            "remaining_cooldown": round(self.remaining_cooldown, 2),
            "timestamp": self.timestamp
        }


class LocalAlarm:
    """
    Local audible alarm controller.
    Plays a WAV sound file or generates a system buzzer beep on Windows.
    Runs asynchronously to avoid blocking the video processing loop.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        sound_path: Optional[Union[str, Path]] = None,
        beep_fallback: bool = True
    ):
        """
        Initialize local alarm.

        Args:
            enabled: Whether local audio alarm is active (defaults to config.LOCAL_ALARM_ENABLED).
            sound_path: Path to custom WAV file (defaults to config.ALARM_SOUND_PATH).
            beep_fallback: Fallback to motherboard / system speaker beep if WAV file is unavailable.
        """
        self.enabled = enabled if enabled is not None else getattr(config, "LOCAL_ALARM_ENABLED", True)
        self.sound_path = str(sound_path or getattr(config, "ALARM_SOUND_PATH", "assets/alarm.wav"))
        self.beep_fallback = beep_fallback
        self._is_playing = False
        self._lock = threading.Lock()

    def play(self, asynchronous: bool = True) -> bool:
        """
        Trigger local alarm sound.

        Args:
            asynchronous: Run sound in background thread without blocking execution.

        Returns:
            bool: True if sound was dispatched, False if disabled or failed.
        """
        if not self.enabled:
            logger.debug("Local alarm is disabled in configuration.")
            return False

        if not WINSOUND_AVAILABLE:
            logger.warning("winsound module not available on this platform. Audible alert simulated.")
            return True

        if asynchronous:
            thread = threading.Thread(target=self._play_sound, daemon=True)
            thread.start()
            return True
        else:
            return self._play_sound()

    def _play_sound(self) -> bool:
        """Internal audio playback handler with graceful fallbacks."""
        with self._lock:
            self._is_playing = True
            try:
                resolved_sound = Path(self.sound_path)
                if resolved_sound.exists() and resolved_sound.suffix.lower() == ".wav":
                    # SND_ASYNC | SND_FILENAME: Play WAV asynchronously without hanging
                    winsound.PlaySound(
                        str(resolved_sound),
                        winsound.SND_FILENAME | winsound.SND_ASYNC
                    )
                    logger.info("Local alarm sound played from %s", resolved_sound)
                    return True
                elif self.beep_fallback:
                    # Fallback to high-pitch emergency beep: 1500Hz for 500ms
                    logger.info("Alarm WAV not found at %s. Sounding system beep fallback.", self.sound_path)
                    winsound.Beep(1500, 500)
                    return True
                else:
                    logger.warning("Alarm sound file not found and beep fallback disabled: %s", self.sound_path)
                    return False
            except Exception as exc:
                logger.error("Failed to play local alarm sound: %s", exc)
                if self.beep_fallback:
                    try:
                        winsound.Beep(1000, 300)
                        return True
                    except Exception:
                        pass
                return False
            finally:
                self._is_playing = False

    def stop(self) -> None:
        """Stop any actively looping or playing sound."""
        if WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception as exc:
                logger.debug("Error stopping sound: %s", exc)
        self._is_playing = False


class TelegramAlerter:
    """
    Remote notification dispatcher for Telegram Bot API.
    Sends formatted incident text and optional snapshot images.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout: int = 5
    ):
        """
        Initialize Telegram alerter.

        Args:
            bot_token: Telegram bot API token (defaults to config.TELEGRAM_BOT_TOKEN).
            chat_id: Telegram target chat ID (defaults to config.TELEGRAM_CHAT_ID).
            enabled: Whether Telegram dispatch is enabled (defaults to config.TELEGRAM_ALERT_ENABLED).
            timeout: HTTP request timeout in seconds.
        """
        self.bot_token = (bot_token or getattr(config, "TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or getattr(config, "TELEGRAM_CHAT_ID", "")).strip()
        self.enabled = enabled if enabled is not None else getattr(config, "TELEGRAM_ALERT_ENABLED", False)
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Check if required credentials are present."""
        return bool(self.bot_token and self.chat_id)

    @staticmethod
    def format_message(
        camera: str,
        detection: str,
        severity: str,
        confidence: float,
        timestamp: Optional[str] = None,
        location: Optional[str] = None
    ) -> str:
        """
        Format standardized alert message matching the project specification:

        FIRE ALERT

        Camera: Kitchen
        Location: Main Floor Kitchen  (optional)
        Detection: Fire
        Severity: HIGH
        Confidence: 91%
        Time: 13:25:32
        """
        if timestamp is None:
            now = datetime.datetime.now()
            time_str = now.strftime("%H:%M:%S")
        else:
            # If timestamp is already formatted or full datetime string
            if " " in timestamp:
                time_str = timestamp.split(" ")[-1]
            else:
                time_str = timestamp

        conf_pct = int(round(confidence * 100)) if confidence <= 1.0 else int(round(confidence))
        label_display = detection.capitalize()
        severity_display = str(severity).upper()

        lines = [
            "FIRE ALERT",
            "",
            f"Camera: {camera}"
        ]

        if location and location != camera:
            lines.append(f"Location: {location}")

        lines.extend([
            f"Detection: {label_display}",
            f"Severity: {severity_display}",
            f"Confidence: {conf_pct}%",
            f"Time: {time_str}"
        ])

        return "\n".join(lines)

    def send_message(self, message: str) -> bool:
        """
        Send text notification via Telegram Bot API.

        Args:
            message: Formatted text message.

        Returns:
            bool: True if delivered successfully, False otherwise.
        """
        if not self.enabled:
            logger.debug("Telegram alert skipped: disabled in configuration.")
            return False

        if not self.is_configured:
            logger.warning("Telegram alert skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                logger.info("Telegram alert delivered successfully.")
                return True
            else:
                logger.error("Telegram API error (%d): %s", response.status_code, response.text)
                return False
        except requests.RequestException as exc:
            logger.error("Failed to deliver Telegram alert: %s", exc)
            return False

    def send_photo(self, photo_path: Union[str, Path], caption: Optional[str] = None) -> bool:
        """
        Send snapshot image attachment via Telegram Bot API.

        Args:
            photo_path: Local file path to snapshot image.
            caption: Optional text caption accompanying photo.

        Returns:
            bool: True if photo was delivered, False otherwise.
        """
        if not self.enabled or not self.is_configured:
            return False

        path_obj = Path(photo_path)
        if not path_obj.exists():
            logger.warning("Telegram photo skipped: snapshot not found at %s", photo_path)
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption

        try:
            with open(path_obj, "rb") as photo_file:
                files = {"photo": photo_file}
                response = requests.post(url, data=data, files=files, timeout=self.timeout)

            if response.status_code == 200:
                logger.info("Telegram photo delivered successfully.")
                return True
            else:
                logger.error("Telegram photo API error (%d): %s", response.status_code, response.text)
                return False
        except requests.RequestException as exc:
            logger.error("Failed to send Telegram photo: %s", exc)
            return False


class CooldownTracker:
    """
    Tracks and enforces alert cooldown periods per camera.
    Prevents alert storms and spamming during sustained fire incidents.
    """

    def __init__(self, cooldown_sec: Optional[float] = None):
        """
        Initialize cooldown tracker.

        Args:
            cooldown_sec: Minimum seconds between consecutive alerts for the same source.
        """
        self.cooldown_sec = (
            float(cooldown_sec)
            if cooldown_sec is not None
            else float(getattr(config, "ALERT_COOLDOWN_SEC", 30))
        )
        self._last_alert_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def can_alert(self, camera: str, current_time: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if an alert can be triggered for the specified camera.

        Args:
            camera: Camera or source identifier.
            current_time: Optional current timestamp (defaults to time.time()).

        Returns:
            Tuple[bool, float]: (allowed_to_alert, remaining_cooldown_seconds)
        """
        now = current_time if current_time is not None else time.time()

        with self._lock:
            last_time = self._last_alert_times.get(camera)
            if last_time is None:
                return True, 0.0

            elapsed = now - last_time
            if elapsed >= self.cooldown_sec:
                return True, 0.0

            remaining = self.cooldown_sec - elapsed
            return False, remaining

    def record_alert(self, camera: str, current_time: Optional[float] = None) -> None:
        """
        Record that an alert was dispatched for the given camera.

        Args:
            camera: Camera identifier.
            current_time: Optional timestamp (defaults to time.time()).
        """
        now = current_time if current_time is not None else time.time()
        with self._lock:
            self._last_alert_times[camera] = now

    def get_remaining_cooldown(self, camera: str, current_time: Optional[float] = None) -> float:
        """Return remaining cooldown in seconds for camera, or 0.0 if ready."""
        _, remaining = self.can_alert(camera, current_time)
        return remaining

    def reset(self, camera: Optional[str] = None) -> None:
        """
        Reset cooldown for a specific camera or all cameras.

        Args:
            camera: If specified, reset only this camera; otherwise reset all.
        """
        with self._lock:
            if camera is not None:
                self._last_alert_times.pop(camera, None)
            else:
                self._last_alert_times.clear()


class AlertManager:
    """
    High-level orchestrator coordinating local alarms, Telegram notifications,
    cooldown enforcement, and duplicate alert suppression.
    """

    def __init__(
        self,
        cooldown_sec: Optional[float] = None,
        local_alarm: Optional[LocalAlarm] = None,
        telegram_alerter: Optional[TelegramAlerter] = None
    ):
        """
        Initialize AlertManager.

        Args:
            cooldown_sec: Alert cooldown duration in seconds.
            local_alarm: Optional custom LocalAlarm instance.
            telegram_alerter: Optional custom TelegramAlerter instance.
        """
        self.cooldown_tracker = CooldownTracker(cooldown_sec=cooldown_sec)
        self.local_alarm = local_alarm or LocalAlarm()
        self.telegram = telegram_alerter or TelegramAlerter()

    @classmethod
    def from_config(cls) -> "AlertManager":
        """Instantiate AlertManager using active project configuration."""
        return cls(
            cooldown_sec=getattr(config, "ALERT_COOLDOWN_SEC", 30),
            local_alarm=LocalAlarm(
                enabled=getattr(config, "LOCAL_ALARM_ENABLED", True),
                sound_path=getattr(config, "ALARM_SOUND_PATH", "assets/alarm.wav")
            ),
            telegram_alerter=TelegramAlerter(
                bot_token=getattr(config, "TELEGRAM_BOT_TOKEN", ""),
                chat_id=getattr(config, "TELEGRAM_CHAT_ID", ""),
                enabled=getattr(config, "TELEGRAM_ALERT_ENABLED", False)
            )
        )

    def is_on_cooldown(self, camera: str) -> bool:
        """Return True if the camera is currently in a cooldown window."""
        allowed, _ = self.cooldown_tracker.can_alert(camera)
        return not allowed

    def get_remaining_cooldown(self, camera: str) -> float:
        """Get remaining cooldown seconds for camera."""
        return self.cooldown_tracker.get_remaining_cooldown(camera)

    def trigger_alert(
        self,
        event: Any,
        photo_path: Optional[Union[str, Path]] = None,
        force: bool = False
    ) -> AlertResult:
        """
        Dispatch alerts for a confirmed incident event.
        Enforces cooldown: if the camera recently triggered an alert, further alerts
        are suppressed until the cooldown window expires.

        Args:
            event: IncidentEvent instance or dict containing:
                   camera, label/detection, severity, confidence, (optional) location, timestamp.
            photo_path: Optional local path to snapshot frame to attach to Telegram.
            force: If True, bypasses cooldown checks.

        Returns:
            AlertResult detailing actions performed.
        """
        # Extract attributes from either IncidentEvent object or dictionary
        if hasattr(event, "camera"):
            camera = getattr(event, "camera")
            label = getattr(event, "label", "fire")
            severity = str(getattr(event, "severity", "HIGH"))
            confidence = float(getattr(event, "confidence", 1.0))
            location = getattr(event, "location", None)
            timestamp = getattr(event, "timestamp", None)
            event_snapshot = getattr(event, "snapshot", None)
        elif isinstance(event, dict):
            camera = event.get("camera", "Unknown Camera")
            label = event.get("label", event.get("detection", "fire"))
            severity = str(event.get("severity", "HIGH"))
            confidence = float(event.get("confidence", 1.0))
            location = event.get("location")
            timestamp = event.get("timestamp")
            event_snapshot = event.get("snapshot")
        else:
            raise TypeError(f"Unsupported event type: {type(event)}. Expected IncidentEvent or dict.")

        snapshot_to_send = photo_path or event_snapshot

        # Evaluate cooldown
        can_fire, remaining = self.cooldown_tracker.can_alert(camera)
        if not can_fire and not force:
            logger.info(
                "Alert suppressed for '%s' due to cooldown (%.1fs remaining).",
                camera, remaining
            )
            return AlertResult(
                triggered=False,
                cooldown_active=True,
                local_alarm_played=False,
                telegram_sent=False,
                message=f"Alert suppressed: Cooldown active ({remaining:.1f}s remaining)",
                camera=camera,
                severity=severity,
                remaining_cooldown=remaining
            )

        # Record alert dispatch timestamp
        self.cooldown_tracker.record_alert(camera)

        # Format Telegram text message
        formatted_message = TelegramAlerter.format_message(
            camera=camera,
            detection=label,
            severity=severity,
            confidence=confidence,
            timestamp=timestamp,
            location=location
        )

        # 1. Trigger Local Alarm
        alarm_played = False
        try:
            alarm_played = self.local_alarm.play(asynchronous=True)
        except Exception as exc:
            logger.error("Failed to trigger local alarm: %s", exc)

        # 2. Trigger Telegram Alert
        telegram_sent = False
        try:
            if self.telegram.enabled:
                if snapshot_to_send and Path(snapshot_to_send).exists():
                    telegram_sent = self.telegram.send_photo(
                        snapshot_to_send,
                        caption=formatted_message
                    )
                else:
                    telegram_sent = self.telegram.send_message(formatted_message)
        except Exception as exc:
            logger.error("Failed to send Telegram alert: %s", exc)

        return AlertResult(
            triggered=True,
            cooldown_active=False,
            local_alarm_played=alarm_played,
            telegram_sent=telegram_sent,
            message=formatted_message,
            camera=camera,
            severity=severity,
            remaining_cooldown=0.0
        )

    def trigger_from_values(
        self,
        camera: str,
        detection: str,
        severity: str,
        confidence: float,
        location: Optional[str] = None,
        photo_path: Optional[Union[str, Path]] = None,
        timestamp: Optional[str] = None,
        force: bool = False
    ) -> AlertResult:
        """Convenience method to trigger alerts using direct values instead of event object."""
        payload = {
            "camera": camera,
            "label": detection,
            "severity": severity,
            "confidence": confidence,
            "location": location,
            "timestamp": timestamp
        }
        return self.trigger_alert(payload, photo_path=photo_path, force=force)

    def stop_alarm(self) -> None:
        """Stop local alarm sound."""
        self.local_alarm.stop()

    def reset_cooldown(self, camera: Optional[str] = None) -> None:
        """Reset cooldown tracking."""
        self.cooldown_tracker.reset(camera)
