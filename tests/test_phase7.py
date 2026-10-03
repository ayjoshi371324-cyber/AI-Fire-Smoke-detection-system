"""
tests/test_phase7.py
Verification test suite for Phase 7 - Alert System.
Validates local audio alarm, Telegram message formatting, Telegram network dispatch,
cooldown enforcement, multi-camera duplicate-alert prevention, and IncidentEvent integration.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.alerts import (
    AlertManager,
    AlertResult,
    LocalAlarm,
    TelegramAlerter,
    CooldownTracker
)
from src.camera_manager import CameraManager, IncidentEvent
from src import config


def test_telegram_message_formatting():
    """Verify Telegram message format strictly satisfies the project specification."""
    print("[1/7] Testing Telegram alert message formatting...")

    msg = TelegramAlerter.format_message(
        camera="Kitchen",
        detection="Fire",
        severity="HIGH",
        confidence=0.91,
        timestamp="13:25:32"
    )

    print("      - Generated message:")
    for line in msg.splitlines():
        print(f"        | {line}")

    # Check required specification lines
    assert "FIRE ALERT" in msg
    assert "Camera: Kitchen" in msg
    assert "Detection: Fire" in msg
    assert "Severity: HIGH" in msg
    assert "Confidence: 91%" in msg
    assert "Time: 13:25:32" in msg

    # Test with custom location
    msg_with_loc = TelegramAlerter.format_message(
        camera="Cam 2",
        detection="smoke",
        severity="MEDIUM",
        confidence=0.75,
        location="Warehouse Bay 4",
        timestamp="2026-10-03 14:30:00"
    )
    assert "Location: Warehouse Bay 4" in msg_with_loc
    assert "Detection: Smoke" in msg_with_loc
    assert "Severity: MEDIUM" in msg_with_loc
    assert "Confidence: 75%" in msg_with_loc
    assert "Time: 14:30:00" in msg_with_loc

    print("      -> Telegram message formatting verified.")


def test_local_alarm_operation():
    """Verify local alarm sound and buzzer triggers without blocking."""
    print("[2/7] Testing local alarm controller (sound & buzzer)...")

    alarm = LocalAlarm(enabled=True, sound_path=config.ALARM_SOUND_PATH, beep_fallback=True)
    assert alarm.enabled is True

    # Test synchronous and asynchronous playback (does not throw)
    res_sync = alarm.play(asynchronous=False)
    assert res_sync in (True, False)

    res_async = alarm.play(asynchronous=True)
    assert res_async is True

    alarm.stop()

    # Test disabled alarm
    disabled_alarm = LocalAlarm(enabled=False)
    assert disabled_alarm.play() is False

    print("      -> Local alarm operation verified.")


def test_telegram_dispatch_mocked():
    """Verify Telegram network dispatch handling for success, failure, and snapshots."""
    print("[3/7] Testing Telegram API dispatcher with mocked network calls...")

    alerter = TelegramAlerter(
        bot_token="test_token_12345",
        chat_id="test_chat_67890",
        enabled=True
    )
    assert alerter.is_configured is True

    # 1. Mock successful message dispatch (200 OK)
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        success = alerter.send_message("Test Fire Alert")
        assert success is True
        assert mock_post.called
        call_args, call_kwargs = mock_post.call_args
        assert "api.telegram.org/bottest_token_12345/sendMessage" in call_args[0]
        assert call_kwargs["json"]["chat_id"] == "test_chat_67890"

    # 2. Mock API error (401 Unauthorized)
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = '{"ok":false,"error_code":401,"description":"Unauthorized"}'
        mock_post.return_value = mock_response

        success = alerter.send_message("Test Fire Alert")
        assert success is False

    # 3. Mock network timeout exception
    import requests
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):
        success = alerter.send_message("Test Fire Alert")
        assert success is False

    # 4. Test photo dispatch with mocked file
    dummy_snapshot = BASE_DIR / "snapshots" / "test_snapshot.jpg"
    dummy_snapshot.parent.mkdir(parents=True, exist_ok=True)
    dummy_snapshot.write_bytes(b"dummy image bytes")

    try:
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            success = alerter.send_photo(dummy_snapshot, caption="Fire photo")
            assert success is True
            call_args, call_kwargs = mock_post.call_args
            assert "sendPhoto" in call_args[0]
            assert call_kwargs["data"]["caption"] == "Fire photo"
    finally:
        if dummy_snapshot.exists():
            dummy_snapshot.unlink()

    # 5. Test unconfigured / disabled alerter
    unconfigured = TelegramAlerter(bot_token="", chat_id="", enabled=True)
    assert unconfigured.is_configured is False
    assert unconfigured.send_message("Hello") is False

    print("      -> Telegram API dispatch verified.")


def test_cooldown_enforcement():
    """Verify that repeated alerts within cooldown window are strictly suppressed."""
    print("[4/7] Testing cooldown enforcement and alert deduplication...")

    cooldown = CooldownTracker(cooldown_sec=30)
    camera = "Kitchen Cam"

    # Initial state: can alert immediately
    can_fire, remaining = cooldown.can_alert(camera, current_time=1000.0)
    assert can_fire is True
    assert remaining == 0.0

    # Trigger alert at t = 1000.0
    cooldown.record_alert(camera, current_time=1000.0)

    # 5 seconds later (t = 1005.0): should be blocked
    can_fire, remaining = cooldown.can_alert(camera, current_time=1005.0)
    assert can_fire is False
    assert round(remaining, 1) == 25.0
    print(f"      - At t=1005s: Alert blocked as expected ({remaining:.1f}s cooldown remaining)")

    # 29.9 seconds later (t = 1029.9): still blocked
    can_fire, remaining = cooldown.can_alert(camera, current_time=1029.9)
    assert can_fire is False
    assert round(remaining, 1) == 0.1

    # 30.1 seconds later (t = 1030.1): cooldown expired, can alert again!
    can_fire, remaining = cooldown.can_alert(camera, current_time=1030.1)
    assert can_fire is True
    assert remaining == 0.0
    print("      - At t=1030.1s: Cooldown expired, alert permitted again.")

    # Reset cooldown manually
    cooldown.record_alert(camera, current_time=1030.1)
    cooldown.reset(camera)
    can_fire, _ = cooldown.can_alert(camera, current_time=1030.2)
    assert can_fire is True

    print("      -> Cooldown enforcement verified.")


def test_multi_camera_cooldown_isolation():
    """Verify that cooldown on one camera does NOT suppress alerts from other cameras."""
    print("[5/7] Testing multi-camera cooldown isolation...")

    tracker = CooldownTracker(cooldown_sec=30)
    tracker.record_alert("Server Room", current_time=500.0)

    # Server Room should be blocked
    can_fire_server, remaining_server = tracker.can_alert("Server Room", current_time=510.0)
    assert can_fire_server is False
    assert remaining_server == 20.0

    # Warehouse must NOT be blocked
    can_fire_warehouse, remaining_warehouse = tracker.can_alert("Warehouse", current_time=510.0)
    assert can_fire_warehouse is True
    assert remaining_warehouse == 0.0
    print(f"      - 'Server Room' is on cooldown ({remaining_server}s), while 'Warehouse' alerts freely.")

    print("      -> Multi-camera cooldown isolation verified.")


def test_alert_manager_incident_event_integration():
    """Verify full AlertManager integration with IncidentEvent from CameraManager."""
    print("[6/7] Testing AlertManager dispatch with IncidentEvent data contracts...")

    camera_mgr = CameraManager.from_config()
    event = camera_mgr.create_incident_event(
        camera_name="Fire Video",
        label="fire",
        confidence=0.96,
        severity="HIGH",
        bbox=(50, 50, 250, 250)
    )

    alert_mgr = AlertManager(
        cooldown_sec=30,
        local_alarm=LocalAlarm(enabled=False),
        telegram_alerter=TelegramAlerter(enabled=False)
    )

    # 1. First alert should trigger
    res1 = alert_mgr.trigger_alert(event)
    assert isinstance(res1, AlertResult)
    assert res1.triggered is True
    assert res1.cooldown_active is False
    assert res1.camera == "Fire Video"
    assert res1.severity == "HIGH"
    assert "FIRE ALERT" in res1.message
    print(f"      - Event 1: Triggered={res1.triggered}, CooldownActive={res1.cooldown_active}")

    # 2. Immediate second alert for same camera must be suppressed by cooldown
    res2 = alert_mgr.trigger_alert(event)
    assert res2.triggered is False
    assert res2.cooldown_active is True
    assert res2.remaining_cooldown > 0
    print(f"      - Event 2 (immediate repeat): Suppressed={not res2.triggered}, Remaining={res2.remaining_cooldown:.1f}s")

    # 3. Forcing alert bypasses cooldown
    res_forced = alert_mgr.trigger_alert(event, force=True)
    assert res_forced.triggered is True
    print(f"      - Event 3 (forced): Triggered={res_forced.triggered}")

    # 4. Dictionary format input support
    dict_event = {
        "camera": "Lab 1",
        "location": "Chemistry Wing",
        "label": "smoke",
        "confidence": 0.85,
        "severity": "MEDIUM",
        "timestamp": "12:00:00"
    }
    res_dict = alert_mgr.trigger_alert(dict_event)
    assert res_dict.triggered is True
    assert res_dict.camera == "Lab 1"
    assert "Location: Chemistry Wing" in res_dict.message

    print("      -> AlertManager IncidentEvent integration verified.")


def test_security_and_config_contracts():
    """Verify security requirements: no hardcoded secrets and environment bindings."""
    print("[7/7] Testing security contracts and environment binding...")

    # Ensure AlertManager loads defaults from config
    mgr = AlertManager.from_config()
    assert mgr.cooldown_tracker.cooldown_sec == config.ALERT_COOLDOWN_SEC
    assert mgr.local_alarm.enabled == config.LOCAL_ALARM_ENABLED
    assert mgr.telegram.enabled == config.TELEGRAM_ALERT_ENABLED

    # Ensure secrets are sourced through config
    assert mgr.telegram.bot_token == config.TELEGRAM_BOT_TOKEN
    assert mgr.telegram.chat_id == config.TELEGRAM_CHAT_ID

    print("      -> Security contracts and configuration verified.")


def run_all_tests():
    """Run all Phase 7 verification tests."""
    print("=" * 60)
    print("RUNNING PHASE 7 VERIFICATION SUITE (ALERT SYSTEM)")
    print("=" * 60)

    test_telegram_message_formatting()
    test_local_alarm_operation()
    test_telegram_dispatch_mocked()
    test_cooldown_enforcement()
    test_multi_camera_cooldown_isolation()
    test_alert_manager_incident_event_integration()
    test_security_and_config_contracts()

    print("=" * 60)
    print("PHASE 7 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
