"""Small status card: layout, content rows, config."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig, ConfigError, DebugConfig
from src.state.agent_status import AgentStatus
from src.ui.dashboard import Dashboard
from src.ui.debug_overlay import build_card, card_lines


def _busy() -> AgentStatus:
    return AgentStatus(
        state="RUNNING", phase=11, goal="STORY 1/3 TRAVEL",
        action="nearest:+045deg", message="riding toward the marker",
        hud_health=0.82, hud_stamina=0.64, hud_dead_eye=0.9,
        threat_level="wanted", threat_wanted=2,
        hud_prompt="PRESS E TO MOUNT", hud_prompt_visible=True,
        capture_fps=11.4, loop_hz=10.0, window_focused=True,
    )


def test_card_is_small_and_fixed_width() -> None:
    card = build_card(_busy())
    assert card.shape[1] == 400
    assert card.dtype.name == "uint8"
    assert card.shape[0] < 500


def test_card_border_follows_state() -> None:
    running = build_card(AgentStatus(state="RUNNING"))
    error = build_card(AgentStatus(state="ERROR"))
    assert running.shape == error.shape
    assert running[3, 200].tolist() != error[3, 200].tolist()


def test_card_waiting_flag_adds_row() -> None:
    plain = build_card(AgentStatus())
    waiting = build_card(AgentStatus(), waiting=True)
    assert waiting.shape[0] > plain.shape[0]


def test_card_focus_hint() -> None:
    assert any("Click the game window" in text for text, _ in card_lines(AgentStatus()))
    focused = AgentStatus(window_focused=True)
    assert not any("Click the game window" in text for text, _ in card_lines(focused))


def test_card_error_row() -> None:
    kinds = [kind for _, kind in card_lines(AgentStatus(error="boom"))]
    assert "error" in kinds
    kinds = [kind for _, kind in card_lines(AgentStatus())]
    assert "error" not in kinds


def test_card_survives_long_message() -> None:
    card = build_card(AgentStatus(state="ERROR", message="x" * 500, error="y" * 500))
    assert card.shape[1] == 400
    assert card.shape[0] <= 420


def test_card_threat_only_when_present() -> None:
    kinds = [kind for _, kind in card_lines(AgentStatus(threat_level="none"))]
    assert "threat" not in kinds and "alert" not in kinds
    kinds = [kind for _, kind in card_lines(AgentStatus(threat_level="wanted"))]
    assert "alert" in kinds


def test_show_video_defaults_off_and_validated() -> None:
    assert DebugConfig().show_video is False
    cfg = AppConfig()
    cfg.debug.show_video = "yes"  # type: ignore[assignment]
    try:
        cfg.validate()
    except ConfigError as exc:
        assert "debug.show_video" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_dashboard_disabled_is_noop() -> None:
    dash = Dashboard(DebugConfig(gui=False))
    dash.show(None, AgentStatus())
    assert dash.shown == 0
    assert dash.enabled is False


def test_card_uses_configured_title() -> None:
    default = build_card(AgentStatus(state="RUNNING"))
    custom = build_card(AgentStatus(state="RUNNING"), title="Arthur")
    assert default.shape == custom.shape
    assert not (default == custom).all()


def test_title_text_is_red() -> None:
    card = build_card(AgentStatus(state="RUNNING"))
    top = card[8:50, 0:400]
    red = top[:, :, 2].astype(int) - np.maximum(
        top[:, :, 0], top[:, :, 1]).astype(int)
    assert (red > 100).mean() > 0.02


def test_app_title_validated() -> None:
    assert DebugConfig().app_title == "RDR2 AI"
    cfg = AppConfig()
    cfg.debug.app_title = "   "
    try:
        cfg.validate()
    except ConfigError as exc:
        assert "debug.app_title" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_dashboard_visible_states(monkeypatch) -> None:
    dash = Dashboard(DebugConfig(gui=False))
    assert dash.visible() is True
    dash2 = Dashboard(DebugConfig(gui=True))
    assert dash2.visible() is True
    dash2._created = True  # noqa: SLF001 - white-box window check
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 0.0)
    assert dash2.visible() is False
    def _boom(*args: object) -> float:
        raise RuntimeError("no display")
    monkeypatch.setattr(cv2, "getWindowProperty", _boom)
    assert dash2.visible() is True


def test_window_autosize_for_card(monkeypatch) -> None:
    seen: dict[str, int] = {}
    monkeypatch.setattr(
        cv2, "namedWindow", lambda name, flags: seen.setdefault(name, flags)
    )
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: None)
    dash = Dashboard(DebugConfig(gui=True))
    dash.show(None, AgentStatus(state="RUNNING"))
    assert dash.shown == 1
    assert seen == {"RDR2 AI - Debug": cv2.WINDOW_AUTOSIZE}


def test_loop_hz_reports_real_rate(tmp_path: Path, monkeypatch) -> None:
    from src.main import main
    from tests.test_main_guard import FakeWindowManager

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join([
            "telemetry:",
            "  file: true",
            f"  dir: {tmp_path.as_posix()}",
            "  console: false",
            "capture:",
            "  backend: synthetic",
            "  target_fps: 30",
            "agent:",
            "  loop_hz: 10",
            "safety:",
            "  hotkey_poll_hz: 100",
            "  startup_grace_s: 1",
            "",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.main.GameWindowManager", FakeWindowManager)
    assert main(["--config", str(cfg_path), "--headless", "--duration", "2"]) == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rates = [row["loop_hz"] for row in rows if row["kind"] == "metrics"]
    assert rates, "no metrics emitted"
    assert all(1.0 <= hz <= 50.0 for hz in rates), rates[-3:]
