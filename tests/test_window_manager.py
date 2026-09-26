"""Win32 window manager tests (no game required)."""

from __future__ import annotations

import time

import pytest

from src.capture.window_manager import (
    GameWindowManager,
    Rect,
    WindowInfo,
    enable_dpi_awareness,
    make_focus_check,
)
from src.config import AppConfig, ConfigError, WindowConfig


def test_rect_properties() -> None:
    r = Rect(10, 20, 110, 220)
    assert r.width == 100
    assert r.height == 200
    assert r.area == 20000
    assert r.is_valid
    assert Rect(0, 0, 0, 10).is_valid is False
    assert Rect(0, 0, 5, 5).as_tuple() == (0, 0, 5, 5)


def test_rect_clamp() -> None:
    r = Rect(-10, -10, 50, 50).clamp(Rect(0, 0, 40, 40))
    assert r.as_tuple() == (0, 0, 40, 40)


def test_find_returns_none_for_impossible_pattern() -> None:
    cfg = WindowConfig(title_patterns=["definitely-not-a-real-window-xyz"])
    wm = GameWindowManager(cfg)
    assert wm.find() is None
    assert wm.current() is None


def test_debug_window_is_excluded_from_title_match() -> None:
    wm = GameWindowManager(WindowConfig())
    assert wm._match("RDR2 AI - Debug") is False
    assert wm._match("rdr2 ai - debug") is False
    assert wm._match("Red Dead Redemption 2") is True
    assert wm._match("RDR2") is True


def test_exact_match_rejects_lookalike_titles() -> None:
    wm = GameWindowManager(WindowConfig())
    assert wm._match(
        "I Installed 70 Red Dead Redemption 2 Mods.. - YouTube - Google Chrome"
    ) is False
    assert wm._match("Red Dead Redemption 2 - Photo Mode") is False
    assert wm._match("my RDR2 shortcuts") is False


def test_substring_match_mode_accepts_partial_titles() -> None:
    wm = GameWindowManager(WindowConfig(match="substring"))
    assert wm._match("Red Dead Redemption 2 - Photo Mode") is True
    assert wm._match("completely unrelated title") is False


def test_invalid_match_mode_rejected() -> None:
    cfg = AppConfig()
    cfg.window.match = "fuzzy"
    with pytest.raises(ConfigError, match="window.match"):
        cfg.validate()


def test_wait_for_window_times_out() -> None:
    cfg = WindowConfig(title_patterns=["definitely-not-a-real-window-xyz"])
    wm = GameWindowManager(cfg)
    t0 = time.monotonic()
    assert wm.wait_for_window(timeout_s=0.3, poll_s=0.05) is None
    assert time.monotonic() - t0 < 3.0


def test_is_alive_and_foreground_with_bogus_handle() -> None:
    wm = GameWindowManager(WindowConfig())
    assert wm.is_alive(0) is False
    assert wm.is_foreground(0) is False
    assert wm.focus(0, timeout_s=0.05) is False


def test_dpi_awareness_reports_a_mode() -> None:
    mode = enable_dpi_awareness()
    assert isinstance(mode, str)
    assert mode


def test_focus_check_is_false_without_window() -> None:
    cfg = WindowConfig(title_patterns=["definitely-not-a-real-window-xyz"])
    wm = GameWindowManager(cfg)
    check = make_focus_check(wm, focus_before_input=True, focus_timeout_s=0.05)
    assert check() is False


def test_window_info_to_dict() -> None:
    info = WindowInfo(
        hwnd=42,
        title="Red Dead Redemption 2",
        rect=Rect(0, 0, 100, 100),
        client=Rect(0, 0, 100, 80),
        focused=True,
        minimized=False,
        visible=True,
    )
    data = info.to_dict()
    assert data["title"] == "Red Dead Redemption 2"
    assert data["size"] == [100, 80]
    assert info.width == 100 and info.height == 80
