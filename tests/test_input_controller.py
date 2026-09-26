"""InputController guarding, tracking and release_all behaviour."""

from __future__ import annotations

import threading

import pytest

from tests.conftest import FakeKeyboard, FakeMouse, make_controller


def test_press_orders_down_then_up(app_config, fake_keyboard, fake_mouse) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    assert ctl.press("w", hold_ms=10) is True
    assert fake_keyboard.events == [("down", "w"), ("up", "w")]
    assert ctl.held_keys == []
    assert ctl.stats.sent == 2


def test_key_down_tracks_until_key_up(app_config, fake_keyboard, fake_mouse) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    assert ctl.key_down("shift") is True
    assert ctl.held_keys == ["shift"]
    assert ctl.key_up("shift") is True
    assert ctl.held_keys == []


def test_guard_blocks_sends(app_config, fake_keyboard, fake_mouse) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse, allow=lambda: False)
    assert ctl.key_down("w") is False
    assert ctl.mouse_move(10, 10) is False
    assert fake_keyboard.events == []
    assert ctl.stats.refused > 0


def test_stop_event_blocks_sends(app_config, fake_keyboard, fake_mouse) -> None:
    stop = threading.Event()
    stop.set()
    ctl = make_controller(app_config, fake_keyboard, fake_mouse, stop_event=stop)
    assert ctl.key_down("w") is False
    assert fake_keyboard.events == []


def test_release_all_frees_keys_and_buttons(
    app_config, fake_keyboard, fake_mouse
) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    ctl.key_down("w")
    ctl.key_down("a")
    ctl.mouse_down("left")
    fake_keyboard.events.clear()
    fake_mouse.events.clear()

    ctl.release_all()

    assert ("up", "w") in fake_keyboard.events
    assert ("up", "a") in fake_keyboard.events
    assert ("up", "left") in fake_mouse.events
    assert ctl.held_keys == []
    assert ctl.held_buttons == []
    assert ctl.stats.released == 3


def test_release_all_bypasses_guard(app_config, fake_keyboard, fake_mouse) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse, allow=lambda: True)
    ctl.key_down("w")
    ctl._allow = lambda: False  # type: ignore[method-assign]
    ctl.release_all()
    assert ctl.held_keys == []


def test_release_all_without_held_inputs_is_silent(
    app_config, fake_keyboard, fake_mouse
) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    ctl.release_all()
    assert fake_keyboard.events == []


def test_hold_aborts_when_guard_drops(app_config, fake_keyboard, fake_mouse) -> None:
    calls = {"n": 0}

    def allow_once() -> bool:
        calls["n"] += 1
        return calls["n"] <= 1

    ctl = make_controller(app_config, fake_keyboard, fake_mouse, allow=allow_once)
    assert ctl.hold("w", duration_s=2.0) is False
    assert fake_keyboard.events == [("down", "w"), ("up", "w")]


def test_unknown_key_raises(app_config, fake_keyboard, fake_mouse) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    with pytest.raises(ValueError):
        ctl.key_down("not-a-key")


def test_mouse_click_tracks_button(app_config, fake_keyboard, fake_mouse) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    assert ctl.mouse_click("right", hold_ms=5) is True
    assert fake_mouse.events[0] == ("down", "right")
    assert fake_mouse.events[-1] == ("up", "right")
    assert ctl.held_buttons == []


def test_failed_injection_counts_as_failure(app_config, fake_keyboard, fake_mouse) -> None:
    fake_keyboard.fail = True
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    assert ctl.key_down("w") is False
    assert ctl.stats.failed == 1
    assert ctl.held_keys == []


def test_snapshot_reports_held(app_config, fake_keyboard, fake_mouse) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    ctl.key_down("w")
    ctl.mouse_down("left")
    snap = ctl.snapshot()
    assert snap["held_keys"] == ["w"]
    assert snap["held_buttons"] == ["left"]


def test_press_interrupted_by_stop_still_releases(
    app_config, fake_keyboard, fake_mouse
) -> None:
    stop = threading.Event()
    ctl = make_controller(app_config, fake_keyboard, fake_mouse, stop_event=stop)
    stop.set()
    assert ctl.press("w", hold_ms=5000) is False
    assert fake_keyboard.events == []


def test_fake_backends_satisfy_protocols(app_config) -> None:
    from src.input.controller import ControllerUnavailable, UnavailableController
    from src.input.keyboard import KeyboardBackend
    from src.input.mouse import MouseBackend

    kb: KeyboardBackend = FakeKeyboard()
    ms: MouseBackend = FakeMouse()
    assert kb.is_down.__self__ is kb  # type: ignore[attr-defined]
    assert ms.button_is_down("left") is False
    with pytest.raises(ControllerUnavailable):
        UnavailableController().button_down("a")
