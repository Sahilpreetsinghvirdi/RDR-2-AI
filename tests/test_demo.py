"""Phase 1 demo sequence tests (inputs are faked)."""

from __future__ import annotations

from src.demo import Phase1Demo
from src.state.agent_status import StatusTracker
from tests.conftest import make_controller


def run_demo(app_config, fake_keyboard, fake_mouse, ensure_focus=lambda: True):
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    results: list[tuple[str, bool, float]] = []
    demo = Phase1Demo(
        ctl,
        ensure_focus=ensure_focus,
        on_result=lambda label, ok, ms: results.append((label, ok, ms)),
        settle_s=0.0,
    )
    status = StatusTracker()
    guard = 0
    while demo.step(status) and guard < 100:
        guard += 1
    return demo, results, status


def test_demo_runs_all_steps(app_config, fake_keyboard, fake_mouse) -> None:
    demo, results, status = run_demo(app_config, fake_keyboard, fake_mouse)
    assert demo.active is False
    assert len(results) == 6
    assert all(ok for _, ok, _ in results)
    assert demo.current == "complete"
    assert "complete:" in status.snapshot().demo_step


def test_demo_sends_expected_inputs(app_config, fake_keyboard, fake_mouse) -> None:
    run_demo(app_config, fake_keyboard, fake_mouse)
    down_keys = [k for kind, k in fake_keyboard.events if kind == "down"]
    assert "w" in down_keys
    assert "shift" in down_keys
    moves = [e for e in fake_mouse.events if e[0] == "move"]
    assert ("move", 150, 0) in moves
    assert ("move", -150, 0) in moves
    up_keys = [k for kind, k in fake_keyboard.events if kind == "up"]
    assert set(down_keys) == set(up_keys)


def test_demo_records_failed_focus(app_config, fake_keyboard, fake_mouse) -> None:
    demo, results, _ = run_demo(
        app_config, fake_keyboard, fake_mouse, ensure_focus=lambda: False
    )
    assert results[0] == ("focus game window", False, results[0][2])
    assert len(results) == 6


def test_demo_current_label(app_config, fake_keyboard, fake_mouse) -> None:
    ctl = make_controller(app_config, fake_keyboard, fake_mouse)
    demo = Phase1Demo(ctl, ensure_focus=lambda: True, settle_s=0.0)
    assert demo.current == "focus game window"
    assert demo.active is True
