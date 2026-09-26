"""Tests for Phase 4 locomotion control and the scripted autopilot planner."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import AppConfig, ConfigError, load_config
from src.control.locomotion import LocomotionController
from src.planning.scripted import SCRIPT, ScriptedPlanner
from src.state.agent_status import StatusTracker


class FakeInput:
    """Records input calls instead of sending anything."""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.fail_keys: set[str] = set()

    def key_down(self, key: str) -> bool:
        if key in self.fail_keys:
            self.events.append(("refused", key))
            return False
        self.events.append(("down", key))
        return True

    def key_up(self, key: str) -> bool:
        self.events.append(("up", key))
        return True

    def mouse_move(self, dx: int, dy: int = 0) -> bool:
        self.events.append(("mouse", dx, dy))
        return True

    def press(self, key: str, hold_ms: int | None = None) -> bool:
        self.events.append(("press", key))
        return True


def make_loco() -> tuple[LocomotionController, FakeInput]:
    fake = FakeInput()
    return LocomotionController(AppConfig().control, fake), fake


class TestLocomotion:
    def test_move_holds_direction_key(self) -> None:
        loco, fake = make_loco()
        assert loco.move("forward", 2.0) is True
        assert ("down", "w") in fake.events
        assert loco.active is True
        assert loco.label.startswith("move:forward")
        assert loco.moves_started == 1

    def test_unknown_direction_rejected(self) -> None:
        loco, fake = make_loco()
        assert loco.move("sideways", 1.0) is False
        assert fake.events == []
        assert loco.active is False

    def test_sprint_adds_and_releases_both_keys(self) -> None:
        loco, fake = make_loco()
        loco.move("forward", 1.0, sprint=True)
        assert ("down", "w") in fake.events
        assert ("down", "shift") in fake.events
        loco.stop()
        assert ("up", "w") in fake.events
        assert ("up", "shift") in fake.events
        assert loco.active is False

    def test_stop_is_idempotent(self) -> None:
        loco, fake = make_loco()
        loco.stop()
        loco.stop()
        assert fake.events == []

    def test_failed_key_down_rolls_back(self) -> None:
        loco, fake = make_loco()
        fake.fail_keys.add("w")
        assert loco.move("forward", 1.0) is False
        assert loco.active is False
        assert ("down", "w") not in fake.events

    def test_tick_releases_after_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loco, fake = make_loco()
        t = [100.0]
        monkeypatch.setattr("src.control.locomotion.time.monotonic", lambda: t[0])
        loco.move("back", 1.0)
        t[0] += 0.9
        loco.tick()
        assert loco.active is True
        t[0] += 0.2
        loco.tick()
        assert loco.active is False
        assert ("up", "s") in fake.events

    def test_duration_clamped_to_max_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loco, fake = make_loco()
        t = [100.0]
        monkeypatch.setattr("src.control.locomotion.time.monotonic", lambda: t[0])
        loco.move("forward", 600.0)  # absurd request
        t[0] += AppConfig().control.max_step_s
        loco.tick()
        assert loco.active is False

    def test_duration_clamped_to_min_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loco, _fake = make_loco()
        t = [100.0]
        monkeypatch.setattr("src.control.locomotion.time.monotonic", lambda: t[0])
        loco.move("forward", 0.001)
        t[0] += 0.05  # below min_step_s (0.1)
        loco.tick()
        assert loco.active is True

    def test_turn_clamped_per_call(self) -> None:
        loco, fake = make_loco()
        cfg = AppConfig().control
        loco.turn(10_000.0)
        assert ("mouse", int(cfg.max_turn_deg_per_tick * cfg.mouse_px_per_degree), 0) in fake.events
        assert loco.degrees_turned == pytest.approx(cfg.max_turn_deg_per_tick)

    def test_turn_converts_degrees_to_pixels(self) -> None:
        loco, fake = make_loco()
        cfg = AppConfig().control
        loco.turn(-10.0)
        assert ("mouse", int(round(-10.0 * cfg.mouse_px_per_degree)), 0) in fake.events

    def test_turn_zero_sends_nothing(self) -> None:
        loco, fake = make_loco()
        loco.turn(0.0)
        assert fake.events == []

    def test_tap_uses_configured_key(self) -> None:
        loco, fake = make_loco()
        assert loco.tap("jump") is True
        assert ("press", "space") in fake.events
        assert loco.tap("unknown_action") is False

    def test_snapshot_reports_state(self) -> None:
        loco, _fake = make_loco()
        loco.move("strafe_right", 1.0)
        snap = loco.snapshot()
        assert snap["active"] is True
        assert snap["keys"] == ["d"]
        assert snap["moves_started"] == 1


class TestPlanner:
    def make(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[ScriptedPlanner, LocomotionController, FakeInput, StatusTracker]:
        loco, fake = make_loco()
        planner = ScriptedPlanner(AppConfig().control, loco)
        t = [0.0]
        monkeypatch.setattr("src.control.locomotion.time.monotonic", lambda: t[0])
        status = StatusTracker(4)
        self._t = t  # type: ignore[attr-defined]
        return planner, loco, fake, status

    def test_first_action_is_a_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        planner, _loco, fake, status = self.make(monkeypatch)
        planner.step(0.0, status)
        assert any(e[0] == "mouse" for e in fake.events)
        assert status.snapshot().goal == "AUTOPILOT (phase 4)"

    def test_big_turn_spans_multiple_ticks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        planner, loco, fake, status = self.make(monkeypatch)
        planner.step(0.0, status)  # applies -30 of -40
        mouse_calls = sum(1 for e in fake.events if e[0] == "mouse")
        assert mouse_calls == 1
        assert planner.current.startswith("turn")
        planner.step(0.0, status)  # applies remaining -10 -> advance
        assert planner.current.startswith("move")
        assert mouse_calls + 1 == sum(1 for e in fake.events if e[0] == "mouse")

    def test_walk_waits_for_release_before_advancing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        planner, loco, fake, status = self.make(monkeypatch)
        t = self._t  # type: ignore[attr-defined]
        planner.step(0.0, status)   # turn -30/-40
        planner.step(0.0, status)   # turn done -> index moves to walk
        planner.step(0.0, status)   # walk begins
        assert loco.active is True
        assert ("down", "w") in fake.events
        planner.step(0.1, status)
        assert loco.active is True          # still walking
        assert planner.current.startswith("move")
        t[0] = 10.0                          # beyond move deadline
        planner.step(10.0, status)
        assert loco.active is False
        assert planner.current.startswith("turn")

    def test_refused_move_advances_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        planner, _loco, fake, status = self.make(monkeypatch)
        fake.fail_keys.add("w")
        planner.step(0.0, status)
        planner.step(0.0, status)   # first turn completes -> index on walk
        planner.step(0.0, status)   # move w refused -> advance same tick
        assert planner.current.startswith("turn")

    def test_full_script_cycles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        planner, _loco, fake, status = self.make(monkeypatch)
        t = self._t  # type: ignore[attr-defined]
        for _ in range(500):
            t[0] += 0.1
            planner.step(t[0], status)
            if planner.cycles >= 1:
                break
        assert planner.cycles >= 1
        pressed = {e[1] for e in fake.events if e[0] == "press"}
        assert "space" in pressed                      # jump tap ran
        downs = [e[1] for e in fake.events if e[0] == "down"]
        for key in ("w", "a", "s", "shift"):
            assert key in downs
        assert len(SCRIPT) > 0

    def test_turn_refused_does_not_stall(self, monkeypatch: pytest.MonkeyPatch) -> None:
        planner, _loco, fake, status = self.make(monkeypatch)

        def refuse_mouse(_dx: int, _dy: int = 0) -> bool:
            fake.events.append(("mouse_refused",))
            return False

        fake.mouse_move = refuse_mouse  # type: ignore[method-assign]
        planner.step(0.0, status)
        assert planner.current.startswith("move")


class TestControlConfig:
    def test_defaults_are_safe(self) -> None:
        cfg = AppConfig().control
        assert cfg.enabled is False
        assert cfg.keys["forward"] == "w"
        assert cfg.keys["sprint"] == "shift"

    def test_yaml_section_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"control": {"enabled": True}}))
        cfg = load_config(path)
        assert cfg.control.enabled is True

    def test_missing_key_rejected(self) -> None:
        cfg = AppConfig()
        del cfg.control.keys["forward"]
        with pytest.raises(ConfigError, match="control.keys missing"):
            cfg.validate()

    def test_unknown_key_rejected(self) -> None:
        cfg = AppConfig()
        cfg.control.keys["drift"] = "z"
        with pytest.raises(ConfigError, match="control.keys unknown"):
            cfg.validate()

    def test_nonpositive_sensitivity_rejected(self) -> None:
        cfg = AppConfig()
        cfg.control.mouse_px_per_degree = 0.0
        with pytest.raises(ConfigError, match="mouse_px_per_degree"):
            cfg.validate()

    def test_bad_step_clamps_rejected(self) -> None:
        cfg = AppConfig()
        cfg.control.min_step_s = 5.0
        cfg.control.max_step_s = 1.0
        with pytest.raises(ConfigError, match="min_step_s"):
            cfg.validate()

    def test_negative_turn_clamp_rejected(self) -> None:
        cfg = AppConfig()
        cfg.control.max_turn_deg_per_tick = -1.0
        with pytest.raises(ConfigError, match="max_turn_deg_per_tick"):
            cfg.validate()
