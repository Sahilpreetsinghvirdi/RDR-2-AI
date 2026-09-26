"""Tests for the Phase 4 route navigation planner."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import AppConfig, ConfigError
from src.control.locomotion import LocomotionController
from src.planning.route import RoutePlanner, angle_delta
from src.planning.scripted import ScriptedPlanner
from src.state.agent_status import StatusTracker
from tests.test_locomotion import FakeInput


def make(
    cfg_overrides: dict | None = None
) -> tuple[RoutePlanner, LocomotionController, FakeInput, StatusTracker]:
    cfg = AppConfig()
    if cfg_overrides:
        for key, value in cfg_overrides.items():
            setattr(cfg.control.nav, key, value)
    fake = FakeInput()
    loco = LocomotionController(cfg.control, fake)
    planner = RoutePlanner(cfg.control, loco, minimap_frac=[0.0, 0.0, 1.0, 1.0])
    return planner, loco, fake, StatusTracker(4)


def run_until(
    planner: RoutePlanner,
    status: StatusTracker,
    condition,
    frame: np.ndarray | None = None,
    start: float = 0.0,
    step: float = 0.5,
    limit: int = 400,
) -> float:
    now = start
    for _ in range(limit):
        planner.step(now, status, frame)
        if condition():
            return now
        now += step
    raise AssertionError("condition not reached within step limit")


def static_frame(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


class TestAngleDelta:
    def test_simple_turns(self) -> None:
        assert angle_delta(10.0, 0.0) == 10.0
        assert angle_delta(0.0, 10.0) == -10.0

    def test_wraps_at_180(self) -> None:
        assert angle_delta(170.0, -170.0) == -20.0
        assert angle_delta(-170.0, 170.0) == 20.0
        assert angle_delta(180.0, 0.0) in (-180.0, 180.0)


class TestRoutePlanner:
    def test_initial_turn_applies_clamped_mouse(self) -> None:
        planner, _loco, fake, status = make({"legs": [[90.0, 5.0]]})
        planner.step(0.0, status)
        assert ("mouse", 30 * 12, 0) in fake.events  # clamp 30 deg * 12 px
        assert planner.heading_deg == 30.0
        assert status.snapshot().action.startswith("turn:")

    def test_faced_leg_switches_to_walk(self) -> None:
        planner, _loco, fake, status = make({"legs": [[0.0, 5.0]]})
        planner.step(0.0, status)   # err 0 -> walk phase
        assert status.snapshot().action == "walk:start"
        planner.step(0.5, status)
        assert ("down", "w") in fake.events

    def test_walk_progress_completes_leg(self) -> None:
        planner, _loco, _fake, status = make({"legs": [[0.0, 2.0]]})
        planner.step(0.0, status)
        run_until(planner, status, lambda: planner.legs_completed >= 1, start=0.5)
        assert planner.legs_completed == 1

    def test_multi_leg_route_arrives(self) -> None:
        planner, loco, _fake, status = make({"legs": [[0.0, 1.0], [180.0, 1.0]]})
        run_until(planner, status, lambda: "ARRIVED" in status.snapshot().goal)
        snap = status.snapshot()
        assert planner.legs_completed == 2
        assert snap.goal == "ROUTE ARRIVED"
        assert "route complete" in snap.message
        assert loco.active is False

    def test_sprint_used_when_far_remaining(self) -> None:
        planner, _loco, fake, status = make({"legs": [[0.0, 20.0]], "sprint_after_m": 6.0})
        planner.step(0.0, status)
        planner.step(0.5, status)
        assert ("down", "shift") in fake.events

    def test_no_sprint_when_close(self) -> None:
        planner, _loco, fake, status = make({"legs": [[0.0, 3.0]], "sprint_after_m": 6.0})
        planner.step(0.0, status)
        planner.step(0.5, status)
        assert ("down", "shift") not in fake.events

    def test_static_minimap_blocks_walk(self) -> None:
        planner, loco, _fake, status = make(
            {"legs": [[0.0, 50.0]], "stall_timeout_s": 2.0}
        )
        frame = static_frame()
        planner.step(0.0, status, frame)
        planner.step(0.5, status, frame)
        run_until(planner, status, lambda: planner.blocked, frame=frame, start=1.0)
        assert loco.active is False
        assert status.snapshot().action == "blocked"
        assert "no minimap motion" in (status.snapshot().message or "")

    def test_moving_minimap_keeps_walking(self) -> None:
        planner, _loco, _fake, status = make(
            {"legs": [[0.0, 50.0]], "stall_timeout_s": 2.0}
        )
        rng = np.random.default_rng(9)
        base = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        planner.step(0.0, status, base)
        for i in range(1, 8):
            shifted = np.roll(base, i * 4, axis=0)
            planner.step(i * 0.5, status, shifted)
            assert planner.blocked is False

    def test_block_resumes_when_motion_returns(self) -> None:
        planner, loco, _fake, status = make(
            {"legs": [[0.0, 50.0]], "stall_timeout_s": 1.0}
        )
        frame = static_frame()
        planner.step(0.0, status, frame)
        planner.step(0.5, status, frame)
        run_until(planner, status, lambda: planner.blocked, frame=frame, start=1.0)
        rng = np.random.default_rng(4)
        base = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        planner.step(20.0, status, base)
        planner.step(20.5, status, np.roll(base, 6, axis=0))
        assert planner.blocked is False
        assert loco.active is True

    def test_no_frame_means_pure_dead_reckoning(self) -> None:
        planner, _loco, _fake, status = make({"legs": [[0.0, 3.0]]})
        planner.step(0.0, status, None)
        run_until(planner, status, lambda: planner.legs_completed >= 1, start=0.5)
        assert planner.blocked is False

    def test_parity_with_scripted_step_signature(self) -> None:
        loco = LocomotionController(AppConfig().control, FakeInput())
        scripted = ScriptedPlanner(AppConfig().control, loco)
        scripted.step(0.0, StatusTracker(4), None)  # frame arg accepted


class TestNavConfig:
    def test_defaults(self) -> None:
        cfg = AppConfig().control
        assert cfg.mode == "script"
        assert cfg.nav.legs == []

    def test_route_mode_requires_legs(self) -> None:
        cfg = AppConfig()
        cfg.control.mode = "route"
        with pytest.raises(ConfigError, match="legs"):
            cfg.validate()

    def test_invalid_mode_rejected(self) -> None:
        cfg = AppConfig()
        cfg.control.mode = "wanderg"
        with pytest.raises(ConfigError, match="control.mode"):
            cfg.validate()

    def test_bad_leg_shape_rejected(self) -> None:
        cfg = AppConfig()
        cfg.control.mode = "route"
        cfg.control.nav.legs = [[90.0, 5.0, 1.0]]
        with pytest.raises(ConfigError, match="legs\\[0\\]"):
            cfg.validate()

    def test_nonpositive_leg_distance_rejected(self) -> None:
        cfg = AppConfig()
        cfg.control.nav.legs = [[0.0, 0.0]]
        with pytest.raises(ConfigError, match="meters must be > 0"):
            cfg.validate()

    def test_nonpositive_speed_rejected(self) -> None:
        cfg = AppConfig()
        cfg.control.nav.walk_speed_mps = 0.0
        with pytest.raises(ConfigError, match="walk_speed_mps"):
            cfg.validate()

    def test_valid_route_config_passes(self) -> None:
        cfg = AppConfig()
        cfg.control.mode = "route"
        cfg.control.nav.legs = [[0.0, 8.0], [-90.0, 4.0]]
        cfg.validate()  # should not raise
