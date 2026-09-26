"""Tests for the Phase 5 mission/task framework."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import AppConfig, ConfigError, MissionConfig, load_config
from src.control.locomotion import LocomotionController
from src.mission.runner import MissionRunner
from src.mission.tasks import Task, parse_tasks, state_field
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState
from tests.test_locomotion import FakeInput


def make_runner(
    tasks: list[dict], *, loop: bool = False, frame: np.ndarray | None = None
) -> tuple[MissionRunner, StatusTracker, FakeInput]:
    cfg = AppConfig()
    mission = MissionConfig(enabled=True, loop=loop, tasks=tasks)
    fake = FakeInput()
    loco = LocomotionController(cfg.control, fake)
    runner = MissionRunner(mission, cfg.control, loco, [0.0, 0.0, 1.0, 1.0])
    return runner, StatusTracker(5), fake


def run(
    runner: MissionRunner,
    status: StatusTracker,
    steps: int = 100,
    frame: np.ndarray | None = None,
    game_state: GameState | None = None,
    start: float = 0.0,
    step: float = 0.5,
) -> float:
    now = start
    for _ in range(steps):
        runner.step(now, status, frame, game_state)
        now += step
    return now


def static_frame(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


class TestParsing:
    def test_all_kinds_parse(self) -> None:
        tasks = parse_tasks([
            {"log": {"text": "hi"}},
            {"turn": {"deg": -45}},
            {"walk": {"meters": 3.0, "bearing": 90}},
            {"wait": {"seconds": 0.5}},
            {"wait_for": {"field": "environment.wilderness", "equals": True,
                          "timeout_s": 10}},
            {"interact": {}},
        ])
        assert [t.kind for t in tasks] == [
            "log", "turn", "walk", "wait", "wait_for", "interact"
        ]
        assert tasks[2].bearing == 90.0
        assert tasks[4].timeout_s == 10.0

    def test_null_value_becomes_empty_mapping(self) -> None:
        tasks = parse_tasks([{"interact": None}])
        assert tasks[0].kind == "interact"

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown task 'dance'"):
            parse_tasks([{"dance": {}}])

    def test_multi_key_task_rejected(self) -> None:
        with pytest.raises(ValueError, match="single-key"):
            parse_tasks([{"turn": {"deg": 1}, "wait": {"seconds": 1}}])

    def test_turn_requires_deg(self) -> None:
        with pytest.raises(ValueError, match="requires a 'deg'"):
            parse_tasks([{"turn": {}}])

    def test_walk_rejects_nonpositive_meters(self) -> None:
        with pytest.raises(ValueError, match="meters must be > 0"):
            parse_tasks([{"walk": {"meters": 0}}])

    def test_wait_rejects_nonpositive_seconds(self) -> None:
        with pytest.raises(ValueError, match="seconds must be > 0"):
            parse_tasks([{"wait": {"seconds": -1}}])

    def test_wait_for_requires_field_and_equals(self) -> None:
        with pytest.raises(ValueError, match="requires 'field' and 'equals'"):
            parse_tasks([{"wait_for": {"field": "x.y"}}])

    def test_wait_for_rejects_bad_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout_s must be > 0"):
            parse_tasks([{"wait_for": {"field": "x.y", "equals": 1,
                                       "timeout_s": 0}}])

    def test_log_requires_string_text(self) -> None:
        with pytest.raises(ValueError, match="text must be a string"):
            parse_tasks([{"log": {"text": 5}}])

    def test_error_names_task_index(self) -> None:
        with pytest.raises(ValueError, match=r"mission\.tasks\[1\]"):
            parse_tasks([{"log": {"text": "ok"}}, {"bogus": {}}])

    def test_state_field_reads_nested_path(self) -> None:
        gs = GameState()
        assert state_field(gs, "environment.water_detected") is False
        assert state_field(gs, "player.health") is None
        assert state_field(gs, "nope.nope") is None


class TestRunner:
    def test_log_tasks_complete_in_order(self) -> None:
        runner, status, _fake = make_runner(
            [{"log": {"text": "one"}}, {"log": {"text": "two"}}]
        )
        run(runner, status, steps=5)
        snap = status.snapshot()
        assert runner.done is True
        assert snap.goal == "MISSION DONE"
        assert "2 tasks" in snap.message

    def test_turn_accumulates_clamped_steps(self) -> None:
        runner, status, fake = make_runner([{"turn": {"deg": 90}}])
        runner.step(0.0, status)
        assert ("mouse", 30 * 12, 0) in fake.events
        assert runner.heading_deg == 30.0
        run(runner, status, steps=5)
        assert runner.done is True
        assert runner.heading_deg == pytest.approx(90.0, abs=1.0)

    def test_walk_completes_dead_reckoned(self) -> None:
        runner, status, _fake = make_runner([{"walk": {"meters": 2.0}}])
        run(runner, status, steps=40)
        assert runner.done is True

    def test_walk_without_bearing_keeps_heading(self) -> None:
        runner, status, _fake = make_runner(
            [{"turn": {"deg": 90}}, {"walk": {"meters": 1.0}}]
        )
        run(runner, status, steps=60)
        assert runner.done is True
        assert runner.heading_deg == pytest.approx(90.0, abs=1.0)

    def test_wait_holds_until_deadline(self) -> None:
        runner, status, _fake = make_runner([{"wait": {"seconds": 2.0}}])
        runner.step(0.0, status)
        assert runner.done is False
        runner.step(1.0, status)
        assert runner.done is False
        runner.step(2.5, status)
        assert runner.done is True

    def test_wait_for_satisfied_immediately(self) -> None:
        runner, status, _fake = make_runner(
            [{"wait_for": {"field": "environment.water_detected",
                           "equals": False, "timeout_s": 5}}]
        )
        runner.step(0.0, status, None, GameState())
        assert runner.done is True

    def test_wait_for_timeout_fails_mission(self) -> None:
        runner, status, _fake = make_runner(
            [{"wait_for": {"field": "environment.water_detected",
                           "equals": True, "timeout_s": 2.0}}]
        )
        run(runner, status, steps=10, game_state=GameState())
        snap = status.snapshot()
        assert runner.failed is True
        assert snap.goal == "MISSION FAILED"
        assert "timed out" in snap.message

    def test_wait_for_without_timeout_waits(self) -> None:
        runner, status, _fake = make_runner(
            [{"wait_for": {"field": "environment.water_detected", "equals": True}}]
        )
        run(runner, status, steps=10, game_state=GameState())
        assert runner.failed is False
        assert runner.done is False

    def test_interact_taps_key(self) -> None:
        runner, status, fake = make_runner([{"interact": {}}])
        run(runner, status, steps=3)
        assert ("press", "e") in fake.events
        assert runner.done is True

    def test_refused_interact_fails_mission(self) -> None:
        class RefusingFake(FakeInput):
            def press(self, key: str, hold_ms: int | None = None) -> bool:
                return False

        cfg = AppConfig()
        mission = MissionConfig(enabled=True, tasks=[{"interact": {}}])
        fake = RefusingFake()
        loco = LocomotionController(cfg.control, fake)
        runner = MissionRunner(mission, cfg.control, loco, None)
        status = StatusTracker(5)
        runner.step(0.0, status)
        snap = status.snapshot()
        assert runner.failed is True
        assert snap.goal == "MISSION FAILED"
        assert "refused" in snap.message
        assert loco.active is False

    def test_blocked_walk_fails_mission(self) -> None:
        runner, status, _fake = make_runner([{"walk": {"meters": 50.0}}])
        frame = static_frame()
        now = 0.0
        for _ in range(60):
            runner.step(now, status, frame)
            if runner.failed:
                break
            now += 0.5
        assert runner.failed is True
        assert "blocked" in (status.snapshot().message or "")

    def test_loop_restarts_after_completion(self) -> None:
        runner, status, _fake = make_runner(
            [{"log": {"text": "round"}}], loop=True
        )
        run(runner, status, steps=6)
        assert runner.done is False
        assert status.snapshot().goal == "MISSION LOOP"
        assert "round" in (status.snapshot().message or "")

    def test_failure_is_persistent(self) -> None:
        runner, status, _fake = make_runner(
            [{"wait_for": {"field": "environment.water_detected",
                           "equals": True, "timeout_s": 1.0}},
             {"log": {"text": "never"}}]
        )
        run(runner, status, steps=6, game_state=GameState())
        assert runner.failed is True
        snap = status.snapshot()
        assert snap.goal == "MISSION FAILED"
        assert runner.tasks_completed == 0

    def test_status_shows_task_progress(self) -> None:
        runner, status, _fake = make_runner(
            [{"wait": {"seconds": 100.0}}, {"log": {"text": "later"}}]
        )
        runner.step(0.0, status)
        assert status.snapshot().goal == "MISSION 1/2"
        assert runner.current == "task 1/2"


class TestMissionConfig:
    def test_defaults_disabled(self) -> None:
        cfg = AppConfig()
        assert cfg.mission.enabled is False
        assert cfg.mission.tasks == []

    def test_enabled_without_tasks_rejected(self) -> None:
        cfg = AppConfig()
        cfg.mission.enabled = True
        with pytest.raises(ConfigError, match="mission.enabled"):
            cfg.validate()

    def test_bad_task_rejected_by_validate(self) -> None:
        cfg = AppConfig()
        cfg.mission.tasks = [{"dodge": {}}]
        with pytest.raises(ConfigError, match="unknown task"):
            cfg.validate()

    def test_valid_mission_passes(self) -> None:
        cfg = AppConfig()
        cfg.mission.enabled = True
        cfg.mission.tasks = [
            {"log": {"text": "go"}},
            {"walk": {"meters": 5.0}},
        ]
        cfg.validate()

    def test_yaml_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({
            "mission": {"enabled": True, "tasks": [{"wait": {"seconds": 1.0}}]}
        }))
        cfg = load_config(path)
        assert cfg.mission.enabled is True
        assert len(cfg.mission.tasks) == 1

    def test_task_labels(self) -> None:
        tasks = parse_tasks([
            {"turn": {"deg": -10}}, {"walk": {"meters": 3.0}},
            {"wait": {"seconds": 2.0}},
            {"wait_for": {"field": "x.y", "equals": 1}},
        ])
        assert tasks[0].label == "turn:-10deg"
        assert tasks[1].label == "walk:3.0m @keep"
        assert tasks[2].label == "wait:2.0s"
        assert tasks[3].label == "wait_for:x.y"
        assert Task(kind="log", text="m").label == "log:m"
