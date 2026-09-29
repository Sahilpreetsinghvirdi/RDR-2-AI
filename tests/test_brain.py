"""Local vision brain: parsing, prompting, threaded decisions, config."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.ai.brain.planner import BrainPlanner
from src.ai.brain.prompt import ACTIONS, build_prompt, parse_decision
from src.config import AppConfig, BrainConfig, ConfigError
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState


class FakeLocomotion:
    def __init__(self) -> None:
        self.moves: list[tuple[str, float, bool]] = []
        self.turns: list[float] = []
        self.taps: list[str] = []
        self.stops = 0

    @property
    def active(self) -> bool:
        return False

    def move(self, direction: str, duration_s: float, *,
             sprint: bool = False) -> bool:
        self.moves.append((direction, duration_s, sprint))
        return True

    def turn(self, degrees: float) -> bool:
        self.turns.append(degrees)
        return True

    def tap(self, role: str) -> bool:
        self.taps.append(role)
        return True

    def tick(self) -> None:
        pass

    def stop(self) -> None:
        self.stops += 1


def _frame() -> np.ndarray:
    return np.full((90, 160, 3), 120, dtype=np.uint8)


def _planner(answer: str = '{"action": "forward", "reason": "ride on"}',
             **kwargs: object) -> tuple[BrainPlanner, FakeLocomotion]:
    base: dict[str, object] = {"enabled": True, "interval_s": 3600.0}
    base.update(kwargs)
    cfg = BrainConfig(**base)  # type: ignore[arg-type]
    loco = FakeLocomotion()
    planner = BrainPlanner(
        cfg, loco,  # type: ignore[arg-type]
        ask_fn=lambda _prompt, _image: answer,
    )
    return planner, loco


def test_parse_clean_json() -> None:
    action, reason, clean = parse_decision('{"action": "forward", "reason": "go"}')
    assert (action, reason, clean) == ("forward", "go", True)


def test_parse_fenced_json() -> None:
    text = '```json\n{"action": "turn_left", "reason": "look"}\n```'
    action, _, clean = parse_decision(text)
    assert (action, clean) == ("turn_left", True)


def test_parse_garbage_falls_back_to_noop() -> None:
    for bad in ("", "no json here", "[1, 2]", '{"action": "fly"}'):
        action, _, clean = parse_decision(bad)
        assert action == "noop"
        assert clean is False


def test_parse_truncated_json_repairs_action() -> None:
    text = ' {"action": "forward", "reason": "...", "distance": 0.85, '
    action, _, clean = parse_decision(text)
    assert action == "forward"
    assert clean is False


def test_parse_missing_reason_defaults() -> None:
    action, reason, clean = parse_decision('prefix {"action": "whistle"} suffix')
    assert action == "whistle"
    assert clean is True
    assert reason == ""


def test_parse_action_aliases() -> None:
    for raw, expected in [("walk", "forward"), ("run", "sprint"),
                          ("left", "turn_left"), ("stop", "noop")]:
        action, _, _ = parse_decision(f'{{"action": "{raw}"}}')
        assert action == expected


def test_prompt_lists_actions_and_state() -> None:
    prompt = build_prompt({"threat": "none", "objective": "Go to Dutch"})
    assert "forward" in prompt and "whistle" in prompt
    assert "Go to Dutch" in prompt
    assert "JSON only" in prompt


def test_decision_executes_once() -> None:
    planner, loco = _planner()
    try:
        status = StatusTracker(phase=11)
        planner.observe(_frame())
        planner._cycle()  # noqa: SLF001 - drive the thinker directly
        assert planner.pending is True
        now = time.monotonic()
        planner.step(now, status, _frame(), GameState())
        assert loco.moves == [("forward", 0.5, False)]
        assert planner.pending is False
        assert status.snapshot().action == "brain:forward"
        planner.step(now + 1.0, status, _frame(), GameState())
        assert len(loco.moves) == 1
    finally:
        planner.close()


def test_stale_decision_is_dropped() -> None:
    planner, loco = _planner(stale_after_s=5.0)
    try:
        status = StatusTracker(phase=11)
        planner.observe(_frame())
        planner._cycle()  # noqa: SLF001
        planner.step(time.monotonic() + 100.0, status, _frame(), GameState())
        assert loco.moves == []
        assert status.snapshot().action == "brain:wait"
    finally:
        planner.close()


def test_brain_takes_planner_branch(tmp_path, monkeypatch) -> None:
    from src.main import main
    from tests.test_main_guard import FakeWindowManager

    class RecordingBrain:
        instances: list[RecordingBrain] = []

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.steps = 0
            RecordingBrain.instances.append(self)

        def step(self, now: float, status: object, frame: object | None = None,
                 game_state: object | None = None) -> None:
            self.steps += 1

        def close(self) -> None:
            pass

    RecordingBrain.instances = []
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join([
            "brain:",
            "  enabled: true",
            "  interval_s: 3600.0",
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
    monkeypatch.setattr("src.main.BrainPlanner", RecordingBrain)
    rc = main(["--config", str(cfg_path), "--headless", "--duration", "2"])
    assert rc == 0
    assert RecordingBrain.instances, "BrainPlanner was not constructed"
    assert RecordingBrain.instances[0].steps > 0


def test_turn_and_tap_actions() -> None:
    for answer, check in [
        ('{"action": "turn_left", "reason": "t"}',
         lambda loco: loco.turns == [-15.0]),
        ('{"action": "sprint", "reason": "t"}',
         lambda loco: loco.moves == [("forward", 0.5, True)]),
        ('{"action": "whistle", "reason": "t"}',
         lambda loco: loco.taps == ["whistle"]),
        ('{"action": "noop", "reason": "t"}',
         lambda loco: loco.moves == [] and loco.turns == [] and loco.taps == []),
    ]:
        planner, loco = _planner(answer)
        try:
            planner.observe(_frame())
            planner._cycle()  # noqa: SLF001
            planner.step(
                time.monotonic(), StatusTracker(phase=11), _frame(), GameState()
            )
            assert check(loco), answer
        finally:
            planner.close()


def test_model_error_never_kills_thread() -> None:
    cfg = BrainConfig(enabled=True, interval_s=0.05)
    loco = FakeLocomotion()
    planner = BrainPlanner(cfg, loco, ask_fn=_boom)  # type: ignore[arg-type]
    try:
        planner.observe(_frame())
        deadline = time.monotonic() + 5.0
        while planner.failures == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert planner.failures >= 1
        assert planner._thread.is_alive()  # noqa: SLF001
        assert planner.decisions == 0
    finally:
        planner.close()


def _boom(_prompt: str, _image: str) -> str:
    raise RuntimeError("model is down")


def test_brain_config_conflicts() -> None:
    for other in ("mission", "control", "story", "rl"):
        cfg = AppConfig()
        cfg.brain.enabled = True
        if other == "mission":
            cfg.mission.enabled = True
            cfg.mission.tasks = [{"kind": "log", "text": "x"}]
        elif other == "control":
            cfg.control.enabled = True
        elif other == "story":
            cfg.story = _story_with_mission()
        else:
            cfg.rl.enabled = True
        with pytest.raises(ConfigError, match="brain.enabled conflicts"):
            cfg.validate()


def _story_with_mission():
    from src.config import StoryConfig

    return StoryConfig(enabled=True, missions=[{"name": "x"}])


def test_brain_config_values_validated() -> None:
    cfg = AppConfig()
    cfg.brain.enabled = True
    cfg.brain.model = ""
    with pytest.raises(ConfigError, match="brain.model"):
        cfg.validate()
    cfg2 = AppConfig()
    cfg2.brain.enabled = True
    cfg2.brain.interval_s = 0.0
    with pytest.raises(ConfigError, match=r"brain\.interval_s"):
        cfg2.validate()


def test_brain_defaults_disabled() -> None:
    assert BrainConfig().enabled is False
    assert AppConfig().brain.enabled is False
    assert set(ACTIONS) >= {"forward", "noop", "interact", "whistle"}


def test_previous_action_feeds_next_prompt() -> None:
    prompts: list[str] = []

    def ask(prompt: str, _image: str) -> str:
        prompts.append(prompt)
        return '{"action": "forward", "reason": "ride on"}'

    cfg = BrainConfig(enabled=True, interval_s=3600.0)
    loco = FakeLocomotion()
    planner = BrainPlanner(cfg, loco, ask_fn=ask)  # type: ignore[arg-type]
    try:
        status = StatusTracker(phase=11)
        planner.observe(_frame())
        planner._cycle()  # noqa: SLF001
        assert "- previous:" not in prompts[0]
        planner.step(time.monotonic(), status, _frame(), GameState())
        planner.observe(_frame())
        planner._cycle()  # noqa: SLF001
        assert "- previous: forward (ride on)" in prompts[1]
    finally:
        planner.close()
