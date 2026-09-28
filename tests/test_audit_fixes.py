"""Regression tests for the from-scratch audit: crashes, leaks, drift, staleness."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from src.ai.threat import ThreatAssessor
from src.config import (
    AppConfig,
    CombatConfig,
    ConfigError,
    ControlConfig,
    InputConfig,
    StoryConfig,
    VisionConfig,
    load_config,
)
from src.control.locomotion import LocomotionController
from src.input.input_controller import InputController
from src.input.keys import Key, normalize_key
from src.mission.runner import MissionRunner
from src.planning.route import RoutePlanner
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState
from src.vision.ocr import NullOcr
from src.vision.perception import Perception
from tests.test_hud import _boxes, _frame, draw_prompt
from tests.test_main_guard import FakeWindowManager
from tests.test_perception import FakeOcr


class FakeKeyboard:
    def __init__(self, down_results: list[bool] | None = None) -> None:
        self.down_results = list(down_results or [])
        self.held: list[str] = []
        self.ups: list[str] = []

    def down(self, key: Key) -> bool:
        name = key.name if isinstance(key, Key) else str(key)
        ok = self.down_results.pop(0) if self.down_results else True
        if ok:
            self.held.append(name)
        return ok

    def up(self, key: Key) -> bool:
        name = key.name if isinstance(key, Key) else str(key)
        self.ups.append(name)
        if name in self.held:
            self.held.remove(name)
        return True


def _input(downs: list[bool] | None = None) -> tuple[InputController, FakeKeyboard]:
    kb = FakeKeyboard(downs)
    ctrl = InputController(
        InputConfig(), allow_input=lambda: True, keyboard=kb  # type: ignore[arg-type]
    )
    return ctrl, kb


def _agent_yaml(tmp_path: Path, extra: str = "") -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join([
            extra,
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
    return cfg_path


def test_run_with_recording_enabled(tmp_path: Path, monkeypatch) -> None:
    from src.main import main

    class NoopStory:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.steps = 0

        def step(self, now: float, status: object, frame: object | None = None,
                 game_state: object | None = None) -> None:
            self.steps += 1

    cfg_path = _agent_yaml(
        tmp_path,
        "story:\n  enabled: true\n  missions:\n    - name: rec\n"
        "recording:\n  enabled: true\n",
    )
    monkeypatch.setattr("src.main.GameWindowManager", FakeWindowManager)
    monkeypatch.setattr("src.main.StoryRunner", NoopStory)
    rc = main(["--config", str(cfg_path), "--headless", "--duration", "2"])
    assert rc == 0
    sessions = sorted((tmp_path / "recordings").glob("session_*"))
    assert sessions, "no recording session was created"
    assert list(sessions[-1].iterdir()), "recording session is empty"
    meta = json.loads((sessions[-1] / "metadata.json").read_text(encoding="utf-8"))
    assert meta["autopilot"] is True


def test_unfocused_window_skips_planner(tmp_path: Path, monkeypatch) -> None:
    from src.capture.window_manager import Rect, WindowInfo
    from src.main import main

    class UnfocusedWindowManager(FakeWindowManager):
        def __init__(self, cfg: object) -> None:
            self._info = WindowInfo(
                hwnd=1, title="Red Dead Redemption 2",
                rect=Rect(0, 0, 1280, 720), client=Rect(0, 0, 1280, 720),
                focused=False, minimized=False, visible=True,
            )

    class CountingPlanner:
        steps = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def step(self, now: float, status: object, frame: object | None = None,
                 game_state: object | None = None) -> None:
            CountingPlanner.steps += 1

    CountingPlanner.steps = 0
    cfg_path = _agent_yaml(
        tmp_path, "mission:\n  enabled: true\n  tasks:\n    - turn: {deg: 10}\n"
    )
    monkeypatch.setattr("src.main.GameWindowManager", UnfocusedWindowManager)
    monkeypatch.setattr("src.main.MissionRunner", CountingPlanner)
    rc = main(["--config", str(cfg_path), "--headless", "--duration", "2"])
    assert rc == 0
    assert CountingPlanner.steps == 0


def test_sprint_partial_failure_releases() -> None:
    ctrl, kb = _input([True, False])
    loco = LocomotionController(ControlConfig(), ctrl)  # type: ignore[arg-type]
    assert loco.move("forward", 0.5, sprint=True) is False
    assert kb.held == []
    assert loco.active is False
    assert kb.ups == ["w"]


def test_refused_move_holds_nothing() -> None:
    ctrl, kb = _input([False])
    loco = LocomotionController(ControlConfig(), ctrl)  # type: ignore[arg-type]
    assert loco.move("forward", 0.5) is False
    assert kb.held == []


def test_route_freezes_on_refused_input() -> None:
    refused = InputController(
        InputConfig(), allow_input=lambda: False  # type: ignore[arg-type]
    )
    loco = LocomotionController(ControlConfig(), refused)  # type: ignore[arg-type]
    route = RoutePlanner(ControlConfig(), loco)
    route.set_legs([(90.0, 100.0)])
    status = StatusTracker(phase=11)
    route.step(0.0, status)
    route.step(0.5, status)
    assert route.heading_deg == pytest.approx(0.0)
    assert route._progress_m == pytest.approx(0.0)  # noqa: SLF001


def test_mission_turn_freezes_on_refused_input() -> None:
    refused = InputController(
        InputConfig(), allow_input=lambda: False  # type: ignore[arg-type]
    )
    loco = LocomotionController(ControlConfig(), refused)  # type: ignore[arg-type]
    runner = MissionRunner(
        _mission_cfg([{"turn": {"deg": 45.0}}]), ControlConfig(), loco
    )
    status = StatusTracker(phase=11)
    runner.step(0.0, status)
    runner.step(0.1, status)
    assert runner.heading_deg == pytest.approx(0.0)
    assert runner.current == "task 1/1"


def _mission_cfg(tasks: list[dict[str, object]]):
    from src.config import MissionConfig

    return MissionConfig(enabled=True, tasks=tasks)


def _wanted_frame() -> np.ndarray:
    import cv2

    cfg = VisionConfig()
    frame = _frame(world=20)
    x, y, w, h = _boxes(cfg.hud)["wanted"]
    crop = frame[y:y + h, x:x + w]
    for i in range(2):
        cv2.circle(crop, (30 + i * 40, h // 2), 7, (255, 255, 255), -1)
    return frame


def test_wanted_clears_when_stars_gone() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=NullOcr())
    state = GameState()
    perception.apply_to_state(state, perception.process(_wanted_frame(), 1))
    assert state.threat.wanted_level == 2
    perception.apply_to_state(state, perception.process(_frame(world=20), 2))
    assert state.threat.wanted_level == 0


def test_horse_clears_when_gauges_gone() -> None:
    from src.vision.hud import scale_regions
    from tests.test_hud import draw_gauge

    cfg = VisionConfig()
    perception = Perception(cfg, ocr=NullOcr())
    frame = _frame()
    wboxes = scale_regions(cfg.world.regions, 1920, 1080)
    draw_gauge(frame, wboxes["horse_health"], 0.8)
    state = GameState()
    perception.apply_to_state(state, perception.process(frame, 1))
    assert state.horse.detected is True
    perception.apply_to_state(state, perception.process(_frame(), 2))
    assert state.horse.detected is False
    assert state.horse.health is None


def test_ammo_clears_when_text_gone() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    perception = Perception(cfg, ocr=FakeOcr(text="24"))
    frame = _frame()
    x, y, w, h = _ammo_box(cfg)
    frame[y:y + h, x:x + w] = 255
    state = GameState()
    perception.apply_to_state(state, perception.process(frame, 1))
    assert state.player.weapon_ammo == 24
    perception.apply_to_state(state, perception.process(_frame(), 2))
    assert state.player.weapon_ammo is None


def _ammo_box(cfg: VisionConfig) -> tuple[int, int, int, int]:
    from src.vision.hud import scale_regions

    return scale_regions(cfg.world.regions, 1920, 1080)["weapon"]


def test_prompt_text_clears_when_prompt_gone() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    perception = Perception(cfg, ocr=FakeOcr())
    frame = _frame()
    draw_prompt(frame, _boxes(cfg.hud)["prompt"])
    first = perception.process(frame, 1)
    assert first.prompt_text == "PRESS E TO MOUNT"
    second = perception.process(_frame(), 2)
    assert second.prompt_text is None
    assert perception.last_prompt == "PRESS E TO MOUNT"


def test_threat_confidence_decays_when_clear() -> None:
    assessor = ThreatAssessor(CombatConfig())
    state = GameState()
    state.threat.wanted_level = 2
    assessor.assess(0.0, state)
    assert state.threat.level == "wanted"
    assert state.confidence.combat == pytest.approx(0.7)
    state.threat.wanted_level = 0
    assessor.assess(1.0, state)
    assert state.threat.level == "none"
    assert state.confidence.combat == pytest.approx(0.0)


def test_guard_exception_in_sleep_returns_false() -> None:
    calls = 0

    def _flaky() -> bool:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("focus check exploded")
        return True

    kb = FakeKeyboard()
    ctrl = InputController(
        InputConfig(), allow_input=_flaky,  # type: ignore[arg-type]
        keyboard=kb,
    )
    assert ctrl.press("w", hold_ms=5) is False
    assert kb.ups == ["w"]


def test_release_all_best_effort_under_lock() -> None:
    import time

    kb = FakeKeyboard()
    ctrl = InputController(
        InputConfig(), allow_input=lambda: True, keyboard=kb  # type: ignore[arg-type]
    )
    assert ctrl.key_down("w") is True
    assert ctrl.held_keys == ["w"]

    locked = threading.Event()

    def _hold() -> None:
        ctrl._lock.acquire()  # noqa: SLF001
        locked.set()
        time.sleep(2.0)
        ctrl._lock.release()  # noqa: SLF001

    holder = threading.Thread(target=_hold)
    holder.start()
    assert locked.wait(timeout=5.0)
    ctrl.release_all()
    holder.join(timeout=5.0)
    assert kb.ups == ["w"]
    assert ctrl.held_keys == []


def test_startup_event_carries_agent_name(tmp_path: Path, monkeypatch) -> None:
    from src.main import main

    cfg_path = _agent_yaml(tmp_path, "")
    monkeypatch.setattr("src.main.GameWindowManager", FakeWindowManager)
    rc = main(["--config", str(cfg_path), "--headless", "--duration", "1"])
    assert rc == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    startups = [row for row in rows if row["kind"] == "startup"]
    assert startups
    assert startups[0]["name"] == "rdr2-ai"


def test_merge_keeps_open_map_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join([
            "vision:",
            "  dialogue:",
            "    respond_to: {greet: interact}",
            "  hud:",
            "    regions: {custom: [0.1, 0.1, 0.2, 0.2]}",
            "survival:",
            "  remedies:",
            "    drink: {keys: [i], needs: {stamina: 0.5}}",
            "mystery_section: {x: 1}",
            "",
        ]),
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    cfg.validate()
    assert cfg.vision.dialogue.respond_to == {"greet": "interact"}
    assert cfg.vision.hud.regions["custom"] == [0.1, 0.1, 0.2, 0.2]
    assert cfg.survival.remedies["drink"].keys == ["i"]
    assert any("mystery_section" in w for w in cfg.warnings)


def test_set_list_index(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join([
            "story:",
            "  missions:",
            "    - name: alpha",
            "",
        ]),
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path), overrides=["story.missions.0.name=beta"])
    assert cfg.story.missions[0]["name"] == "beta"
    with pytest.raises(ConfigError, match="path not found"):
        load_config(str(cfg_path), overrides=["story.missions.5.name=nope"])


def test_fire_button_alias_accepted() -> None:
    cfg = AppConfig()
    cfg.combat.fire_button = "rmb"
    cfg.validate()
    bad = AppConfig()
    bad.combat.fire_button = "banana"
    with pytest.raises(ConfigError, match="fire_button"):
        bad.validate()


def test_control_keys_reject_unknown_key() -> None:
    cfg = AppConfig()
    cfg.control.keys["forward"] = "foo"
    with pytest.raises(ConfigError, match="not a known key"):
        cfg.validate()


def test_story_rejects_bad_wait_completion() -> None:
    cfg = AppConfig()
    cfg.story = StoryConfig(
        enabled=True, missions=[{"name": "x", "wait_completion": "yes"}]
    )
    with pytest.raises(ConfigError, match="wait_completion"):
        cfg.validate()


def test_key_task_role_validated() -> None:
    cfg = AppConfig()
    cfg.mission.enabled = True
    cfg.mission.tasks = [{"key": {"name": "nope"}}]
    with pytest.raises(ConfigError, match="must name a control.keys entry"):
        cfg.validate()

    story = AppConfig()
    story.story = StoryConfig(
        enabled=True,
        missions=[{"name": "x", "tasks": [{"key": {"name": "nope"}}]}],
    )
    with pytest.raises(ConfigError, match="must name a control.keys entry"):
        story.validate()

    ok_cfg = AppConfig()
    ok_cfg.mission.enabled = True
    ok_cfg.mission.tasks = [{"key": {"name": "whistle"}}]
    ok_cfg.validate()


def test_parse_key_task() -> None:
    from src.mission.tasks import parse_tasks

    (task,) = parse_tasks([{"key": {"name": "whistle"}}])
    assert task.kind == "key"
    assert task.key == "whistle"
    assert task.label == "key:whistle"
    with pytest.raises(ValueError, match="requires a 'name'"):
        parse_tasks([{"key": {}}])


def test_runner_taps_key_role() -> None:
    from src.config import MissionConfig

    class TapLocomotion:
        def __init__(self) -> None:
            self.taps: list[str] = []

        def tick(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def tap(self, role: str) -> bool:
            self.taps.append(role)
            return True

    loco = TapLocomotion()
    runner = MissionRunner(
        MissionConfig(enabled=True, tasks=[{"key": {"name": "whistle"}}]),
        ControlConfig(),
        loco,  # type: ignore[arg-type]
    )
    status = StatusTracker(phase=11)
    runner.step(0.0, status)
    assert loco.taps == ["whistle"]
    assert runner.done is True


def test_tap_falls_back_to_raw_key() -> None:
    ctrl, kb = _input()
    loco = LocomotionController(ControlConfig(), ctrl)  # type: ignore[arg-type]
    assert loco.tap("down") is True
    assert kb.ups == ["down"]
    assert kb.held == []


def test_key_task_accepts_raw_keys() -> None:
    cfg = AppConfig()
    cfg.mission.enabled = True
    cfg.mission.tasks = [{"key": {"name": "down"}}]
    cfg.validate()


def test_whistle_key_role_exists() -> None:
    assert ControlConfig().keys["whistle"] == "h"
    assert normalize_key("h").name == "h"
