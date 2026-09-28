"""Story mode: config validation, stage sequencing, completion, failures."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import AppConfig, ConfigError, ControlConfig, StoryConfig
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState
from src.story.runner import StoryRunner
from src.vision.ocr import OcrEngine


def _story_cfg(**kwargs: object) -> StoryConfig:
    base: dict[str, object] = {
        "enabled": True,
        "missions": [{"name": "first", "legs": [[0.0, 10.0]], "tasks": []}],
        "pause_grace_s": 3600.0,
    }
    base.update(kwargs)
    return StoryConfig(**base)  # type: ignore[arg-type]


class FakeLocomotion:
    def __init__(self) -> None:
        self.stops = 0
        self.ticks = 0
        self.taps: list[str] = []
        self.tap_ok = True
        self.turns: list[float] = []
        self.turn_ok = True
        self.moves: list[tuple[str, float, bool]] = []
        self.move_ok = True
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def stop(self) -> None:
        self.stops += 1

    def tick(self) -> None:
        self.ticks += 1

    def tap(self, role: str) -> bool:
        self.taps.append(role)
        return self.tap_ok

    def turn(self, degrees: float) -> bool:
        self.turns.append(degrees)
        return self.turn_ok

    def move(self, direction: str, duration_s: float, *,
             sprint: bool = False) -> bool:
        self.moves.append((direction, duration_s, sprint))
        return self.move_ok


class FakeRoute:
    instances: list[FakeRoute] = []
    arrive_after = 1
    fail_after = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.steps = 0
        self.arrived = False
        self.failed = False
        self.current = "leg 1/1"
        self.legs: list[tuple[float, float]] = []
        self.heading = 0.0
        self.nudges: list[float] = []
        self.control_arg = args[0] if args else None
        FakeRoute.instances.append(self)

    def nudge_heading(self, delta_deg: float) -> None:
        self.nudges.append(delta_deg)
        self.heading += delta_deg

    @property
    def heading_deg(self) -> float:
        return self.heading

    def set_legs(self, legs: list[tuple[float, float]]) -> None:
        self.legs = list(legs)

    def step(self, now: float, status: StatusTracker, frame: object = None,
             game_state: object = None) -> None:
        self.steps += 1
        status.update(goal="ROUTE 1/1", action="walk")
        if FakeRoute.fail_after and self.steps >= FakeRoute.fail_after:
            self.failed = True
            return
        if self.steps >= FakeRoute.arrive_after:
            self.arrived = True


class FakeMission:
    instances: list[FakeMission] = []
    done_after = 1
    fail_after = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.steps = 0
        self.done = False
        self.failed = False
        self.current = "task 1/1"
        FakeMission.instances.append(self)

    def step(self, now: float, status: StatusTracker, frame: object = None,
             game_state: object | None = None) -> None:
        self.steps += 1
        status.update(goal="MISSION 1/1", action="log")
        if FakeMission.fail_after and self.steps >= FakeMission.fail_after:
            self.failed = True
            return
        if self.steps >= FakeMission.done_after:
            self.done = True


class KeywordOcr(OcrEngine):
    name = "kw"
    available = True

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def read(self, image: np.ndarray) -> tuple[str, float]:
        self.calls += 1
        return self.text, 0.9


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRoute.instances = []
    FakeMission.instances = []
    FakeRoute.arrive_after = 1
    FakeRoute.fail_after = 0
    FakeMission.done_after = 1
    FakeMission.fail_after = 0
    monkeypatch.setattr("src.story.runner.RoutePlanner", FakeRoute)
    monkeypatch.setattr("src.story.runner.MissionRunner", FakeMission)


def _runner(cfg: StoryConfig, ocr: OcrEngine | None = None) -> StoryRunner:
    return StoryRunner(
        cfg, ControlConfig(), FakeLocomotion(),  # type: ignore[arg-type]
        ocr=ocr,
    )


def _status() -> StatusTracker:
    return StatusTracker(phase=11)


def test_defaults_are_disabled_and_inert() -> None:
    assert StoryConfig().enabled is False
    assert StoryConfig().missions == []
    assert AppConfig().story.enabled is False


def test_valid_story_config_passes_validation() -> None:
    cfg = AppConfig()
    cfg.story = _story_cfg()
    cfg.validate()


def test_story_requires_missions_when_enabled() -> None:
    cfg = AppConfig()
    cfg.story = StoryConfig(enabled=True)
    with pytest.raises(ConfigError, match="story.enabled requires"):
        cfg.validate()


def test_story_rejects_bad_leg_shape() -> None:
    cfg = AppConfig()
    cfg.story = _story_cfg(missions=[{"name": "x", "legs": [[0.0]]}])
    with pytest.raises(ConfigError, match=r"legs\[1\]"):
        cfg.validate()


def test_story_rejects_unknown_task_kind() -> None:
    cfg = AppConfig()
    cfg.story = _story_cfg(
        missions=[{"name": "x", "tasks": [{"kind": "bogus"}]}]
    )
    with pytest.raises(ConfigError, match="story.missions\\[1\\]"):
        cfg.validate()


def test_story_conflicts_with_other_primary_drivers() -> None:
    mission_cfg = AppConfig()
    mission_cfg.story = _story_cfg()
    mission_cfg.mission.enabled = True
    mission_cfg.mission.tasks = [{"kind": "log", "text": "hi"}]
    with pytest.raises(ConfigError, match="story.enabled conflicts with mission"):
        mission_cfg.validate()

    control_cfg = AppConfig()
    control_cfg.story = _story_cfg()
    control_cfg.control.enabled = True
    with pytest.raises(ConfigError, match="story.enabled conflicts with control"):
        control_cfg.validate()

    rl_cfg = AppConfig()
    rl_cfg.story = _story_cfg()
    rl_cfg.rl.enabled = True
    with pytest.raises(ConfigError, match="story.enabled conflicts with rl"):
        rl_cfg.validate()


def test_story_rejects_bad_completion_region() -> None:
    cfg = AppConfig()
    cfg.story = _story_cfg(completion_region=[0.1, 0.1, 0.0, 0.2])
    with pytest.raises(ConfigError, match="story.completion_region"):
        cfg.validate()


def test_travel_arrives_then_keyword_completes() -> None:
    runner = _runner(_story_cfg())
    status = _status()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = GameState()

    runner.step(0.0, status, frame, state)
    assert runner.stage == "complete"
    assert status.snapshot().goal.startswith("STORY 1/1")

    runner.step(0.1, status, frame, state)
    assert runner.stage == "complete"
    state.mission.prompt_text = "MISSION PASSED"
    runner.step(0.2, status, frame, state)
    assert runner.done is True
    runner.step(0.3, status, frame, state)
    assert status.snapshot().goal == "STORY DONE"


def test_travel_skipped_without_legs() -> None:
    runner = _runner(_story_cfg(missions=[{"name": "x", "tasks": []}]))
    runner.step(0.0, _status())
    assert runner.stage == "complete"
    assert FakeRoute.instances == []


def test_objective_stage_runs_tasks() -> None:
    FakeMission.done_after = 2
    runner = _runner(
        _story_cfg(missions=[{"name": "x", "legs": [], "tasks": [{"kind": "log", "text": "go"}]}])
    )
    status = _status()
    runner.step(0.0, status)
    assert runner.stage == "objective"
    assert len(FakeMission.instances) == 1
    runner.step(0.1, status)
    assert runner.stage == "complete"


def test_objective_without_completion_watch_advances() -> None:
    runner = _runner(_story_cfg(missions=[
        {"name": "x", "legs": [], "tasks": [{"kind": "log", "text": "go"}],
         "wait_completion": False},
        {"name": "y", "legs": [], "tasks": []},
    ]))
    status = _status()
    runner.step(0.0, status)
    runner.step(0.1, status)
    assert runner.index == 1
    assert runner.stage == "complete"


def test_route_failure_skips_forward() -> None:
    FakeRoute.fail_after = 1
    runner = _runner(_story_cfg(missions=[
        {"name": "x", "legs": [[0.0, 10.0]]},
        {"name": "y", "legs": [[0.0, 5.0]]},
    ]))
    status = _status()
    runner.step(0.0, status)
    assert runner.index == 1
    assert runner.stage == "travel"


def test_travel_timeout_skips_forward() -> None:
    FakeRoute.arrive_after = 10_000
    cfg = _story_cfg(travel_timeout_s=10.0, missions=[
        {"name": "x", "legs": [[0.0, 10.0]]},
        {"name": "y", "legs": []},
    ])
    runner = _runner(cfg)
    status = _status()
    runner.step(0.0, status)
    runner.step(5.0, status)
    assert runner.index == 0
    runner.step(11.0, status)
    assert runner.index == 1


def test_completion_timeout_advances_without_keyword() -> None:
    cfg = _story_cfg(completion_timeout_s=10.0, missions=[
        {"name": "x", "legs": [[0.0, 10.0]]},
        {"name": "y", "legs": []},
    ])
    runner = _runner(cfg)
    status = _status()
    runner.step(0.0, status)   # travel -> arrived
    runner.step(0.1, status)   # complete watch starts
    runner.step(11.0, status)  # past timeout
    assert runner.index == 1


def test_banner_ocr_keyword_completes() -> None:
    ocr = KeywordOcr("Mission Passed")
    cfg = _story_cfg(completion_every_n_frames=1)
    runner = _runner(cfg, ocr=ocr)
    status = _status()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    runner.step(0.0, status, frame)  # travel arrives
    runner.step(0.1, status, frame)  # banner OCR hits
    assert runner.done is True
    assert ocr.calls >= 1


def test_loop_wraps_after_last_mission() -> None:
    cfg = _story_cfg(loop=True, missions=[
        {"name": "a", "legs": [[0.0, 1.0]]},
        {"name": "b", "legs": [[0.0, 1.0]]},
    ])
    runner = _runner(cfg)
    status = _status()
    state = GameState()
    state.mission.prompt_text = "mission complete"
    for i in range(12):
        runner.step(float(i), status, None, state)
    assert runner.done is False
    assert runner.index == 0


def test_pause_gap_does_not_burn_stage_timeout() -> None:
    FakeRoute.arrive_after = 10_000
    cfg = _story_cfg(travel_timeout_s=10.0, pause_grace_s=15.0)
    runner = _runner(cfg)
    status = _status()
    runner.step(0.0, status)
    runner.step(20.0, status)  # 20s gap > grace -> timer rebased, no timeout
    assert runner.index == 0
    assert runner.stage == "travel"


def test_done_runner_parks_status() -> None:
    runner = _runner(_story_cfg())
    status = _status()
    state = GameState()
    runner.step(0.0, status, None, state)
    state.mission.prompt_text = "mission passed"
    runner.step(0.1, status, None, state)
    assert runner.done
    runner.step(0.2, status, None, state)
    snap = status.snapshot()
    assert snap.goal == "STORY DONE"
    assert snap.action == "none"


def test_story_rejects_bad_kind() -> None:
    cfg = AppConfig()
    cfg.story = _story_cfg(missions=[{"name": "x", "kind": "epic"}])
    with pytest.raises(ConfigError, match=r"kind must be story\|stranger"):
        cfg.validate()


def test_story_rejects_non_bool_mount_flag() -> None:
    cfg = AppConfig()
    cfg.story = _story_cfg(missions=[{"name": "x", "mount": "yes"}])
    with pytest.raises(ConfigError, match=r"\.mount must be a boolean"):
        cfg.validate()


def test_bad_profile_keeps_pending_begin() -> None:
    runner = _runner(_story_cfg(missions=[{"name": "x", "legs": [["bogus"]]}]))
    with pytest.raises((ValueError, TypeError)):
        runner.step(0.0, _status())
    assert runner._pending_begin is True  # noqa: SLF001 - white-box retry check


def test_mount_whistles_boards_and_rides_mounted() -> None:
    cfg = _story_cfg(
        mount_wait_s=0.1,
        missions=[{"name": "ride", "mount": True, "legs": [[0.0, 10.0]]}],
    )
    loco = FakeLocomotion()
    runner = StoryRunner(cfg, ControlConfig(), loco)  # type: ignore[arg-type]
    status = _status()
    state = GameState()

    runner.step(0.0, status, None, state)
    assert runner.stage == "mount"
    assert loco.taps == ["whistle"]

    state.dialogue.encounter = "mount"
    runner.step(0.1, status, None, state)
    assert loco.taps == ["whistle", "interact"]
    assert runner.stage == "mount"

    runner.step(0.3, status, None, state)
    assert runner.stage == "travel"
    runner.step(0.4, status, None, state)
    assert FakeRoute.instances, "route was not built after mounting"
    mounted_nav = FakeRoute.instances[-1].control_arg.nav
    assert mounted_nav.walk_speed_mps == pytest.approx(cfg.mounted_speed_mps)


def test_mount_timeout_continues_on_foot() -> None:
    cfg = _story_cfg(
        mount_timeout_s=0.2,
        missions=[{"name": "walk", "mount": True, "legs": [[0.0, 10.0]]}],
    )
    loco = FakeLocomotion()
    runner = StoryRunner(cfg, ControlConfig(), loco)  # type: ignore[arg-type]
    status = _status()
    state = GameState()
    runner.step(0.0, status, None, state)
    assert loco.taps == ["whistle"]
    runner.step(0.5, status, None, state)
    assert runner.stage == "travel"
    runner.step(0.6, status, None, state)
    plain_nav = FakeRoute.instances[-1].control_arg.nav
    assert plain_nav.walk_speed_mps == pytest.approx(1.6)


def test_mount_whistle_refused_continues() -> None:
    cfg = _story_cfg(missions=[{"name": "walk", "mount": True}])
    loco = FakeLocomotion()
    loco.tap_ok = False
    runner = StoryRunner(cfg, ControlConfig(), loco)  # type: ignore[arg-type]
    runner.step(0.0, _status())
    assert runner.stage == "complete"
    assert loco.taps == ["whistle"]


def test_seek_nudges_toward_persistent_marker() -> None:
    cfg = _story_cfg(
        marker_persist_frames=2,
        missions=[{"name": "seek", "legs": [[0.0, 50.0]], "seek_marker": True}],
    )
    FakeRoute.arrive_after = 10_000
    loco = FakeLocomotion()
    runner = StoryRunner(cfg, ControlConfig(), loco)  # type: ignore[arg-type]
    status = _status()
    state = GameState()
    state.mission.objective_location_estimate = [0.3, -0.3]
    runner.step(0.0, status, None, state)
    assert FakeRoute.instances[-1].nudges == []
    runner.step(0.1, status, None, state)
    nudges = FakeRoute.instances[-1].nudges
    assert nudges == [pytest.approx(30.0)]
    assert loco.turns == [pytest.approx(30.0)]


def test_seek_ignores_centered_marker() -> None:
    cfg = _story_cfg(
        marker_persist_frames=1,
        missions=[{"name": "seek", "legs": [[0.0, 50.0]], "seek_marker": True}],
    )
    FakeRoute.arrive_after = 10_000
    loco = FakeLocomotion()
    runner = StoryRunner(cfg, ControlConfig(), loco)  # type: ignore[arg-type]
    state = GameState()
    state.mission.objective_location_estimate = [0.0, -0.4]
    runner.step(0.0, _status(), None, state)
    assert FakeRoute.instances[-1].nudges == []
    assert loco.turns == []


def test_seek_resets_when_marker_lost() -> None:
    cfg = _story_cfg(
        marker_persist_frames=1,
        missions=[{"name": "seek", "legs": [[0.0, 50.0]], "seek_marker": True}],
    )
    FakeRoute.arrive_after = 10_000
    loco = FakeLocomotion()
    runner = StoryRunner(cfg, ControlConfig(), loco)  # type: ignore[arg-type]
    status = _status()
    state = GameState()
    state.mission.objective_location_estimate = [0.4, -0.1]
    runner.step(0.0, status, None, state)
    assert len(FakeRoute.instances[-1].nudges) == 1
    state.mission.objective_location_estimate = None
    runner.step(0.1, status, None, state)
    assert len(FakeRoute.instances[-1].nudges) == 1
    assert runner._marker_frames == 0  # noqa: SLF001 - white-box reset check


def _stranger_cfg(**kwargs: object) -> StoryConfig:
    base: dict[str, object] = {
        "enabled": True,
        "pause_grace_s": 3600.0,
        "missions": [
            {"name": "story-a", "legs": [[0.0, 5.0]]},
            {"name": "stranger-b", "kind": "stranger", "legs": [[0.0, 5.0]]},
            {"name": "story-c", "legs": [[0.0, 5.0]]},
        ],
    }
    base.update(kwargs)
    return StoryConfig(**base)  # type: ignore[arg-type]


def test_opportunistic_detour_and_resume() -> None:
    runner = _runner(_stranger_cfg())
    status = _status()
    state = GameState()
    state.mission.objective_location_estimate = [0.2, -0.2]

    runner.step(0.0, status, None, state)
    assert runner.index == 1
    runner.step(0.1, status, None, state)
    assert runner.stage == "complete"

    state.mission.prompt_text = "mission passed"
    state.mission.objective_location_estimate = None
    runner.step(0.2, status, None, state)
    assert runner.index == 0
    runner.step(0.3, status, None, state)
    assert runner.index == 0
    assert runner.stage == "complete"


def test_no_detour_when_opportunistic_off() -> None:
    runner = _runner(_stranger_cfg(opportunistic=False))
    state = GameState()
    state.mission.objective_location_estimate = [0.2, -0.2]
    runner.step(0.0, _status(), None, state)
    assert runner.index == 0


def test_no_detour_without_marker() -> None:
    runner = _runner(_stranger_cfg())
    runner.step(0.0, _status(), None, GameState())
    assert runner.index == 0


def _mounted_runner(cfg: StoryConfig, loco: FakeLocomotion,
                    status: StatusTracker, state: GameState) -> StoryRunner:
    runner = StoryRunner(cfg, ControlConfig(), loco)  # type: ignore[arg-type]
    runner.step(0.0, status, None, state)
    state.dialogue.encounter = "mount"
    runner.step(1.0, status, None, state)
    runner.step(6.0, status, None, state)
    assert runner.stage == "travel"
    return runner


def test_dismount_after_mounted_travel() -> None:
    cfg = _story_cfg(missions=[{
        "name": "ride", "mount": True, "legs": [[0.0, 10.0]],
        "dismount": True, "tasks": [{"kind": "log", "text": "hi"}],
    }])
    loco = FakeLocomotion()
    status = _status()
    state = GameState()
    runner = _mounted_runner(cfg, loco, status, state)
    runner.step(7.0, status, None, state)
    assert runner.stage == "dismount"
    assert loco.taps == ["whistle", "interact"]
    runner.step(8.0, status, None, state)
    assert loco.taps == ["whistle", "interact", "interact"]
    runner.step(12.0, status, None, state)
    assert runner._mounted is None  # noqa: SLF001
    assert runner.stage == "objective"
    runner.step(13.0, status, None, state)
    assert len(FakeMission.instances) == 1


def test_dismount_skipped_when_on_foot() -> None:
    cfg = _story_cfg(missions=[{
        "name": "walk", "legs": [[0.0, 10.0]], "dismount": True,
    }])
    runner = _runner(cfg)
    status = _status()
    runner.step(0.0, status)
    assert runner.stage == "complete"
    assert FakeRoute.instances[-1].control_arg.nav.walk_speed_mps == pytest.approx(1.6)


def test_dismount_bearing_turns_first() -> None:
    cfg = _story_cfg(missions=[{
        "name": "ride", "mount": True, "legs": [[0.0, 10.0]],
        "dismount": True, "dismount_bearing": 90.0,
    }])
    loco = FakeLocomotion()
    status = _status()
    state = GameState()
    runner = _mounted_runner(cfg, loco, status, state)
    runner.step(7.0, status, None, state)
    assert runner.stage == "dismount"
    runner.step(8.0, status, None, state)
    runner.step(9.0, status, None, state)
    runner.step(10.0, status, None, state)
    assert loco.turns == [pytest.approx(30.0)] * 3
    assert loco.taps == ["whistle", "interact"]
    runner.step(11.0, status, None, state)
    assert loco.taps == ["whistle", "interact", "interact"]


def test_camp_forces_dismount() -> None:
    cfg = _story_cfg(missions=[{
        "name": "camp", "kind": "camp", "mount": True, "legs": [[0.0, 10.0]],
    }])
    loco = FakeLocomotion()
    status = _status()
    state = GameState()
    runner = _mounted_runner(cfg, loco, status, state)
    runner.step(7.0, status, None, state)
    assert runner.stage == "dismount"
    runner.step(8.0, status, None, state)
    runner.step(12.0, status, None, state)
    assert runner.stage == "complete"
    assert runner._mounted is None  # noqa: SLF001


def test_roam_wanders_and_turns() -> None:
    cfg = _story_cfg(
        roam_leg_m=3.0,
        missions=[{"name": "roam", "roam": True}],
    )
    loco = FakeLocomotion()
    runner = StoryRunner(cfg, ControlConfig(), loco)  # type: ignore[arg-type]
    status = _status()
    runner.step(0.0, status)
    assert runner.stage == "roam"
    runner.step(1.0, status)
    runner.step(2.0, status)
    assert loco.moves, "roam never walked"
    runner.step(3.0, status)
    runner.step(4.0, status)
    assert loco.turns == [pytest.approx(30.0)] * 2
    runner.step(5.0, status)
    assert runner.stage == "roam"
    assert runner.done is False


def test_roam_profile_validated() -> None:
    cfg = AppConfig()
    cfg.story = _story_cfg(missions=[{"name": "r", "roam": "yes"}])
    with pytest.raises(ConfigError, match=r"\.roam must be a boolean"):
        cfg.validate()
