"""Fast vision pass and game state data model tests."""

from __future__ import annotations

import numpy as np

from src.config import FastPassConfig
from src.state.agent_status import StatusTracker
from src.state.game_state import Confidence, GameState
from src.vision.fast_pass import FastPass


def solid(color: int, size: tuple[int, int] = (90, 160)) -> np.ndarray:
    return np.full((size[0], size[1], 3), color, dtype=np.uint8)


def test_identical_frames_have_zero_motion() -> None:
    fast = FastPass(FastPassConfig())
    first = fast.process(solid(50))
    second = fast.process(solid(50))
    assert first.motion == 0.0
    assert second.motion == 0.0
    assert first.latency_ms >= 0.0


def test_changed_frames_have_motion() -> None:
    fast = FastPass(FastPassConfig())
    fast.process(solid(0))
    result = fast.process(solid(255))
    assert result.motion > 10.0


def test_brightness_reflects_content() -> None:
    fast = FastPass(FastPassConfig())
    dark = fast.process(solid(5))
    bright = fast.process(solid(250))
    assert dark.brightness < 30
    assert bright.brightness > 200


def test_frame_skipping_reuses_result() -> None:
    fast = FastPass(FastPassConfig(every_n_frames=3))
    a = fast.process(solid(10))
    b = fast.process(solid(200))
    c = fast.process(solid(200))
    assert a.skipped is False
    assert b.skipped is True
    assert c.skipped is False


def test_disabled_fast_pass_returns_empty() -> None:
    fast = FastPass(FastPassConfig(enabled=False))
    result = fast.process(solid(128))
    assert result.skipped is True
    assert result.brightness == 0.0


def test_reset_clears_motion_history() -> None:
    fast = FastPass(FastPassConfig())
    fast.process(solid(0))
    fast.reset()
    result = fast.process(solid(255))
    assert result.motion == 0.0


def test_game_state_defaults_are_unknown_not_guessed() -> None:
    state = GameState()
    assert state.player.health is None
    assert state.horse.detected is False
    assert state.threat.level == "none"
    assert state.confidence.overall == 0.0
    assert state.nearby_entities == []
    assert state.mission.active is False


def test_game_state_to_dict() -> None:
    data = GameState().to_dict()
    assert isinstance(data, dict)
    assert data["confidence"]["overall"] == 0.0
    assert data["player"]["mounted"] is None
    assert "environment" in data


def test_game_state_not_stale_initially() -> None:
    assert GameState().stale is False


def test_confidence_dict_rounding() -> None:
    conf = Confidence(overall=0.123456, navigation=0.8765)
    assert conf.as_dict()["overall"] == 0.123
    assert conf.as_dict()["navigation"] == 0.876


def test_status_tracker_phase() -> None:
    tracker = StatusTracker(phase=2)
    assert tracker.snapshot().phase == 2
