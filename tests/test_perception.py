"""Perception pipeline: HUD -> GameState, OCR throttling, telemetry summary."""

from __future__ import annotations

import numpy as np

from src.config import VisionConfig
from src.state.game_state import GameState
from src.vision.ocr import NullOcr, OcrEngine
from src.vision.perception import Perception, hud_summary
from tests.test_hud import _boxes, _frame, draw_gauge, draw_prompt


class FakeOcr(OcrEngine):
    name = "fake"
    available = True

    def __init__(self, text: str = "PRESS E TO MOUNT") -> None:
        self.text = text
        self.calls = 0

    def read(self, image: np.ndarray) -> tuple[str, float]:
        self.calls += 1
        return self.text, 0.9


def test_process_returns_result_for_plain_frame() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=NullOcr())
    pres = perception.process(_frame(), frame_id=42)
    assert pres.frame_id == 42
    assert pres.latency_ms > 0.0
    assert pres.ocr_engine == "off"
    assert pres.prompt_text is None
    assert not pres.prompt_changed


def test_apply_to_state_leaves_undetected_fields_none() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=NullOcr())
    state = GameState()
    pres = perception.process(_frame(), frame_id=1)
    perception.apply_to_state(state, pres)
    assert state.player.health is None
    assert state.player.stamina is None
    assert state.confidence.player == 0.0
    assert state.frame_id == 1
    assert state.source == "hud+world+dialogue"


def test_gauges_flow_into_state_with_confidence() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=NullOcr())
    frame = _frame()
    boxes = _boxes(cfg.hud)
    draw_gauge(frame, boxes["health"], 0.75)
    state = GameState()
    perception.apply_to_state(state, perception.process(frame, frame_id=7))
    assert state.player.health is not None
    assert abs(state.player.health - 0.75) <= 0.12
    assert state.confidence.player > 0.0
    assert state.confidence.overall > 0.0


def test_prompt_change_detected_once() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    fake = FakeOcr()
    perception = Perception(cfg, ocr=fake)
    frame = _frame()
    draw_prompt(frame, _boxes(cfg.hud)["prompt"])
    first = perception.process(frame, frame_id=1)
    second = perception.process(frame, frame_id=2)
    assert first.prompt_changed is True
    assert first.prompt_text == "PRESS E TO MOUNT"
    assert second.prompt_changed is False
    assert perception.last_prompt == "PRESS E TO MOUNT"
    assert fake.calls == 2


def test_ocr_is_throttled() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 3
    fake = FakeOcr()
    perception = Perception(cfg, ocr=fake)
    frame = _frame()
    draw_prompt(frame, _boxes(cfg.hud)["prompt"])
    for i in range(6):
        perception.process(frame, frame_id=i)
    assert fake.calls == 2


def test_ocr_skipped_without_visible_prompt() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    fake = FakeOcr()
    perception = Perception(cfg, ocr=fake)
    for i in range(3):
        perception.process(_frame(), frame_id=i)
    assert fake.calls == 0


def test_ocr_below_confidence_rejected() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    cfg.ocr.min_confidence = 95.0

    class LowConfOcr(FakeOcr):
        def read(self, image: np.ndarray) -> tuple[str, float]:
            self.calls += 1
            return "SOMETHING", 0.5

    perception = Perception(cfg, ocr=LowConfOcr())
    frame = _frame()
    draw_prompt(frame, _boxes(cfg.hud)["prompt"])
    pres = perception.process(frame, frame_id=1)
    assert pres.prompt_changed is False
    assert pres.prompt_text is None


def test_disabled_hud_short_circuits() -> None:
    cfg = VisionConfig()
    cfg.hud.enabled = False
    perception = Perception(cfg, ocr=FakeOcr())
    pres = perception.process(_frame(), frame_id=3)
    assert pres.hud.skipped
    assert pres.prompt_text is None


def test_overlay_boxes_respect_detection() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=NullOcr())
    frame = _frame()
    draw_gauge(frame, _boxes(cfg.hud)["health"], 0.75)
    boxes, labels = perception.overlay_boxes(perception.process(frame, 1))
    assert len(boxes) == len(labels) == 1


def test_hud_summary_shape() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=NullOcr())
    summary = hud_summary(perception.process(_frame(), 1))
    assert set(summary) >= {
        "health", "stamina", "dead_eye", "minimap", "prompt_visible",
        "prompt", "wanted", "ocr", "latency_ms",
    }


def test_wanted_stars_flow_into_threat() -> None:
    import cv2

    cfg = VisionConfig()
    perception = Perception(cfg, ocr=NullOcr())
    frame = _frame(world=20)
    x, y, w, h = _boxes(cfg.hud)["wanted"]
    crop = frame[y:y + h, x:x + w]
    for i in range(2):
        cv2.circle(crop, (30 + i * 40, h // 2), 7, (255, 255, 255), -1)
    state = GameState()
    perception.apply_to_state(state, perception.process(frame, 5))
    assert state.threat.wanted_level == 2
    assert state.threat.level == "wanted"


def test_objective_text_flows_into_state() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    perception = Perception(cfg, ocr=FakeOcr(text="Go to the sheriff"))
    frame = _frame()
    draw_prompt(frame, _boxes(cfg.hud)["objective"])
    state = GameState()
    perception.apply_to_state(state, perception.process(frame, 1))
    assert state.mission.objective_text == "Go to the sheriff"
    assert state.mission.active is True


def test_objective_clears_when_region_empty() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    perception = Perception(cfg, ocr=FakeOcr(text="Go to the sheriff"))
    frame = _frame()
    draw_prompt(frame, _boxes(cfg.hud)["objective"])
    state = GameState()
    perception.apply_to_state(state, perception.process(frame, 1))
    assert state.mission.active is True
    perception.apply_to_state(state, perception.process(_frame(), 2))
    assert state.mission.objective_text is None
    assert state.mission.active is False


def test_objective_ocr_not_run_when_region_absent() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    fake = FakeOcr()
    perception = Perception(cfg, ocr=fake)
    for i in range(3):
        perception.process(_frame(), frame_id=i)
    assert fake.calls == 0
