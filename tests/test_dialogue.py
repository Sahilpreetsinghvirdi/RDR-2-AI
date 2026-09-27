"""Phase 6: dialogue band detection, encounter keywords, prompt responder."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import AppConfig, ConfigError, DialogueConfig, VisionConfig
from src.control.locomotion import LocomotionController
from src.mission.respond import PromptResponder
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState
from src.vision.dialogue import DialogueReader, match_keyword
from src.vision.ocr import NullOcr, OcrEngine
from src.vision.perception import Perception, dialogue_summary
from tests.test_hud import _boxes, _frame, draw_prompt

FRAME_W, FRAME_H = 1920, 1080


class FakeInput:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.fail_keys: set[str] = set()

    def key_down(self, key: str) -> bool:
        self.events.append(("down", key))
        return True

    def key_up(self, key: str) -> bool:
        self.events.append(("up", key))
        return True

    def mouse_move(self, dx: int, dy: int = 0) -> bool:
        self.events.append(("mouse", dx))
        return True

    def press(self, key: str, hold_ms: int | None = None) -> bool:
        if key in self.fail_keys:
            return False
        self.events.append(("press", key))
        return True


class FakeOcr(OcrEngine):
    name = "fake"
    available = True

    def __init__(self, text: str = "hello there") -> None:
        self.text = text
        self.calls = 0

    def read(self, image: np.ndarray) -> tuple[str, float]:
        self.calls += 1
        return self.text, 0.9


def _dialogue_frame(bright: bool = True) -> np.ndarray:
    frame = _frame()
    if bright:
        x, y, w, h = [0.20, 0.82, 0.60, 0.07]
        px, py = int(x * FRAME_W), int(y * FRAME_H)
        pw, ph = int(w * FRAME_W), int(h * FRAME_H)
        frame[py:py + ph, px:px + pw] = (245, 245, 245)
    return frame


def make_responder(
    respond_to: dict[str, str] | None = None, cooldown: float = 2.0
) -> tuple[PromptResponder, FakeInput, StatusTracker]:
    cfg = DialogueConfig(
        respond_to=respond_to if respond_to is not None else {"greet": "interact"},
        respond_cooldown_s=cooldown,
    )
    fake = FakeInput()
    loco = LocomotionController(AppConfig().control, fake)
    return PromptResponder(cfg, loco), fake, StatusTracker(phase=6)


class TestMatchKeyword:
    def test_finds_first_keyword_case_insensitive(self) -> None:
        assert match_keyword("Press E to Greet", ["rob", "greet"]) == "greet"

    def test_returns_none_when_no_match(self) -> None:
        assert match_keyword("nothing here", ["greet"]) is None

    def test_none_and_empty_text(self) -> None:
        assert match_keyword(None, ["greet"]) is None
        assert match_keyword("", ["greet"]) is None

    def test_skips_empty_keywords(self) -> None:
        assert match_keyword("greet me", ["", "greet"]) == "greet"


class TestDialogueReader:
    def test_defaults_are_not_present_and_skipped(self) -> None:
        reading = DialogueReader(DialogueConfig()).process(_dialogue_frame(False))
        assert reading.present is False
        assert reading.text is None
        assert reading.skipped is False

    def test_disabled_reader_skips(self) -> None:
        cfg = DialogueConfig(enabled=False)
        reading = DialogueReader(cfg).process(_dialogue_frame(True))
        assert reading.skipped is True
        assert reading.present is False

    def test_bright_band_marks_present(self) -> None:
        reading = DialogueReader(DialogueConfig()).process(_dialogue_frame(True))
        assert reading.present is True
        assert reading.confidence > 0.0

    def test_tiny_frame_returns_skipped(self) -> None:
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        reading = DialogueReader(DialogueConfig()).process(tiny)
        assert reading.skipped is True

    def test_as_dict_shape(self) -> None:
        reading = DialogueReader(DialogueConfig()).process(_dialogue_frame(True))
        data = reading.as_dict()
        assert set(data) >= {"present", "text", "confidence", "skipped", "latency_ms"}


class TestDialogueConfigValidation:
    def test_default_config_valid(self) -> None:
        AppConfig().validate()

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("region", [0.1, 0.1, 0.0, 0.5], "region"),
            ("region", [0.1, 0.1, 0.5], "region"),
            ("region", [0.1, 0.1, 0.5, 0.5, 0.1], "region"),
            ("region", [-0.2, 0.1, 0.5, 0.5], "region"),
            ("min_text_frac", 1.5, "min_text_frac"),
            ("bright_luma", 300, "bright_luma"),
            ("every_n_frames", 0, "every_n_frames"),
            ("encounter_keywords", ["greet", 5], "encounter_keywords"),
            ("respond_to", "not-a-map", "respond_to"),
            ("respond_to", {"greet": "not_a_key"}, "control.keys"),
            ("respond_cooldown_s", 0.0, "respond_cooldown_s"),
        ],
    )
    def test_invalid_values_rejected(
        self, field: str, value: object, match: str
    ) -> None:
        cfg = AppConfig()
        setattr(cfg.vision.dialogue, field, value)
        with pytest.raises(ConfigError, match=match):
            cfg.validate()


class TestPerceptionDialogue:
    def test_bright_band_flows_into_state(self) -> None:
        cfg = VisionConfig()
        perception = Perception(cfg, ocr=NullOcr())
        state = GameState()
        assert state.dialogue.active is None
        perception.apply_to_state(state, perception.process(_dialogue_frame(True), 1))
        assert state.dialogue.active is True
        assert state.source == "hud+world+dialogue"

    def test_dark_band_marks_inactive(self) -> None:
        cfg = VisionConfig()
        perception = Perception(cfg, ocr=NullOcr())
        state = GameState()
        perception.apply_to_state(state, perception.process(_dialogue_frame(False), 1))
        assert state.dialogue.active is False

    def test_disabled_dialogue_leaves_state_none(self) -> None:
        cfg = VisionConfig()
        cfg.dialogue.enabled = False
        perception = Perception(cfg, ocr=NullOcr())
        state = GameState()
        perception.apply_to_state(state, perception.process(_dialogue_frame(True), 1))
        assert state.dialogue.active is None
        assert perception.process(_dialogue_frame(True), 2).dialogue.skipped

    def test_dialogue_ocr_throttled(self) -> None:
        cfg = VisionConfig()
        cfg.dialogue.every_n_frames = 3
        cfg.hud.enabled = False
        fake = FakeOcr()
        perception = Perception(cfg, ocr=fake)
        for i in range(6):
            perception.process(_dialogue_frame(True), i)
        assert fake.calls == 2

    def test_dialogue_ocr_stops_when_band_gone(self) -> None:
        cfg = VisionConfig()
        cfg.dialogue.every_n_frames = 1
        cfg.hud.enabled = False
        fake = FakeOcr()
        perception = Perception(cfg, ocr=fake)
        perception.process(_dialogue_frame(True), 1)
        calls_after_present = fake.calls
        perception.process(_dialogue_frame(False), 2)
        assert fake.calls == calls_after_present

    def test_encounter_from_dialogue_text(self) -> None:
        cfg = VisionConfig()
        cfg.dialogue.every_n_frames = 1
        cfg.hud.enabled = False
        perception = Perception(cfg, ocr=FakeOcr("Nice day to Greet someone"))
        pres = perception.process(_dialogue_frame(True), 1)
        assert pres.dialogue.present
        assert pres.dialogue.text == "Nice day to Greet someone"
        assert pres.encounter == "greet"

    def test_encounter_from_visible_prompt(self) -> None:
        cfg = VisionConfig()
        cfg.ocr.every_n_frames = 1
        perception = Perception(cfg, ocr=FakeOcr("PRESS E TO GREET"))
        frame = _frame()
        draw_prompt(frame, _boxes(cfg.hud)["prompt"])
        pres = perception.process(frame, 1)
        assert pres.prompt_text == "PRESS E TO GREET"
        assert pres.encounter == "greet"

    def test_no_encounter_without_keyword(self) -> None:
        cfg = VisionConfig()
        cfg.dialogue.every_n_frames = 1
        cfg.hud.enabled = False
        perception = Perception(cfg, ocr=FakeOcr("just ordinary words"))
        pres = perception.process(_dialogue_frame(True), 1)
        assert pres.encounter is None

    def test_stale_prompt_does_not_match_after_it_disappears(self) -> None:
        cfg = VisionConfig()
        cfg.ocr.every_n_frames = 1
        cfg.dialogue.enabled = False
        perception = Perception(cfg, ocr=FakeOcr("PRESS E TO GREET"))
        frame = _frame()
        draw_prompt(frame, _boxes(cfg.hud)["prompt"])
        assert perception.process(frame, 1).encounter == "greet"
        assert perception.process(_frame(), 2).encounter is None

    def test_dialogue_summary_shape(self) -> None:
        cfg = VisionConfig()
        perception = Perception(cfg, ocr=NullOcr())
        summary = dialogue_summary(perception.process(_dialogue_frame(True), 1))
        assert set(summary) >= {
            "enabled", "present", "text", "encounter", "confidence", "latency_ms",
        }

    def test_dialogue_summary_disabled(self) -> None:
        cfg = VisionConfig()
        cfg.dialogue.enabled = False
        perception = Perception(cfg, ocr=NullOcr())
        summary = dialogue_summary(perception.process(_dialogue_frame(True), 1))
        assert summary == {"enabled": False}


class TestPromptResponder:
    def test_mapped_encounter_taps_key(self) -> None:
        responder, fake, status = make_responder()
        state = GameState()
        state.dialogue.encounter = "greet"
        result = responder.step(100.0, status, state)
        assert result == "greet"
        assert ("press", "e") in fake.events
        assert responder.taps == 1
        assert status.snapshot().action == "respond:greet"

    def test_no_encounter_no_input(self) -> None:
        responder, fake, status = make_responder()
        assert responder.step(100.0, status, GameState()) is None
        assert fake.events == []

    def test_unmapped_encounter_ignored(self) -> None:
        responder, fake, status = make_responder({"greet": "interact"})
        state = GameState()
        state.dialogue.encounter = "rob"
        assert responder.step(100.0, status, state) is None
        assert fake.events == []

    def test_cooldown_blocks_rapid_taps(self) -> None:
        responder, fake, status = make_responder(cooldown=2.0)
        state = GameState()
        state.dialogue.encounter = "greet"
        assert responder.step(100.0, status, state) == "greet"
        assert responder.step(100.5, status, state) is None
        assert responder.step(102.1, status, state) == "greet"
        assert responder.taps == 2

    def test_refused_tap_does_not_consume_cooldown(self) -> None:
        responder, fake, status = make_responder()
        fake.fail_keys.add("e")
        state = GameState()
        state.dialogue.encounter = "greet"
        assert responder.step(100.0, status, state) is None
        assert responder.taps == 0
        fake.fail_keys.clear()
        assert responder.step(100.1, status, state) == "greet"

    def test_empty_mapping_never_taps(self) -> None:
        responder, fake, status = make_responder(respond_to={})
        state = GameState()
        state.dialogue.encounter = "greet"
        assert responder.step(100.0, status, state) is None
        assert fake.events == []
