"""Perception pipeline (Phase 2): frame -> HUD readings -> GameState.

Combines the always-on HUD region detectors with the throttled OCR pass and
projects the results onto the shared :class:`GameState`. Values are only
written when actually detected; untouched fields stay ``None``/default so
downstream planning never acts on a guess.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.config import VisionConfig
from src.state.game_state import GameState
from src.vision.hud import HudDetection, HudReader, RegionReading
from src.vision.ocr import OcrEngine, create_ocr


@dataclass
class PerceptionResult:
    hud: HudDetection
    prompt_text: str | None = None
    prompt_changed: bool = False
    ocr_engine: str = "off"
    ocr_running: bool = False
    frame_id: int = 0
    latency_ms: float = 0.0


class Perception:
    """Owns the per-frame HUD/OCR work; safe to call at the main loop rate."""

    def __init__(self, cfg: VisionConfig, ocr: OcrEngine | None = None) -> None:
        self._cfg = cfg
        self._hud = HudReader(cfg.hud)
        self._ocr = ocr if ocr is not None else create_ocr(cfg.ocr)
        self._every = max(1, cfg.ocr.every_n_frames)
        self._counter = 0
        self._last_prompt: str = ""

    @property
    def last_prompt(self) -> str:
        return self._last_prompt

    @property
    def ocr_engine(self) -> str:
        return self._ocr.name

    def process(self, frame: np.ndarray, frame_id: int) -> PerceptionResult:
        t0 = time.perf_counter()
        hud = self._hud.process(frame)
        result = PerceptionResult(hud=hud, frame_id=frame_id, ocr_engine=self._ocr.name)
        if self._cfg.hud.enabled and self._ocr.available:
            result.ocr_running = True
            self._counter += 1
            prompt = hud.prompt
            if prompt is not None and prompt.present and self._counter % self._every == 0:
                x, y, w, h = prompt.bbox
                crop = frame[y:y + h, x:x + w]
                text, conf = self._ocr.read(crop)
                if text and conf * 100.0 >= self._cfg.ocr.min_confidence:
                    if text != self._last_prompt:
                        result.prompt_changed = True
                        self._last_prompt = text
        result.prompt_text = self._last_prompt or None
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def overlay_boxes(self, result: PerceptionResult) -> tuple[list[list[int]], list[str]]:
        hud = result.hud
        return hud.boxes, hud.labels

    def apply_to_state(self, state: GameState, result: PerceptionResult) -> None:
        """Project detections onto the shared game state (detections only)."""
        state.timestamp = time.time()
        state.frame_id = result.frame_id
        state.source = "hud"
        hud = result.hud
        if hud.skipped:
            return

        confidences: list[float] = []
        mapping = {"health": "health", "stamina": "stamina", "dead_eye": "dead_eye"}
        for key, attr in mapping.items():
            reading = hud.gauges.get(key)
            if reading is not None and reading.value is not None:
                setattr(state.player, attr, reading.value)
                confidences.append(reading.confidence)
        if confidences:
            state.confidence.player = round(sum(confidences) / len(confidences), 3)

        if hud.minimap is not None and hud.minimap.present:
            state.confidence.navigation = hud.minimap.confidence

        if hud.prompt is not None and hud.prompt.present:
            state.confidence.mission = hud.prompt.confidence
            if result.prompt_text:
                state.mission.prompt_text = result.prompt_text

        if hud.wanted is not None and hud.wanted.value:
            state.threat.wanted_level = int(hud.wanted.value)
            state.threat.level = "wanted"
            state.confidence.combat = hud.wanted.confidence

        values = [
            state.confidence.player, state.confidence.navigation,
            state.confidence.mission, state.confidence.combat,
        ]
        state.confidence.overall = round(sum(values) / len(values), 3)


def hud_summary(result: PerceptionResult) -> dict[str, object]:
    """Compact HUD snapshot for telemetry events."""
    hud = result.hud
    return {
        "health": _gauge_value(hud, "health"),
        "stamina": _gauge_value(hud, "stamina"),
        "dead_eye": _gauge_value(hud, "dead_eye"),
        "minimap": bool(hud.minimap and hud.minimap.present),
        "prompt_visible": bool(hud.prompt and hud.prompt.present),
        "prompt": result.prompt_text or "",
        "wanted": (hud.wanted.value if hud.wanted else None),
        "ocr": result.ocr_engine,
        "latency_ms": round(result.latency_ms, 2),
    }


def _gauge_value(hud: HudDetection, key: str) -> float | None:
    reading: RegionReading | None = hud.gauges.get(key)
    if reading is None:
        return None
    return reading.value
