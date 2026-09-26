"""Perception pipeline (Phase 3): frame -> HUD + world readings -> GameState.

Combines the always-on HUD region detectors, the Phase 3 world estimators
(minimap/sky/weapon/horse) with the throttled OCR pass, and projects results
onto the shared :class:`GameState`. Values are only written when actually
detected; untouched fields stay ``None``/default so downstream planning never
acts on a guess.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from src.config import VisionConfig
from src.state.game_state import GameState
from src.vision.hud import HudDetection, HudReader, RegionReading, scale_regions
from src.vision.ocr import OcrEngine, create_ocr
from src.vision.world import WorldReader, WorldReading

_AMMO_RE = re.compile(r"\d{1,4}")


@dataclass
class PerceptionResult:
    hud: HudDetection
    world: WorldReading
    prompt_text: str | None = None
    prompt_changed: bool = False
    ammo: int | None = None
    ocr_engine: str = "off"
    ocr_running: bool = False
    frame_id: int = 0
    latency_ms: float = 0.0


class Perception:
    """Owns the per-frame HUD/world/OCR work; safe at the main loop rate."""

    def __init__(self, cfg: VisionConfig, ocr: OcrEngine | None = None) -> None:
        self._cfg = cfg
        self._hud = HudReader(cfg.hud)
        self._world = WorldReader(cfg.world, cfg.hud)
        self._ocr = ocr if ocr is not None else create_ocr(cfg.ocr)
        self._every = max(1, cfg.ocr.every_n_frames)
        self._counter = 0
        self._ammo_counter = 0
        self._last_prompt: str = ""
        self._minimap_frac = cfg.hud.regions.get("minimap")
        self._ammo: int | None = None

    @property
    def last_prompt(self) -> str:
        return self._last_prompt

    @property
    def ocr_engine(self) -> str:
        return self._ocr.name

    def process(self, frame: np.ndarray, frame_id: int) -> PerceptionResult:
        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]
        hud = self._hud.process(frame)
        minimap = None
        if self._minimap_frac is not None:
            scaled = scale_regions({"m": list(self._minimap_frac)}, fw, fh)
            minimap = scaled.get("m")
        world = self._world.process(frame, minimap)
        result = PerceptionResult(
            hud=hud, world=world, frame_id=frame_id, ocr_engine=self._ocr.name,
        )

        if self._cfg.hud.enabled and self._ocr.available:
            result.ocr_running = True
            prompt = hud.prompt
            if prompt is not None and prompt.present:
                self._counter += 1
                if self._counter % self._every == 0:
                    x, y, w, h = prompt.bbox
                    text, conf = self._ocr.read(frame[y:y + h, x:x + w])
                    if text and conf * 100.0 >= self._cfg.ocr.min_confidence:
                        if text != self._last_prompt:
                            result.prompt_changed = True
                            self._last_prompt = text
            weapon = world.weapon
            if weapon is not None and weapon.ammo_visible:
                self._ammo_counter += 1
                if self._ammo_counter % self._every == 0:
                    x, y, w, h = weapon.bbox
                    text, conf = self._ocr.read(frame[y:y + h, x:x + w])
                    if text and conf * 100.0 >= self._cfg.ocr.min_confidence:
                        match = _AMMO_RE.search(text)
                        if match:
                            self._ammo = int(match.group())
        result.prompt_text = self._last_prompt or None
        result.ammo = self._ammo
        if world.weapon is not None:
            world.weapon.ammo = self._ammo
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def overlay_boxes(
        self, result: PerceptionResult
    ) -> tuple[list[list[int]], list[str]]:
        boxes = list(result.hud.boxes) + list(result.world.boxes)
        labels = list(result.hud.labels) + list(result.world.labels)
        return boxes, labels

    def apply_to_state(self, state: GameState, result: PerceptionResult) -> None:
        """Project detections onto the shared game state (detections only)."""
        state.timestamp = time.time()
        state.frame_id = result.frame_id
        state.source = "hud+world"
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

        self._apply_world(state, result)

        values = [
            state.confidence.player, state.confidence.navigation,
            state.confidence.mission, state.confidence.combat,
        ]
        state.confidence.overall = round(sum(values) / len(values), 3)

    def _apply_world(self, state: GameState, result: PerceptionResult) -> None:
        world = result.world
        if world.skipped:
            return
        env = state.environment
        wc = self._cfg.world

        minimap = world.minimap
        if minimap is not None and minimap.present:
            env.water_detected = minimap.water_frac >= wc.water_min_frac
            env.road_detected = minimap.road_frac >= wc.road_min_frac
            env.wilderness = minimap.green_frac >= wc.green_min_frac
            boost = 0.7 if minimap.marker_present else 0.4
            state.confidence.navigation = max(state.confidence.navigation, boost)

        sky = world.sky
        if sky is not None and sky.present and sky.time_of_day:
            env.time_of_day = sky.time_of_day
            env.weather = sky.weather

        weapon = world.weapon
        if weapon is not None and weapon.ammo is not None:
            state.player.weapon_ammo = weapon.ammo

        horse = world.horse
        if horse is not None and horse.detected:
            state.horse.detected = True
            if horse.health is not None:
                state.horse.health = horse.health
            if horse.stamina is not None:
                state.horse.stamina = horse.stamina
            state.confidence.horse = horse.confidence


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


def world_summary(result: PerceptionResult) -> dict[str, object]:
    """Compact world snapshot for telemetry events."""
    world = result.world
    if world.skipped:
        return {"enabled": False}
    minimap = world.minimap
    sky = world.sky
    horse = world.horse
    return {
        "enabled": True,
        "sky": sky.time_of_day if sky else None,
        "weather": sky.weather if sky else None,
        "road": (minimap.road_frac if minimap else 0.0),
        "water": (minimap.water_frac if minimap else 0.0),
        "marker": bool(minimap and minimap.marker_present),
        "ammo": result.ammo,
        "horse_detected": bool(horse and horse.detected),
        "latency_ms": round(world.latency_ms, 2),
    }


def _gauge_value(hud: HudDetection, key: str) -> float | None:
    reading: RegionReading | None = hud.gauges.get(key)
    if reading is None:
        return None
    return reading.value
