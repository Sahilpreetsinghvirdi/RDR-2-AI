"""Perception pipeline (Phase 3): frame -> HUD + world readings -> GameState.

Combines the always-on HUD region detectors, the Phase 3 world estimators
(minimap/sky/weapon/horse) with the throttled OCR pass, and projects results
onto the shared :class:`GameState`. Values are only written when actually
detected; untouched fields stay ``None``/default so downstream planning never
acts on a guess.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import numpy as np

from src.config import VisionConfig
from src.state.game_state import GameState
from src.vision.dialogue import DialogueReader, DialogueReading, match_keyword
from src.vision.hud import HudDetection, HudReader, RegionReading, scale_regions
from src.vision.ocr import OcrEngine, create_ocr
from src.vision.world import WorldReader, WorldReading

_AMMO_RE = re.compile(r"\d{1,4}")


@dataclass
class PerceptionResult:
    hud: HudDetection
    world: WorldReading
    dialogue: DialogueReading
    prompt_text: str | None = None
    prompt_changed: bool = False
    objective_text: str | None = None
    ammo: int | None = None
    encounter: str | None = None
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
        self._dialogue = DialogueReader(cfg.dialogue)
        self._ocr = ocr if ocr is not None else create_ocr(cfg.ocr)
        self._every = max(1, cfg.ocr.every_n_frames)
        self._dlg_every = max(1, cfg.dialogue.every_n_frames)
        self._counter = 0
        self._ammo_counter = 0
        self._obj_counter = 0
        self._dlg_counter = 0
        self._last_prompt: str = ""
        self._last_dialogue_text: str = ""
        self._minimap_frac = cfg.hud.regions.get("minimap")
        self._ammo: int | None = None

    @property
    def last_prompt(self) -> str:
        return self._last_prompt

    @property
    def last_dialogue_text(self) -> str:
        return self._last_dialogue_text

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
        dialogue = self._dialogue.process(frame)
        result = PerceptionResult(
            hud=hud, world=world, dialogue=dialogue, frame_id=frame_id,
            ocr_engine=self._ocr.name,
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
            objective = hud.objective
            if objective is not None and objective.present:
                self._obj_counter += 1
                if self._obj_counter % self._every == 0:
                    x, y, w, h = objective.bbox
                    text, conf = self._ocr.read(frame[y:y + h, x:x + w])
                    if text and conf * 100.0 >= self._cfg.ocr.min_confidence:
                        result.objective_text = text
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
            elif weapon is not None:
                self._ammo = None
        if dialogue.present and self._cfg.dialogue.enabled and self._ocr.available:
            self._dlg_counter += 1
            if self._dlg_counter % self._dlg_every == 0:
                fh2, fw2 = frame.shape[:2]
                dx, dy, dw, dh = self._cfg.dialogue.region
                bx = int(dx * fw2)
                by = int(dy * fh2)
                bw = int(dw * fw2)
                bh = int(dh * fh2)
                text, conf = self._ocr.read(frame[by:by + bh, bx:bx + bw])
                if text and conf * 100.0 >= self._cfg.ocr.min_confidence:
                    self._last_dialogue_text = text
        elif not dialogue.present:
            self._dlg_counter = 0
            self._last_dialogue_text = ""
        result.prompt_text = self._last_prompt or None
        prompt_visible = (
            result.hud.prompt is not None and result.hud.prompt.present
        )
        if not prompt_visible:
            result.prompt_text = None
        result.dialogue.text = self._last_dialogue_text or None
        result.encounter = self._match_encounter(result)
        result.ammo = self._ammo
        if world.weapon is not None:
            world.weapon.ammo = self._ammo
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def _match_encounter(self, result: PerceptionResult) -> str | None:
        """Match keywords against text that is on screen right now."""
        keywords = self._cfg.dialogue.encounter_keywords
        prompt_visible = result.hud.prompt is not None and result.hud.prompt.present
        if prompt_visible:
            hit = match_keyword(result.prompt_text, keywords)
            if hit is not None:
                return hit
        if result.dialogue.present:
            return match_keyword(result.dialogue.text, keywords)
        return None

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
        state.source = "hud+world+dialogue"
        self._apply_dialogue(state, result)
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
        else:
            state.mission.prompt_text = None

        objective = hud.objective
        if objective is not None and objective.present:
            if result.objective_text:
                state.mission.objective_text = result.objective_text
                state.mission.active = True
        elif objective is not None:
            state.mission.objective_text = None
            state.mission.active = False

        if hud.wanted is not None:
            if hud.wanted.value:
                state.threat.wanted_level = int(hud.wanted.value)
                state.threat.level = "wanted"
                state.confidence.combat = hud.wanted.confidence
            else:
                state.threat.wanted_level = 0

        self._apply_world(state, result)

        values = [
            state.confidence.player, state.confidence.navigation,
            state.confidence.mission, state.confidence.combat,
        ]
        state.confidence.overall = round(sum(values) / len(values), 3)

    def _apply_dialogue(self, state: GameState, result: PerceptionResult) -> None:
        dialogue = result.dialogue
        if dialogue.skipped:
            return
        state.dialogue.active = dialogue.present
        state.dialogue.text = dialogue.text if dialogue.present else None
        state.dialogue.encounter = result.encounter

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
            state.threat.enemies_detected = minimap.enemy_dots
            state.threat.enemy_offset = (
                minimap.enemy_offset if minimap.enemy_dots else None
            )
            state.mission.objective_location_estimate = (
                _nearest_marker(minimap.marker_blobs, wc.marker_center_deadzone)
            )

        sky = world.sky
        if sky is not None and sky.present and sky.time_of_day:
            env.time_of_day = sky.time_of_day
            env.weather = sky.weather

        weapon = world.weapon
        if weapon is not None:
            state.player.weapon_ammo = weapon.ammo

        horse = world.horse
        if horse is not None:
            if horse.detected:
                state.horse.detected = True
                if horse.health is not None:
                    state.horse.health = horse.health
                if horse.stamina is not None:
                    state.horse.stamina = horse.stamina
                state.confidence.horse = horse.confidence
            else:
                state.horse.detected = False
                state.horse.health = None
                state.horse.stamina = None


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
        "enemies": (minimap.enemy_dots if minimap else None),
        "ammo": result.ammo,
        "horse_detected": bool(horse and horse.detected),
        "latency_ms": round(world.latency_ms, 2),
    }


def dialogue_summary(result: PerceptionResult) -> dict[str, object]:
    """Compact dialogue snapshot for telemetry events."""
    dialogue = result.dialogue
    if dialogue.skipped:
        return {"enabled": False}
    return {
        "enabled": True,
        "present": dialogue.present,
        "text": dialogue.text or "",
        "encounter": result.encounter or "",
        "confidence": round(dialogue.confidence, 3),
        "latency_ms": round(dialogue.latency_ms, 2),
    }


def threat_summary(state: GameState) -> dict[str, object]:
    """Compact threat snapshot for telemetry events."""
    threat = state.threat
    return {
        "level": threat.level,
        "enemies": threat.enemies_detected,
        "wanted": threat.wanted_level,
        "incoming_fire": threat.incoming_fire,
    }


def _gauge_value(hud: HudDetection, key: str) -> float | None:
    reading: RegionReading | None = hud.gauges.get(key)
    if reading is None:
        return None
    return reading.value


def _nearest_marker(
    blobs: list[list[float]] | None, deadzone: float
) -> list[float] | None:
    """Nearest marker blob to the minimap centre, excluding Arthur's arrow."""
    if not blobs:
        return None
    best: list[float] | None = None
    best_key: tuple[float, float] | None = None
    for blob in blobs:
        if len(blob) != 3:
            continue
        dist = math.hypot(float(blob[0]), float(blob[1]))
        if dist < deadzone:
            continue
        key = (dist, -float(blob[2]))
        if best_key is None or key < best_key:
            best_key = key
            best = [float(blob[0]), float(blob[1])]
    return best
