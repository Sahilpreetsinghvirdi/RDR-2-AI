"""Structured game state model shared by vision, planning and telemetry.

Phase 1 only constructs the data model; detections are populated starting in
Phase 2. Every field defaults to "unknown" with zero confidence rather than a
guessed value.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlayerState:
    position_estimate: list[float] | None = None
    health: float | None = None
    stamina: float | None = None
    dead_eye: float | None = None
    weapon: str | None = None
    weapon_ammo: int | None = None
    mounted: bool | None = None
    movement_state: str = "unknown"
    combat_state: str = "unknown"
    screen_bbox: list[int] | None = None


@dataclass
class HorseState:
    detected: bool = False
    distance: float | None = None
    health: float | None = None
    stamina: float | None = None
    mounted: bool | None = None
    screen_bbox: list[int] | None = None


@dataclass
class EnvironmentState:
    road_detected: bool = False
    water_detected: bool = False
    town_detected: bool = False
    building_detected: bool = False
    wilderness: bool = False
    cover_available: bool = False
    time_of_day: str | None = None
    weather: str | None = None


@dataclass
class EntityDetection:
    kind: str
    bbox: list[int]
    confidence: float
    label: str | None = None
    velocity: list[float] | None = None


@dataclass
class MissionState:
    active: bool = False
    objective_text: str | None = None
    objective_type: str | None = None
    # Nearest minimap marker to Arthur, excluding his own arrow: [dx, dy]
    # in -0.5..0.5. The marker's identity (Dutch, John, stranger, waypoint)
    # is NOT readable - arrival plus the mission title card identifies it.
    objective_location_estimate: list[float] | None = None
    prompt_text: str | None = None
    prompt_action: str | None = None


@dataclass
class ThreatState:
    level: str = "none"
    enemies_detected: int | None = None  # None = minimap not read this frame
    enemy_offset: list[float] | None = None  # dx, dy of nearest red blip (-0.5..0.5)
    wanted_level: int = 0
    incoming_fire: bool = False


@dataclass
class DialogueState:
    """Subtitle band + encounter keyword detections (Phase 6)."""

    active: bool | None = None     # None until first dialogue pass
    text: str | None = None        # last OCR'd subtitle/prompt text
    encounter: str | None = None   # matched encounter_keyword, if any


@dataclass
class SurvivalState:
    """Core-triggered provisioning status (Phase 8).

    ``low_cores`` is filled only while ``survival.enabled`` is on; empty means
    "not evaluated", never "all cores fine".
    """

    active_remedy: str | None = None
    low_cores: list[str] = field(default_factory=list)


@dataclass
class EconomyState:
    """Cash/valuation. The gameplay HUD does not show it: stays unknown."""

    cash: float | None = None


@dataclass
class Confidence:
    overall: float = 0.0
    player: float = 0.0
    horse: float = 0.0
    navigation: float = 0.0
    combat: float = 0.0
    mission: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in asdict(self).items()}


@dataclass
class GameState:
    timestamp: float = field(default_factory=time.time)
    player: PlayerState = field(default_factory=PlayerState)
    horse: HorseState = field(default_factory=HorseState)
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    mission: MissionState = field(default_factory=MissionState)
    threat: ThreatState = field(default_factory=ThreatState)
    dialogue: DialogueState = field(default_factory=DialogueState)
    survival: SurvivalState = field(default_factory=SurvivalState)
    economy: EconomyState = field(default_factory=EconomyState)
    nearby_entities: list[EntityDetection] = field(default_factory=list)
    confidence: Confidence = field(default_factory=Confidence)
    frame_id: int | None = None
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = self.confidence.as_dict()
        return data

    @property
    def stale(self) -> bool:
        return (time.time() - self.timestamp) > 1.0
