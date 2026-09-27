"""Thread-safe snapshot of everything the debug dashboard displays."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field


@dataclass
class AgentStatus:
    state: str = "INIT"
    phase: int = 1
    message: str = "starting"
    window_title: str = ""
    window_bounds: list[int] | None = None
    window_focused: bool = False
    capture_backend: str = "none"
    capture_fps: float = 0.0
    capture_latency_ms: float = 0.0
    frame_age_ms: float = 0.0
    frame_id: int = 0
    frame_size: list[int] | None = None
    vision_ms: float = 0.0
    brightness: float = 0.0
    motion: float = 0.0
    loop_hz: float = 0.0
    input_latency_ms: float = 0.0
    held_keys: list[str] = field(default_factory=list)
    held_buttons: list[str] = field(default_factory=list)
    goal: str = "IDLE"
    action: str = "none"
    confidence: float | None = None
    last_action: str = ""
    demo_step: str = ""
    error: str = ""
    hud_health: float | None = None
    hud_stamina: float | None = None
    hud_dead_eye: float | None = None
    hud_minimap: bool | None = None
    hud_prompt_visible: bool = False
    hud_prompt: str = ""
    hud_boxes: list[list[int]] = field(default_factory=list)
    hud_labels: list[str] = field(default_factory=list)
    hud_ms: float = 0.0
    ocr_engine: str = "off"
    world_time: str | None = None
    world_weather: str | None = None
    world_ammo: int | None = None
    world_horse_detected: bool = False
    world_ms: float = 0.0
    dialogue_active: bool | None = None
    dialogue_text: str = ""
    dialogue_encounter: str = ""
    dialogue_ms: float = 0.0
    threat_level: str = "none"
    threat_enemies: int | None = None
    threat_wanted: int = 0
    threat_fire: bool = False
    survival_active: str = ""
    survival_low: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class StatusTracker:
    """Single writer/reader-friendly wrapper over :class:`AgentStatus`."""

    def __init__(self, phase: int = 1) -> None:
        self._lock = threading.Lock()
        self._status = AgentStatus(phase=phase)

    def update(self, **kwargs: object) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self._status, key):
                    raise AttributeError(f"AgentStatus has no field {key!r}")
                setattr(self._status, key, value)

    def snapshot(self) -> AgentStatus:
        with self._lock:
            return AgentStatus(**asdict(self._status))
