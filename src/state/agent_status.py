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
