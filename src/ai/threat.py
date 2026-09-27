"""Rolling threat assessment from detected signals (Phase 7).

Refines :class:`ThreatState` from what perception measured: a health drop
inside a short window marks incoming fire, and the threat level ladder
(``none`` < ``enemy`` < ``wanted`` < ``under_fire``) is derived only from
those detections. It never sends input.
"""

from __future__ import annotations

import time
from collections import deque

from src.config import CombatConfig
from src.state.game_state import GameState

LEVEL_ORDER = ("none", "enemy", "wanted", "under_fire")


class ThreatAssessor:
    """Stateful detector of recent damage -> incoming_fire + level ladder."""

    def __init__(self, cfg: CombatConfig) -> None:
        self._cfg = cfg
        self._health: deque[tuple[float, float]] = deque()
        self._last_drop_at = -1e9

    def assess(self, now: float | None, state: GameState) -> None:
        """Update ``state.threat`` in place from the latest detections."""
        ts = time.monotonic() if now is None else now
        cfg = self._cfg
        health = state.player.health
        if health is not None:
            window = cfg.health_drop_window_s
            self._health.append((ts, health))
            while self._health and ts - self._health[0][0] > window:
                self._health.popleft()
            peak = max(value for _, value in self._health)
            if peak - health >= cfg.health_drop_frac:
                self._last_drop_at = ts

        hold = cfg.incoming_fire_hold_s
        state.threat.incoming_fire = (
            cfg.engage_on_incoming_fire and (ts - self._last_drop_at) <= hold
        )

        threat = state.threat
        if threat.incoming_fire:
            threat.level = "under_fire"
        elif threat.wanted_level >= 1:
            threat.level = "wanted"
        elif threat.enemies_detected:
            threat.level = "enemy"
        else:
            threat.level = "none"

        base = state.confidence.combat
        if threat.incoming_fire:
            state.confidence.combat = max(base, 0.8)
        elif threat.wanted_level >= 1:
            state.confidence.combat = max(base, 0.7)
        elif threat.enemies_detected:
            state.confidence.combat = max(base, 0.5)
