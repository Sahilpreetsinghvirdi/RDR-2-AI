"""Self-defense combat planner: aim at minimap threats, fire short bursts (Phase 7).

Engages only on detected signals (wanted stars, minimap red blips, incoming
fire from the threat assessor), aims using the nearest blip's minimap offset,
fires in config-clamped bursts, retreats below a health threshold, and stands
down after the threat has been clear for a configured period. The
InputController's focus/state guards still gate every button event.
"""

from __future__ import annotations

import logging
import math

from src.config import CombatConfig
from src.control.locomotion import LocomotionController
from src.input.input_controller import InputController
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState

log = logging.getLogger(__name__)


class CombatPlanner:
    """Duck-typed planner: ``step(now, status, frame, game_state)``."""

    def __init__(
        self,
        cfg: CombatConfig,
        locomotion: LocomotionController,
        input_ctrl: InputController,
    ) -> None:
        self._cfg = cfg
        self._locomotion = locomotion
        self._input = input_ctrl
        self._engaged = False
        self._calm_since: float | None = None
        self._firing_until: float | None = None
        self._next_burst = 0.0
        self._next_aim = 0.0
        self._bursts = 0

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def bursts(self) -> int:
        return self._bursts

    def step(
        self,
        now: float,
        status: StatusTracker,
        frame: object | None = None,
        game_state: GameState | None = None,
    ) -> str | None:
        """Advance one combat tick; returns the active label or None."""
        self._locomotion.tick()
        self._release_fire(now)
        if game_state is None:
            return None

        reason = self._threat_reason(game_state)
        if reason is None:
            if not self._engaged:
                return None
            if self._calm_since is None:
                self._calm_since = now
            elif now - self._calm_since >= self._cfg.disengage_clear_s:
                self._disengage(status)
            return None

        self._calm_since = None
        if not self._engaged:
            self._engage(now, status, reason)

        health = game_state.player.health
        if health is not None and health <= self._cfg.retreat_health_frac:
            self._locomotion.move("back", self._cfg.retreat_step_s)
            status.update(
                action="combat:retreat",
                goal=f"COMBAT (phase {status.phase})",
            )
            return "retreat"

        label = "combat:cover"
        if self._aim(now, game_state):
            label = "combat:aim"
        if self._try_fire(now, game_state):
            label = "combat:fire"
        status.update(action=label, goal=f"COMBAT (phase {status.phase})")
        return label

    def stop(self) -> None:
        """Release the fire button and movement; safe to call at any time."""
        self._release_fire(-1e9, force=True)
        self._locomotion.stop()

    def _threat_reason(self, state: GameState) -> str | None:
        cfg = self._cfg
        threat = state.threat
        if cfg.engage_on_incoming_fire and threat.incoming_fire:
            return "incoming_fire"
        if threat.wanted_level >= cfg.engage_min_wanted and threat.wanted_level >= 1:
            return "wanted"
        if (threat.enemies_detected or 0) >= cfg.engage_enemies:
            return "enemies"
        return None

    def _engage(self, now: float, status: StatusTracker, reason: str) -> None:
        self._engaged = True
        self._calm_since = None
        self._next_burst = now
        self._next_aim = now
        self._locomotion.stop()
        log.warning("combat engaged (%s)", reason)
        status.update(goal=f"COMBAT (phase {status.phase})",
                      action=f"combat:{reason}")

    def _disengage(self, status: StatusTracker) -> None:
        self.stop()
        self._engaged = False
        self._calm_since = None
        log.info("combat: threat clear, standing down")
        status.update(action="none")

    def _aim(self, now: float, state: GameState) -> bool:
        if now < self._next_aim:
            return False
        offset = state.threat.enemy_offset
        if not offset or not state.threat.enemies_detected:
            return False
        dx, dy = offset[0], offset[1]
        self._next_aim = now + self._cfg.aim_interval_s
        angle = math.degrees(math.atan2(dx, -dy))
        if abs(angle) < self._cfg.aim_dead_zone_deg:
            return False
        step = max(
            -self._cfg.aim_max_step_deg,
            min(angle, self._cfg.aim_max_step_deg),
        )
        return self._locomotion.turn(step)

    def _try_fire(self, now: float, state: GameState) -> bool:
        if now < self._next_burst or self._firing_until is not None:
            return False
        if state.player.weapon_ammo == 0:
            return False
        if not self._input.mouse_down(self._cfg.fire_button):
            return False
        self._firing_until = now + self._cfg.burst_s
        self._next_burst = now + self._cfg.burst_interval_s
        self._bursts += 1
        log.debug("combat: burst %d", self._bursts)
        return True

    def _release_fire(self, now: float, force: bool = False) -> None:
        if self._firing_until is None:
            return
        if force or now >= self._firing_until:
            self._input.mouse_up(self._cfg.fire_button)
            self._firing_until = None
