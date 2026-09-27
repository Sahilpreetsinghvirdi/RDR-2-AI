"""Survival planner: core-triggered provisioning macros (Phase 8).

Watches the HUD-derived cores (health / stamina / dead eye) and, when one
drops below a configured threshold, runs that remedy's key sequence through
the focus-gated InputController. Runs are cooldown-limited and must recover
past the threshold plus a margin before re-arming (hysteresis), and - when
``require_clear`` is on - nothing starts or continues while a threat is
active. Sequences are raw key names tuned by the user to their control
scheme; nothing about the in-game menus is assumed.
"""

from __future__ import annotations

import logging

from src.config import RemedyConfig, SurvivalConfig
from src.control.locomotion import LocomotionController
from src.input.input_controller import InputController
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState

log = logging.getLogger(__name__)

_CORE_ATTRS = {"health": "health", "stamina": "stamina", "dead_eye": "dead_eye"}


def survival_summary(state: GameState) -> dict[str, object]:
    """Compact survival snapshot for telemetry events."""
    return {
        "active": state.survival.active_remedy or "",
        "low_cores": list(state.survival.low_cores),
    }


class SurvivalPlanner:
    """Duck-typed planner: ``step(now, status, frame, game_state) -> busy``."""

    def __init__(
        self,
        cfg: SurvivalConfig,
        input_ctrl: InputController,
        locomotion: LocomotionController,
    ) -> None:
        self._cfg = cfg
        self._input = input_ctrl
        self._locomotion = locomotion
        self._active: str | None = None
        self._sequence: list[str] = []
        self._index = 0
        self._next_at = 0.0
        self._runs: dict[str, float] = {}
        self._armed: dict[str, bool] = {}
        self._completed = 0

    @property
    def busy(self) -> bool:
        return self._active is not None

    @property
    def completed(self) -> int:
        return self._completed

    def step(
        self,
        now: float,
        status: StatusTracker,
        frame: object | None = None,
        game_state: GameState | None = None,
    ) -> bool:
        """Advance one control tick; True while a remedy sequence is running."""
        self._locomotion.tick()
        if game_state is None:
            return self.busy
        game_state.survival.low_cores = self._low_cores(game_state)
        if self._active is not None:
            return self._advance(now, status, game_state)
        self._maybe_start(now, status, game_state)
        game_state.survival.active_remedy = self._active
        return self.busy

    def abort(self) -> None:
        """Drop a running sequence without recording a completed run."""
        if self._active is not None:
            log.info("survival: aborted remedy %r", self._active)
        self._active = None
        self._sequence = []
        self._index = 0

    def stop(self, game_state: GameState | None = None) -> None:
        """Halt any running remedy (pause / takeover / fault); safe repeatedly."""
        if self._active is None and self._index == 0:
            return
        self.abort()
        if game_state is not None:
            game_state.survival.active_remedy = None

    def _remedies(self) -> list[tuple[str, RemedyConfig]]:
        if not isinstance(self._cfg.remedies, dict):
            return []
        return [
            (name, remedy)
            for name, remedy in self._cfg.remedies.items()
            if isinstance(name, str) and isinstance(remedy, RemedyConfig)
        ]

    def _low_cores(self, state: GameState) -> list[str]:
        low: list[str] = []
        for _, remedy in self._remedies():
            for core, threshold in remedy.needs.items():
                attr = _CORE_ATTRS.get(core)
                if attr is None:
                    continue
                value = getattr(state.player, attr, None)
                if value is not None and value < threshold and core not in low:
                    low.append(core)
        return low

    def _maybe_start(
        self, now: float, status: StatusTracker, state: GameState
    ) -> None:
        threat_clear = state.threat.level == "none"
        for name, remedy in self._remedies():
            if not self._armed.get(name, True):
                if self._recovered(remedy, state):
                    self._armed[name] = True
                else:
                    continue
            if now - self._runs.get(name, -1e9) < remedy.cooldown_s:
                continue
            if not self._any_low(remedy, state):
                continue
            if self._cfg.require_clear and not threat_clear:
                continue
            self._active = name
            self._sequence = list(remedy.keys)
            self._index = 0
            self._next_at = now
            log.info(
                "survival: starting remedy %r (%d keys)", name, len(self._sequence)
            )
            status.update(goal=f"SURVIVAL (phase {status.phase})",
                          action=f"survival:{name}")
            return

    def _advance(
        self, now: float, status: StatusTracker, state: GameState
    ) -> bool:
        name = self._active
        if name is None:
            return False
        remedy_map = dict(self._remedies())
        remedy = remedy_map.get(name)
        if remedy is None:
            self.abort()
            return False

        if self._cfg.require_clear and state.threat.level != "none":
            self._runs[name] = now
            self.abort()
            state.survival.active_remedy = None
            status.update(action="none")
            return False

        if now < self._next_at:
            return True

        key = remedy.keys[self._index]
        if not self._input.press(key, hold_ms=remedy.hold_ms):
            return True  # guard refused; retry next tick
        self._index += 1
        self._next_at = now + remedy.gap_s
        status.update(
            action=f"survival:{name} {self._index}/{len(self._sequence)}",
            goal=f"SURVIVAL (phase {status.phase})",
        )
        if self._index >= len(self._sequence):
            self._runs[name] = now
            self._armed[name] = False
            self._completed += 1
            self._active = None
            self._sequence = []
            self._index = 0
            state.survival.active_remedy = None
            log.info("survival: remedy %r completed", name)
            status.update(action=f"survival:{name} done")
        return self._active is not None

    def _any_low(self, remedy: RemedyConfig, state: GameState) -> bool:
        for core, threshold in remedy.needs.items():
            attr = _CORE_ATTRS.get(core)
            if attr is None:
                continue
            value = getattr(state.player, attr, None)
            if value is not None and value < threshold:
                return True
        return False

    def _recovered(self, remedy: RemedyConfig, state: GameState) -> bool:
        """True when every *known* needed core is past threshold + margin."""
        margin = self._cfg.recover_margin
        for core, threshold in remedy.needs.items():
            attr = _CORE_ATTRS.get(core)
            if attr is None:
                continue
            value = getattr(state.player, attr, None)
            if value is None:
                continue
            if value < threshold + margin:
                return False
        return True
