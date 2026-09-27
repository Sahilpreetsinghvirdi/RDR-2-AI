"""Prompt-driven reactions: tap a mapped key when an encounter keyword shows up (Phase 6).

Reads ``game_state.dialogue.encounter`` (matched against on-screen prompt /
subtitle text by perception) and presses the configured control key once per
cooldown. Pure reaction layer: it never decides *what* the encounter means,
only forwards the configured keyword -> key mapping.
"""

from __future__ import annotations

import logging

from src.config import DialogueConfig
from src.control.locomotion import LocomotionController
from src.state.agent_status import AgentStatus
from src.state.game_state import GameState

log = logging.getLogger(__name__)


class PromptResponder:
    """Taps a configured key when an encounter keyword is currently visible."""

    def __init__(
        self,
        dialogue_cfg: DialogueConfig,
        locomotion: LocomotionController,
    ) -> None:
        self._respond_to = dict(dialogue_cfg.respond_to)
        self._cooldown = dialogue_cfg.respond_cooldown_s
        self._locomotion = locomotion
        self._last_at = -1e9
        self._taps = 0

    @property
    def taps(self) -> int:
        return self._taps

    def step(
        self, now: float, status: AgentStatus, game_state: GameState
    ) -> str | None:
        """React to the current encounter if mapped; returns the keyword or None."""
        encounter = game_state.dialogue.encounter
        if not encounter or encounter not in self._respond_to:
            return None
        if now - self._last_at < self._cooldown:
            return None
        keyname = self._respond_to[encounter]
        if not self._locomotion.tap(keyname):
            return None
        self._last_at = now
        self._taps += 1
        status.update(action=f"respond:{encounter}")
        log.info("responding to encounter '%s' via key '%s'", encounter, keyname)
        return encounter
