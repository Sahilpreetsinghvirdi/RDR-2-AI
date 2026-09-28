"""Tick-based locomotion control: non-blocking movement holds and camera turns.

Phase 4 plumbing. Every command is clamped by configuration, advances on
``tick()`` from the decision loop (never sleeps), and is released by
``stop()`` on pause, takeover, watchdog faults, lost focus or shutdown.
The InputController's ``allow_input`` guard still gates every keystroke.
"""

from __future__ import annotations

import logging
import time

from src.config import ControlConfig
from src.input.input_controller import InputController

log = logging.getLogger(__name__)

DIRECTIONS = ("forward", "back", "strafe_left", "strafe_right")


class LocomotionController:
    """Plans movement holds and look turns as commands the loop ticks forward."""

    def __init__(self, cfg: ControlConfig, input_ctrl: InputController) -> None:
        self._cfg = cfg
        self._input = input_ctrl
        self._active_keys: tuple[str, ...] = ()
        self._deadline = 0.0
        self._label = ""
        self._moves_started = 0
        self._turns_done = 0.0

    @property
    def active(self) -> bool:
        return bool(self._active_keys)

    @property
    def label(self) -> str:
        return self._label

    @property
    def moves_started(self) -> int:
        return self._moves_started

    @property
    def degrees_turned(self) -> float:
        return self._turns_done

    def move(self, direction: str, duration_s: float, *, sprint: bool = False) -> bool:
        """Hold *direction*'s key for a clamped duration; non-blocking."""
        if direction not in DIRECTIONS:
            log.warning("unknown move direction: %r", direction)
            return False
        duration = max(self._cfg.min_step_s, min(duration_s, self._cfg.max_step_s))
        keys = [self._cfg.keys[direction]]
        if sprint:
            keys.append(self._cfg.keys["sprint"])
        self.stop()
        held: list[str] = []
        ok = True
        for key in keys:
            if self._input.key_down(key):
                held.append(key)
            else:
                ok = False
        if not ok:
            for key in held:
                self._input.key_up(key)
            return False
        self._active_keys = tuple(keys)
        self._deadline = time.monotonic() + duration
        self._label = f"move:{direction} {duration:.1f}s"
        self._moves_started += 1
        log.info("%s", self._label)
        return True

    def turn(self, degrees: float) -> bool:
        """Look left (negative) or right (positive); clamped per call."""
        if abs(degrees) < 1e-6:
            return True
        cfg = self._cfg
        applied = max(-cfg.max_turn_deg_per_tick, min(degrees, cfg.max_turn_deg_per_tick))
        dx = int(round(applied * cfg.mouse_px_per_degree))
        if dx == 0:
            return True
        ok = self._input.mouse_move(dx, 0)
        if ok:
            self._turns_done += applied
        return ok

    def tap(self, action: str) -> bool:
        """Single press of a configured action key (jump/interact)."""
        key = self._cfg.keys.get(action)
        if key is None:
            log.warning("unknown action key: %r", action)
            return False
        return self._input.press(key)

    def tick(self) -> None:
        """Release held movement keys once their deadline passes."""
        if self._active_keys and time.monotonic() >= self._deadline:
            self.stop()

    def stop(self) -> None:
        """Release movement keys immediately; safe to call at any time."""
        for key in self._active_keys:
            self._input.key_up(key)
        if self._active_keys:
            log.debug("locomotion stop: released %s", ", ".join(self._active_keys))
        self._active_keys = ()
        self._label = ""

    def snapshot(self) -> dict[str, object]:
        return {
            "active": self.active,
            "label": self._label,
            "keys": list(self._active_keys),
            "moves_started": self._moves_started,
            "degrees_turned": round(self._turns_done, 1),
        }


