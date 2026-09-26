"""Scripted autopilot planner: a fixed look-around / walk cycle (Phase 4).

Exercises the locomotion layer end to end so later phases can replace the
script with a real decision layer. Runs only when ``control.enabled`` is on
(or ``--autopilot`` is passed) and only inside the RUNNING state, which the
main loop already gates on window presence and focus.
"""

from __future__ import annotations

import logging

from src.config import ControlConfig
from src.control.locomotion import LocomotionController
from src.state.agent_status import StatusTracker

log = logging.getLogger(__name__)

Action = dict[str, object]

SCRIPT: tuple[Action, ...] = (
    {"kind": "turn", "amount": -40.0},
    {"kind": "move", "direction": "forward", "duration": 2.0},
    {"kind": "turn", "amount": 40.0},
    {"kind": "move", "direction": "strafe_left", "duration": 1.0},
    {"kind": "move", "direction": "back", "duration": 1.0},
    {"kind": "turn", "amount": 30.0},
    {"kind": "wait", "duration": 1.0},
    {"kind": "move", "direction": "forward", "duration": 1.5, "sprint": True},
    {"kind": "tap", "action": "jump"},
    {"kind": "wait", "duration": 1.5},
)


class ScriptedPlanner:
    """Advances through ``SCRIPT`` one action at a time, on loop ticks."""

    def __init__(self, cfg: ControlConfig, locomotion: LocomotionController) -> None:
        self._cfg = cfg
        self._locomotion = locomotion
        self._index = 0
        self._started = False
        self._wait_until = 0.0
        self._turn_remaining = 0.0
        self._cycles = 0

    @property
    def cycles(self) -> int:
        return self._cycles

    @property
    def current(self) -> str:
        return _label(SCRIPT[self._index])

    def step(self, now: float, status: StatusTracker) -> None:
        """Advance one control tick; *now* is the loop's monotonic timestamp."""
        self._locomotion.tick()
        action = SCRIPT[self._index]
        kind = action["kind"]

        if not self._started:
            self._begin(action, now)
            self._started = True

        if kind == "turn":
            remaining = self._turn_remaining
            if abs(remaining) < 1e-6:
                self._advance()
                return
            applied = self._locomotion.turn(remaining)
            step = max(-self._cfg.max_turn_deg_per_tick,
                       min(remaining, self._cfg.max_turn_deg_per_tick))
            self._turn_remaining = remaining - step
            if not applied:
                self._turn_remaining = 0.0
            status.update(action=_label(action), goal="AUTOPILOT (phase 4)")
            if abs(self._turn_remaining) < 1e-6:
                self._advance()
        elif kind == "move":
            status.update(action=_label(action), goal="AUTOPILOT (phase 4)")
            if not self._locomotion.active:
                self._advance()
        elif kind == "wait":
            status.update(action=_label(action), goal="AUTOPILOT (phase 4)")
            if now >= self._wait_until:
                self._advance()
        else:  # tap
            status.update(action=_label(action), goal="AUTOPILOT (phase 4)")
            self._advance()

    def _begin(self, action: Action, now: float) -> None:
        kind = action["kind"]
        if kind == "turn":
            self._turn_remaining = float(action["amount"])
        elif kind == "move":
            self._locomotion.move(
                str(action["direction"]),
                float(action["duration"]),
                sprint=bool(action.get("sprint", False)),
            )
        elif kind == "wait":
            self._wait_until = now + float(action["duration"])
        elif kind == "tap":
            self._locomotion.tap(str(action["action"]))
        log.debug("autopilot action: %s", _label(action))

    def _advance(self) -> None:
        self._index += 1
        self._started = False
        if self._index >= len(SCRIPT):
            self._index = 0
            self._cycles += 1
            log.info("autopilot script cycle %d complete", self._cycles)


def _label(action: Action) -> str:
    kind = action["kind"]
    if kind == "turn":
        return f"turn:{float(action['amount']):+.0f}deg"
    if kind == "move":
        suffix = " sprint" if action.get("sprint") else ""
        return f"move:{action['direction']} {float(action['duration']):.1f}s{suffix}"
    if kind == "wait":
        return f"wait:{float(action['duration']):.1f}s"
    return f"tap:{action['action']}"
