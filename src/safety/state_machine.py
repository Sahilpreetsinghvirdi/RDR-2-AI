"""Agent lifecycle state machine shared by every subsystem."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import StrEnum

log = logging.getLogger(__name__)

TransitionCallback = Callable[[str, str, str], None]


class AgentState(StrEnum):
    INIT = "INIT"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    TAKEOVER = "TAKEOVER"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


ALLOWED_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.INIT: {AgentState.RUNNING, AgentState.STOPPED, AgentState.ERROR},
    AgentState.RUNNING: {
        AgentState.PAUSED,
        AgentState.TAKEOVER,
        AgentState.STOPPED,
        AgentState.ERROR,
    },
    AgentState.PAUSED: {
        AgentState.RUNNING,
        AgentState.TAKEOVER,
        AgentState.STOPPED,
        AgentState.ERROR,
    },
    AgentState.TAKEOVER: {
        AgentState.RUNNING,
        AgentState.PAUSED,
        AgentState.STOPPED,
        AgentState.ERROR,
    },
    AgentState.STOPPED: set(),
    AgentState.ERROR: set(),
}

TERMINAL_STATES = {AgentState.STOPPED, AgentState.ERROR}


class StateMachine:
    """Thread-safe agent state with validated transitions and a stop event."""

    def __init__(
        self,
        initial: AgentState = AgentState.INIT,
        on_transition: TransitionCallback | None = None,
    ) -> None:
        self._state = initial
        self._lock = threading.RLock()
        self._reason = "init"
        self._changed_at = time.monotonic()
        self._on_transition = on_transition
        self.stop_event = threading.Event()
        if initial in TERMINAL_STATES:
            self.stop_event.set()

    @property
    def state(self) -> AgentState:
        with self._lock:
            return self._state

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    @property
    def active(self) -> bool:
        return self.state is AgentState.RUNNING

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def seconds_in_state(self) -> float:
        with self._lock:
            return time.monotonic() - self._changed_at

    def request(self, new_state: AgentState | str, reason: str = "") -> bool:
        """Attempt a transition; returns False when it is not allowed."""
        if isinstance(new_state, str):
            try:
                new_state = AgentState(new_state.upper())
            except ValueError:
                log.error("unknown state requested: %s", new_state)
                return False
        with self._lock:
            old = self._state
            if new_state is old:
                return True
            if new_state not in ALLOWED_TRANSITIONS[old]:
                log.warning(
                    "transition refused: %s -> %s (%s)", old.value, new_state.value, reason
                )
                return False
            self._state = new_state
            self._reason = reason or new_state.value.lower()
            self._changed_at = time.monotonic()
            if new_state in TERMINAL_STATES:
                self.stop_event.set()
            log.info("state %s -> %s (%s)", old.value, new_state.value, self._reason)
            callback = self._on_transition
        if callback is not None:
            try:
                callback(old.value, new_state.value, self._reason)
            except Exception:
                log.exception("state transition callback failed")
        return True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self._state.value,
                "reason": self._reason,
                "seconds_in_state": round(time.monotonic() - self._changed_at, 2),
            }
