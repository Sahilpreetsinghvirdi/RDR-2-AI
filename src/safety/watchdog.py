"""Watchdog: detects stalled loops and forces the agent into a safe state."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from src.config import SafetyConfig
from src.safety.state_machine import AgentState, StateMachine

log = logging.getLogger(__name__)


class HeartbeatRegistry:
    """Named loop liveness timestamps, updated by the loops themselves."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._beats: dict[str, float] = {}

    def beat(self, name: str) -> None:
        with self._lock:
            self._beats[name] = time.monotonic()

    def age(self, name: str) -> float | None:
        with self._lock:
            ts = self._beats.get(name)
        if ts is None:
            return None
        return time.monotonic() - ts

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._beats)


class Watchdog(threading.Thread):
    """Faults (ERROR + release inputs) when a critical loop stops beating."""

    def __init__(
        self,
        cfg: SafetyConfig,
        machine: StateMachine,
        heartbeats: HeartbeatRegistry,
        release_inputs: Callable[[], None],
        threads: dict[str, threading.Thread | None] | None = None,
        on_fault: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(name="watchdog", daemon=True)
        self._cfg = cfg
        self._machine = machine
        self._heartbeats = heartbeats
        self._release = release_inputs
        self._threads: dict[str, threading.Thread | None] = dict(threads or {})
        self._on_fault = on_fault
        self._shutdown = threading.Event()
        self._armed_at = time.monotonic()
        self.faults: list[str] = []

    def register_thread(self, name: str, thread: threading.Thread | None) -> None:
        self._threads[name] = thread

    def stop(self) -> None:
        self._shutdown.set()
        if self.is_alive():
            self.join(2.0)

    def _in_grace(self) -> bool:
        return (time.monotonic() - self._armed_at) < self._cfg.startup_grace_s

    def _fault(self, message: str) -> None:
        log.critical("WATCHDOG FAULT: %s", message)
        self.faults.append(message)
        try:
            self._release()
        except Exception:
            log.exception("release_all failed during watchdog fault")
        self._machine.request(AgentState.ERROR, f"watchdog: {message}")
        if self._on_fault is not None:
            try:
                self._on_fault(message)
            except Exception:
                log.exception("watchdog fault callback failed")

    def run(self) -> None:
        log.info(
            "watchdog armed (interval=%.2fs stall=%.2fs grace=%.1fs)",
            self._cfg.watchdog_interval_s,
            self._cfg.heartbeat_stall_s,
            self._cfg.startup_grace_s,
        )
        faulted = False
        while not self._shutdown.is_set():
            if self._shutdown.wait(self._cfg.watchdog_interval_s):
                break
            if faulted or self._machine.terminal:
                continue
            if self._machine.state is not AgentState.RUNNING:
                continue
            if self._in_grace():
                continue
            for name, thread in self._threads.items():
                if thread is None:
                    continue
                if thread is threading.current_thread() or thread.is_alive():
                    continue
                self._fault(f"thread '{name}' is not alive")
                faulted = True
                break
            if faulted:
                continue
            for name in self._heartbeats.names():
                age = self._heartbeats.age(name)
                if age is not None and age > self._cfg.heartbeat_stall_s:
                    self._fault(f"loop '{name}' stalled for {age:.1f}s")
                    faulted = True
                    break
        log.info("watchdog stopped")
