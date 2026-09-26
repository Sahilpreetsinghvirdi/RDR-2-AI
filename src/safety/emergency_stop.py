"""Global hotkeys for emergency stop, pause and human takeover.

The listener is a standalone high-priority thread polling
``GetAsyncKeyState`` so it keeps working even if the planner, model or main
loop is frozen.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections.abc import Callable

from src.config import SafetyConfig
from src.input.keys import normalize_key
from src.input.win32_input import user32
from src.safety.state_machine import AgentState, StateMachine

log = logging.getLogger(__name__)

THREAD_PRIORITY_HIGHEST = 2


class EmergencyStop:
    """Owns the hotkey thread and the emergency/pause/takeover reactions."""

    def __init__(
        self,
        cfg: SafetyConfig,
        machine: StateMachine,
        release_inputs: Callable[[], None],
        on_pause_change: Callable[[bool], None] | None = None,
        on_takeover_change: Callable[[bool], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._machine = machine
        self._release = release_inputs
        self._on_pause = on_pause_change
        self._on_takeover = on_takeover_change
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bindings, self._vks = self._resolve_keys(cfg)
        self.trigger_count = 0
        self.last_trigger = ""

    @staticmethod
    def _resolve_keys(cfg: SafetyConfig) -> tuple[dict[str, str], dict[str, int]]:
        specs = (
            ("emergency", cfg.emergency_key or "F12"),
            ("pause", cfg.pause_key or "F11"),
            ("takeover", cfg.takeover_key or "F10"),
        )
        bindings: dict[str, str] = {}
        vks: dict[str, int] = {}
        resolved: list[int] = []
        for role, raw in specs:
            key = normalize_key(raw)
            bindings[key.name] = role
            vks[key.name] = key.vk
            resolved.append(key.vk)
        if len(set(resolved)) != len(resolved):
            raise ValueError(
                "safety hotkeys must be distinct: "
                f"{cfg.emergency_key}, {cfg.pause_key}, {cfg.takeover_key}"
            )
        return bindings, vks

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.active:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="hotkeys", daemon=True
        )
        self._thread.start()
        log.info(
            "hotkeys armed: %s=stop %s=pause %s=takeover",
            self._cfg.emergency_key,
            self._cfg.pause_key,
            self._cfg.takeover_key,
        )

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(1.0)
        self._thread = None

    def _run(self) -> None:
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITY_HIGHEST)
        except Exception:
            log.debug("could not raise hotkey thread priority", exc_info=True)
        previous = {vk: False for vk in self._vks.values()}
        interval = 1.0 / max(1, self._cfg.hotkey_poll_hz)
        while not self._stop.is_set():
            for name, vk in self._vks.items():
                try:
                    down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
                except Exception:
                    log.exception("hotkey poll failed")
                    break
                if down and not previous[vk]:
                    previous[vk] = True
                    self._dispatch(name)
                elif not down:
                    previous[vk] = False
            self._stop.wait(interval)

    def _dispatch(self, key_name: str) -> None:
        role = self._bindings.get(key_name)
        if role is None:
            return
        try:
            if role == "emergency":
                self.trigger(f"manual:{self._cfg.emergency_key}")
            elif role == "pause":
                self.toggle_pause()
            else:
                self.toggle_takeover()
        except Exception:
            log.exception("hotkey handler failed for %s", key_name)

    def trigger(self, reason: str) -> None:
        """Immediate stop: refuse new decisions, release everything, halt the agent."""
        self.trigger_count += 1
        self.last_trigger = reason
        log.critical("EMERGENCY STOP (%s)", reason)
        try:
            self._release()
        except Exception:
            log.exception("release_all failed during emergency stop")
        self._machine.request(AgentState.STOPPED, reason)

    def toggle_pause(self) -> bool:
        machine = self._machine
        if machine.state is AgentState.RUNNING:
            if self._cfg.release_on_pause:
                self._safe_release()
            ok = machine.request(AgentState.PAUSED, "manual")
            if ok and self._on_pause is not None:
                self._on_pause(True)
            return ok
        if machine.state is AgentState.PAUSED:
            ok = machine.request(AgentState.RUNNING, "resume")
            if ok and self._on_pause is not None:
                self._on_pause(False)
            return ok
        log.info("pause ignored in state %s", machine.state.value)
        return False

    def toggle_takeover(self) -> bool:
        machine = self._machine
        if machine.state in (AgentState.RUNNING, AgentState.PAUSED):
            self._safe_release()
            ok = machine.request(AgentState.TAKEOVER, "human-takeover")
            if ok and self._on_takeover is not None:
                self._on_takeover(True)
            return ok
        if machine.state is AgentState.TAKEOVER:
            ok = machine.request(AgentState.RUNNING, "ai-resume")
            if ok and self._on_takeover is not None:
                self._on_takeover(False)
            return ok
        log.info("takeover ignored in state %s", machine.state.value)
        return False

    def _safe_release(self) -> None:
        if not self._cfg.release_on_pause:
            return
        try:
            self._release()
        except Exception:
            log.exception("release_all failed")

    def wait_until_stopped(self, poll_s: float = 0.1) -> None:
        while not self._stop.is_set() and not self._machine.stop_event.is_set():
            time.sleep(poll_s)
