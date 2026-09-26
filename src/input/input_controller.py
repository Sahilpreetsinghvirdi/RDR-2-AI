"""Single gateway for every automated input the agent sends to the game."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from src.config import InputConfig
from src.input.controller import ControllerBackend, UnavailableController
from src.input.keyboard import KeyboardBackend, SendInputKeyboard
from src.input.keys import Key, normalize_key
from src.input.mouse import MouseBackend, SendInputMouse, normalize_button

log = logging.getLogger(__name__)


@dataclass
class InputStats:
    sent: int = 0
    refused: int = 0
    failed: int = 0
    released: int = 0
    last_latency_ms: float = 0.0
    last_error: str = ""
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "sent": self.sent,
            "refused": self.refused,
            "failed": self.failed,
            "last_latency_ms": round(self.last_latency_ms, 2),
            "held": [],
        }


class InputController:
    """Serialize, guard and track all keyboard/mouse/controller activity.

    *allow_input* is consulted before every send (agent state + window focus);
    *release_all* deliberately bypasses that guard so safety paths always work.
    """

    def __init__(
        self,
        cfg: InputConfig,
        allow_input: Callable[[], bool],
        keyboard: KeyboardBackend | None = None,
        mouse: MouseBackend | None = None,
        controller: ControllerBackend | None = None,
        stop_event: threading.Event | None = None,
        history_limit: int = 64,
    ) -> None:
        self._cfg = cfg
        self._allow = allow_input
        self._keyboard = keyboard or SendInputKeyboard(
            mode=cfg.keyboard_mode,
            verify=cfg.verify_injection,
            verify_timeout_ms=cfg.verify_timeout_ms,
        )
        self._mouse = mouse or SendInputMouse(max_delta=cfg.mouse_max_delta)
        self._controller = controller or UnavailableController()
        self._stop = stop_event
        self._lock = threading.RLock()
        self._held_keys: dict[str, Key] = {}
        self._held_buttons: dict[str, None] = {}
        self.stats = InputStats()
        self._history_limit = history_limit
        self._last_action = ""

    @property
    def held_keys(self) -> list[str]:
        with self._lock:
            return sorted(self._held_keys)

    @property
    def held_buttons(self) -> list[str]:
        with self._lock:
            return sorted(self._held_buttons)

    @property
    def last_action(self) -> str:
        return self._last_action

    def _record(self, entry: str) -> None:
        self._last_action = entry
        hist = self.stats.history
        hist.append(entry)
        if len(hist) > self._history_limit:
            del hist[: len(hist) - self._history_limit]

    def _prepare(self) -> bool:
        if self._stop is not None and self._stop.is_set():
            self.stats.refused += 1
            return False
        try:
            allowed = self._allow()
        except Exception:
            log.exception("input guard raised")
            allowed = False
        if not allowed:
            self.stats.refused += 1
            log.debug("input refused by guard")
            return False
        return True

    def _finish(self, ok: bool, latency_s: float, entry: str) -> bool:
        if ok:
            self.stats.sent += 1
            self.stats.last_latency_ms = latency_s * 1000.0
            self._record(entry)
        else:
            self.stats.failed += 1
            self.stats.last_error = entry
        return ok

    def key_down(self, key: str | int | Key) -> bool:
        if not self._prepare():
            return False
        resolved = normalize_key(key)
        with self._lock:
            t0 = time.perf_counter()
            ok = self._keyboard.down(resolved)
            latency = time.perf_counter() - t0
            if ok:
                self._held_keys[resolved.name] = resolved
        return self._finish(ok, latency, f"key_down:{resolved.name}")

    def key_up(self, key: str | int | Key) -> bool:
        resolved = normalize_key(key)
        with self._lock:
            t0 = time.perf_counter()
            ok = self._keyboard.up(resolved)
            latency = time.perf_counter() - t0
            self._held_keys.pop(resolved.name, None)
        if ok:
            self.stats.sent += 1
            self.stats.last_latency_ms = latency * 1000.0
            self._record(f"key_up:{resolved.name}")
            return True
        self.stats.failed += 1
        return False

    def press(self, key: str | int | Key, hold_ms: int | None = None) -> bool:
        duration = self._cfg.default_hold_ms if hold_ms is None else max(0, int(hold_ms))
        if not self.key_down(key):
            return False
        try:
            return self._interruptible_sleep(duration / 1000.0)
        finally:
            self.key_up(key)

    def hold(self, key: str | int | Key, duration_s: float) -> bool:
        """Hold a key for *duration_s* seconds; aborts early on pause/stop."""
        if not self.key_down(key):
            return False
        try:
            return self._interruptible_sleep(duration_s)
        finally:
            self.key_up(key)

    def _interruptible_sleep(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        slice_s = max(0.005, self._cfg.hold_check_interval_s)
        while True:
            now = time.monotonic()
            if now >= deadline:
                return True
            if self._stop is not None and self._stop.is_set():
                return False
            if not self._allow():
                return False
            time.sleep(min(slice_s, deadline - now))

    def mouse_move(self, dx: int, dy: int) -> bool:
        if not self._prepare():
            return False
        with self._lock:
            t0 = time.perf_counter()
            ok = self._mouse.move(int(dx), int(dy))
            latency = time.perf_counter() - t0
        return self._finish(ok, latency, f"mouse_move:{int(dx)},{int(dy)}")

    def mouse_down(self, button: str = "left") -> bool:
        if not self._prepare():
            return False
        name = normalize_button(button)
        with self._lock:
            t0 = time.perf_counter()
            ok = self._mouse.button_down(name)
            latency = time.perf_counter() - t0
            if ok:
                self._held_buttons[name] = None
        return self._finish(ok, latency, f"mouse_down:{name}")

    def mouse_up(self, button: str = "left") -> bool:
        name = normalize_button(button)
        with self._lock:
            t0 = time.perf_counter()
            ok = self._mouse.button_up(name)
            latency = time.perf_counter() - t0
            self._held_buttons.pop(name, None)
        if ok:
            self.stats.sent += 1
            self.stats.last_latency_ms = latency * 1000.0
            self._record(f"mouse_up:{name}")
            return True
        self.stats.failed += 1
        return False

    def mouse_click(self, button: str = "left", hold_ms: int | None = None) -> bool:
        duration = self._cfg.click_hold_ms if hold_ms is None else max(0, int(hold_ms))
        if not self.mouse_down(button):
            return False
        try:
            return self._interruptible_sleep(duration / 1000.0)
        finally:
            self.mouse_up(button)

    def mouse_wheel(self, delta: int) -> bool:
        if not self._prepare():
            return False
        with self._lock:
            t0 = time.perf_counter()
            ok = self._mouse.wheel(int(delta))
            latency = time.perf_counter() - t0
        return self._finish(ok, latency, f"mouse_wheel:{int(delta)}")

    def controller_button_down(self, button: str) -> bool:
        if not self._prepare():
            return False
        return bool(self._controller.button_down(button))

    def controller_button_up(self, button: str) -> bool:
        return bool(self._controller.button_up(button))

    def controller_axis(self, x: float, y: float) -> bool:
        if not self._prepare():
            return False
        return bool(self._controller.axis(float(x), float(y)))

    def release_all(self) -> None:
        """Release everything the AI is holding. Never blocked by the guard."""
        if not self._lock.acquire(timeout=0.5):
            log.error("release_all could not acquire the input lock")
            return
        try:
            released: list[str] = []
            for name, key in list(self._held_keys.items()):
                try:
                    if self._keyboard.up(key):
                        released.append(name)
                except Exception:
                    log.exception("failed to release key '%s'", name)
                self._held_keys.pop(name, None)
            for name in list(self._held_buttons):
                try:
                    if self._mouse.button_up(name):
                        released.append(name)
                except Exception:
                    log.exception("failed to release mouse button '%s'", name)
                self._held_buttons.pop(name, None)
            if released:
                self.stats.released += len(released)
                self._record("release_all:" + ",".join(released))
                log.warning("released held inputs: %s", ", ".join(released))
        finally:
            self._lock.release()

    def snapshot(self) -> dict[str, object]:
        return {
            "held_keys": self.held_keys,
            "held_buttons": self.held_buttons,
            "sent": self.stats.sent,
            "refused": self.stats.refused,
            "failed": self.stats.failed,
            "last_latency_ms": round(self.stats.last_latency_ms, 2),
            "last_action": self._last_action,
        }
