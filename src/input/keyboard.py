"""Keyboard backends built on Win32 SendInput."""

from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

from src.input.keys import Key
from src.input.win32_input import key_is_down, make_key_input, send_inputs, wait_for_key_state

log = logging.getLogger(__name__)


class KeyboardError(Exception):
    """Raised when a keystroke could not be injected."""


@runtime_checkable
class KeyboardBackend(Protocol):
    def down(self, key: Key) -> bool: ...

    def up(self, key: Key) -> bool: ...

    def is_down(self, key: Key) -> bool: ...


class SendInputKeyboard:
    """Injects key events; optionally confirms the system registered them."""

    def __init__(
        self,
        mode: str = "scancode",
        verify: bool = True,
        verify_timeout_ms: int = 50,
    ) -> None:
        if mode not in {"scancode", "virtual_key"}:
            raise ValueError(f"unknown keyboard mode: {mode!r}")
        self._scancode = mode == "scancode"
        self._verify = verify
        self._timeout_s = verify_timeout_ms / 1000.0
        self.last_verify_ms = 0.0

    def _inject(self, key: Key, key_up: bool) -> bool:
        sent = send_inputs([make_key_input(key, key_up, self._scancode)])
        if sent != 1:
            log.error("SendInput failed for key '%s' (sent=%d)", key.name, sent)
            return False
        if not self._verify:
            self.last_verify_ms = 0.0
            return True
        elapsed = wait_for_key_state(key.vk, not key_up, self._timeout_s)
        self.last_verify_ms = max(0.0, elapsed) * 1000.0
        if elapsed < 0:
            log.warning(
                "key '%s' %s not confirmed within %dms",
                key.name, "down" if not key_up else "up", int(self._timeout_s * 1000),
            )
            return False
        return True

    def down(self, key: Key) -> bool:
        t0 = time.perf_counter()
        ok = self._inject(key, key_up=False)
        self.last_verify_ms = max(self.last_verify_ms, (time.perf_counter() - t0) * 1000.0)
        return ok

    def up(self, key: Key) -> bool:
        t0 = time.perf_counter()
        ok = self._inject(key, key_up=True)
        self.last_verify_ms = max(self.last_verify_ms, (time.perf_counter() - t0) * 1000.0)
        return ok

    def is_down(self, key: Key) -> bool:
        return key_is_down(key.vk)
