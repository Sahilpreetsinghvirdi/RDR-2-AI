"""Gamepad (XInput) backend interface.

Phase 1 ships the interface with a safe unimplemented backend so callers fail
loudly and explicitly instead of pretending a controller is connected.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ControllerUnavailable(Exception):
    """Raised when gamepad input is requested but no backend is enabled."""


@runtime_checkable
class ControllerBackend(Protocol):
    def button_down(self, button: str) -> bool: ...

    def button_up(self, button: str) -> bool: ...

    def axis(self, x: float, y: float) -> bool: ...

    def rumble(self, big: float, small: float) -> bool: ...


class UnavailableController:
    """Placeholder backend: every call fails with a clear message."""

    def button_down(self, button: str) -> bool:
        raise ControllerUnavailable(
            "gamepad backend not enabled (set input.controller.enabled: true "
            "- implemented in a later phase)"
        )

    def button_up(self, button: str) -> bool:
        raise ControllerUnavailable("gamepad backend not enabled")

    def axis(self, x: float, y: float) -> bool:
        raise ControllerUnavailable("gamepad backend not enabled")

    def rumble(self, big: float, small: float) -> bool:
        raise ControllerUnavailable("gamepad backend not enabled")
