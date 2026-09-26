"""Mouse backends built on Win32 SendInput."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.input.win32_input import (
    MAX_DELTA,
    MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MIDDLEDOWN,
    MOUSEEVENTF_MIDDLEUP,
    MOUSEEVENTF_RIGHTDOWN,
    MOUSEEVENTF_RIGHTUP,
    MOUSEEVENTF_WHEEL,
    MOUSEEVENTF_XDOWN,
    MOUSEEVENTF_XUP,
    VK_LBUTTON,
    VK_MBUTTON,
    VK_RBUTTON,
    VK_XBUTTON1,
    VK_XBUTTON2,
    XBUTTON1,
    XBUTTON2,
    key_is_down,
    make_button_input,
    make_move_input,
    send_inputs,
)

log = logging.getLogger(__name__)

BUTTONS: dict[str, tuple[int, int, int, int]] = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, VK_LBUTTON, 0),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, VK_RBUTTON, 0),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, VK_MBUTTON, 0),
    "x1": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, VK_XBUTTON1, XBUTTON1),
    "x2": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, VK_XBUTTON2, XBUTTON2),
}


def normalize_button(button: str) -> str:
    name = button.strip().lower()
    aliases = {"rmb": "right", "lmb": "left", "mmb": "middle"}
    name = aliases.get(name, name)
    if name not in BUTTONS:
        raise ValueError(f"unknown mouse button: {button!r}")
    return name


class MouseError(Exception):
    """Raised when a mouse event could not be injected."""


@runtime_checkable
class MouseBackend(Protocol):
    def move(self, dx: int, dy: int) -> bool: ...

    def button_down(self, button: str) -> bool: ...

    def button_up(self, button: str) -> bool: ...

    def wheel(self, delta: int) -> bool: ...

    def button_is_down(self, button: str) -> bool: ...


class SendInputMouse:
    """Relative mouse movement (camera look) and button control."""

    def __init__(self, max_delta: int = MAX_DELTA) -> None:
        self._max_delta = max(1, min(int(max_delta), MAX_DELTA))

    def move(self, dx: int, dy: int) -> bool:
        sent = 0
        total = 0
        remaining_x = int(dx)
        remaining_y = int(dy)
        while remaining_x != 0 or remaining_y != 0:
            step_x = max(-self._max_delta, min(self._max_delta, remaining_x))
            step_y = max(-self._max_delta, min(self._max_delta, remaining_y))
            sent += send_inputs([make_move_input(step_x, step_y)])
            total += 1
            remaining_x -= step_x
            remaining_y -= step_y
        if sent != total:
            log.error("mouse move injection failed (%d/%d)", sent, total)
            return False
        return True

    def button_down(self, button: str) -> bool:
        name = normalize_button(button)
        down_flags, _, vk, data = BUTTONS[name]
        if send_inputs([make_button_input(down_flags, data)]) != 1:
            log.error("mouse button down failed: %s", name)
            return False
        return True

    def button_up(self, button: str) -> bool:
        name = normalize_button(button)
        _, up_flags, vk, data = BUTTONS[name]
        if send_inputs([make_button_input(up_flags, data)]) != 1:
            log.error("mouse button up failed: %s", name)
            return False
        return True

    def wheel(self, delta: int) -> bool:
        return send_inputs([make_button_input(MOUSEEVENTF_WHEEL, int(delta))]) == 1

    def button_is_down(self, button: str) -> bool:
        name = normalize_button(button)
        _, _, vk, _ = BUTTONS[name]
        return key_is_down(vk)
