"""Shared Win32 SendInput structures and helpers (keyboard + mouse)."""

from __future__ import annotations

import ctypes
import time
from typing import Final

from src.input.keys import Key

INPUT_MOUSE: Final = 0
INPUT_KEYBOARD: Final = 1
INPUT_HARDWARE: Final = 2

KEYEVENTF_EXTENDEDKEY: Final = 0x0001
KEYEVENTF_KEYUP: Final = 0x0002
KEYEVENTF_SCANCODE: Final = 0x0008

MOUSEEVENTF_MOVE: Final = 0x0001
MOUSEEVENTF_LEFTDOWN: Final = 0x0002
MOUSEEVENTF_LEFTUP: Final = 0x0004
MOUSEEVENTF_RIGHTDOWN: Final = 0x0008
MOUSEEVENTF_RIGHTUP: Final = 0x0010
MOUSEEVENTF_MIDDLEDOWN: Final = 0x0020
MOUSEEVENTF_MIDDLEUP: Final = 0x0040
MOUSEEVENTF_XDOWN: Final = 0x0080
MOUSEEVENTF_XUP: Final = 0x0100
MOUSEEVENTF_WHEEL: Final = 0x0800
MOUSEEVENTF_MOVE_NOCOALESCE: Final = 0x2000

XBUTTON1: Final = 0x0001
XBUTTON2: Final = 0x0002
WHEEL_DELTA: Final = 120

VK_LBUTTON: Final = 0x01
VK_RBUTTON: Final = 0x02
VK_MBUTTON: Final = 0x04
VK_XBUTTON1: Final = 0x05
VK_XBUTTON2: Final = 0x06

EXTRA_INFO: Final = 0x52444149


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort),
                ("wParamH", ctypes.c_ushort)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


user32 = ctypes.windll.user32
user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = ctypes.c_uint
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.MapVirtualKeyW.argtypes = [ctypes.c_uint, ctypes.c_uint]
user32.MapVirtualKeyW.restype = ctypes.c_uint

MAX_DELTA: Final = 32767


def map_scancode(vk: int) -> int:
    return int(user32.MapVirtualKeyW(vk, 0))


def send_inputs(inputs: list[INPUT]) -> int:
    if not inputs:
        return 0
    array = (INPUT * len(inputs))(*inputs)
    return int(user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT)))


def make_key_input(key: Key, key_up: bool, scancode_mode: bool) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    flags = 0
    if key.extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    if scancode_mode:
        flags |= KEYEVENTF_SCANCODE
        inp.ki.wVk = 0
        inp.ki.wScan = map_scancode(key.vk)
    else:
        inp.ki.wVk = key.vk
        inp.ki.wScan = 0
    if key_up:
        flags |= KEYEVENTF_KEYUP
    inp.ki.dwFlags = flags
    inp.ki.dwExtraInfo = EXTRA_INFO
    return inp


def make_move_input(dx: int, dy: int) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dx = int(dx)
    inp.mi.dy = int(dy)
    inp.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_MOVE_NOCOALESCE
    inp.mi.dwExtraInfo = EXTRA_INFO
    return inp


def make_button_input(flags: int, data: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.mouseData = data
    inp.mi.dwFlags = flags
    inp.mi.dwExtraInfo = EXTRA_INFO
    return inp


def key_is_down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def wait_for_key_state(vk: int, down: bool, timeout_s: float, poll_s: float = 0.002) -> float:
    """Poll until the key state matches *down*; returns elapsed seconds or -1 on timeout."""
    deadline = time.perf_counter() + timeout_s
    t0 = time.perf_counter()
    while True:
        if key_is_down(vk) == down:
            return time.perf_counter() - t0
        if time.perf_counter() >= deadline:
            return -1.0
        time.sleep(poll_s)
