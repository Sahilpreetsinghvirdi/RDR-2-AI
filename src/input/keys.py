"""Key name normalization and Win32 virtual-key mapping."""

from __future__ import annotations

from dataclasses import dataclass

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_ESCAPE = 0x1B
VK_SNAPSHOT = 0x2C

NAMED_KEYS: dict[str, int] = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "shift": VK_SHIFT,
    "lshift": 0xA0,
    "rshift": 0xA1,
    "control": VK_CONTROL,
    "ctrl": VK_CONTROL,
    "lcontrol": 0xA2,
    "rcontrol": 0xA3,
    "alt": VK_MENU,
    "menu": VK_MENU,
    "lalt": 0xA4,
    "ralt": 0xA5,
    "pause": 0x13,
    "capslock": 0x14,
    "escape": VK_ESCAPE,
    "esc": VK_ESCAPE,
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "printscreen": VK_SNAPSHOT,
    "print": VK_SNAPSHOT,
    "insert": 0x2D,
    "delete": 0x2E,
    "lwin": 0x5B,
    "rwin": 0x5C,
    "apps": 0x5D,
    "numlock": 0x90,
    "scrolllock": 0x91,
    "multiply": 0x6A,
    "add": 0x6B,
    "subtract": 0x6D,
    "decimal": 0x6E,
    "divide": 0x6F,
}

for _i in range(10):
    NAMED_KEYS[f"f{_i + 1}"] = 0x70 + _i
    NAMED_KEYS[f"num{_i}"] = 0x60 + _i
    NAMED_KEYS[f"numpad{_i}"] = 0x60 + _i
for _i in range(11, 25):
    NAMED_KEYS[f"f{_i}"] = 0x70 + (_i - 1)

EXTENDED_VK: frozenset[int] = frozenset(
    {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x5C, 0x5D,
     0x90, 0x6F, 0xA3, 0xA5}
)

ALIASES: dict[str, str] = {
    " ": "space",
    "spacebar": "space",
    "esc": "escape",
    "return": "enter",
    "ctrl": "control",
    "ctl": "control",
    "altgr": "ralt",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "pgup": "pageup",
    "pgdn": "pagedown",
}


class KeyError_(ValueError):
    """Unknown key name supplied to the input layer."""


@dataclass(frozen=True)
class Key:
    name: str
    vk: int

    @property
    def extended(self) -> bool:
        return self.vk in EXTENDED_VK


def normalize_key(key: str | int | Key) -> Key:
    """Resolve a user-facing key name (or VK) to a canonical :class:`Key`."""
    if isinstance(key, Key):
        return key
    if isinstance(key, int):
        if not 1 <= key <= 0xFE:
            raise KeyError_(f"virtual-key code out of range: {key}")
        if 0x41 <= key <= 0x5A:
            return Key(name=chr(key).lower(), vk=key)
        if 0x30 <= key <= 0x39:
            return Key(name=chr(key), vk=key)
        for name, vk in NAMED_KEYS.items():
            if vk == key:
                return Key(name=name, vk=vk)
        return Key(name=f"vk{key:02x}", vk=key)

    lowered = key.lower()
    lowered = ALIASES.get(lowered, lowered).strip()
    raw = ALIASES.get(lowered, lowered)
    if not raw:
        raise KeyError_("empty key name")

    if raw in NAMED_KEYS:
        return Key(name=raw, vk=NAMED_KEYS[raw])
    if len(raw) == 1 and raw.isalpha():
        return Key(name=raw, vk=ord(raw.upper()))
    if len(raw) == 1 and raw.isdigit():
        return Key(name=raw, vk=0x30 + int(raw))
    raise KeyError_(f"unknown key name: {key!r}")


def key_display(key: str | int | Key) -> str:
    return normalize_key(key).name
