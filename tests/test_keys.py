"""Key name normalization tests."""

from __future__ import annotations

import pytest

from src.input.keys import EXTENDED_VK, Key, KeyError_, key_display, normalize_key


def test_letters_map_to_vk() -> None:
    assert normalize_key("w").vk == 0x57
    assert normalize_key("W").vk == 0x57
    assert normalize_key("a").name == "a"


def test_digits_map_to_vk() -> None:
    assert normalize_key("5").vk == 0x35


def test_named_keys() -> None:
    assert normalize_key("space").vk == 0x20
    assert normalize_key("shift").vk == 0x10
    assert normalize_key("f12").vk == 0x7B
    assert normalize_key("f1").vk == 0x70


def test_aliases() -> None:
    assert normalize_key("ctrl").name == "control"
    assert normalize_key("esc").name == "escape"
    assert normalize_key("return").name == "enter"
    assert normalize_key(" ").name == "space"


def test_unknown_key_raises() -> None:
    with pytest.raises(KeyError_):
        normalize_key("hyper")
    with pytest.raises(KeyError_):
        normalize_key("")


def test_int_vk_round_trip() -> None:
    key = normalize_key(0x57)
    assert key.vk == 0x57
    assert normalize_key(0x70).name == "f1"
    with pytest.raises(KeyError_):
        normalize_key(0)


def test_key_object_passthrough() -> None:
    key = Key(name="w", vk=0x57)
    assert normalize_key(key) is key


def test_extended_flags() -> None:
    assert normalize_key("left").extended is True
    assert normalize_key("right").extended is True
    assert normalize_key("rcontrol").extended is True
    assert normalize_key("w").extended is False
    assert normalize_key("left").vk in EXTENDED_VK


def test_key_display() -> None:
    assert key_display("SHIFT") == "shift"
    assert key_display(0x57) == "w"
