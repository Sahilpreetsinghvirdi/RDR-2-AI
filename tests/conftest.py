"""Shared fixtures: fake input backends and a loaded configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AppConfig, load_config
from src.input.input_controller import InputController

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class FakeKeyboard:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.pressed: set[str] = set()
        self.fail = False

    def down(self, key) -> bool:
        if self.fail:
            return False
        self.events.append(("down", key.name))
        self.pressed.add(key.name)
        return True

    def up(self, key) -> bool:
        if self.fail:
            return False
        self.events.append(("up", key.name))
        self.pressed.discard(key.name)
        return True

    def is_down(self, key) -> bool:
        return key.name in self.pressed


class FakeMouse:
    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.down_buttons: set[str] = set()

    def move(self, dx: int, dy: int) -> bool:
        self.events.append(("move", dx, dy))
        return True

    def button_down(self, button: str) -> bool:
        self.events.append(("down", button))
        self.down_buttons.add(button)
        return True

    def button_up(self, button: str) -> bool:
        self.events.append(("up", button))
        self.down_buttons.discard(button)
        return True

    def wheel(self, delta: int) -> bool:
        self.events.append(("wheel", delta))
        return True

    def button_is_down(self, button: str) -> bool:
        return button in self.down_buttons


@pytest.fixture
def app_config() -> AppConfig:
    return load_config(CONFIG_PATH)


@pytest.fixture
def fake_keyboard() -> FakeKeyboard:
    return FakeKeyboard()


@pytest.fixture
def fake_mouse() -> FakeMouse:
    return FakeMouse()


@pytest.fixture
def allow_true():
    return lambda: True


@pytest.fixture
def allow_false():
    return lambda: False


def make_controller(
    app_config: AppConfig,
    keyboard: FakeKeyboard,
    mouse: FakeMouse,
    allow=lambda: True,
    stop_event=None,
) -> InputController:
    return InputController(
        app_config.input,
        allow_input=allow,
        keyboard=keyboard,
        mouse=mouse,
        stop_event=stop_event,
    )
