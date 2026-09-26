"""Emergency stop, pause and human-takeover behaviour."""

from __future__ import annotations

import pytest

from src.config import SafetyConfig
from src.input.keys import normalize_key
from src.safety.emergency_stop import EmergencyStop
from src.safety.state_machine import AgentState, StateMachine


def make_stack(release_on_pause: bool = True):
    machine = StateMachine(AgentState.RUNNING)
    released: list[int] = []
    modes: list[tuple[str, bool]] = []
    stop = EmergencyStop(
        SafetyConfig(release_on_pause=release_on_pause),
        machine,
        release_inputs=lambda: released.append(1),
        on_pause_change=lambda active: modes.append(("pause", active)),
        on_takeover_change=lambda active: modes.append(("takeover", active)),
    )
    return machine, released, modes, stop


def test_emergency_stop_releases_and_halts() -> None:
    machine, released, _, stop = make_stack()
    stop.trigger("test")
    assert released == [1]
    assert machine.state is AgentState.STOPPED
    assert machine.stop_event.is_set()
    assert stop.trigger_count == 1
    assert stop.last_trigger == "test"


def test_emergency_stop_works_from_any_pre_state() -> None:
    def attempt(state: AgentState) -> None:
        machine = StateMachine(state)
        released: list[int] = []
        stop = EmergencyStop(SafetyConfig(), machine, lambda: released.append(1))
        stop.trigger("x")
        assert machine.state is AgentState.STOPPED
        assert released == [1]

    for state in (AgentState.INIT, AgentState.RUNNING, AgentState.PAUSED,
                  AgentState.TAKEOVER):
        attempt(state)


def test_pause_toggles_and_releases() -> None:
    machine, released, modes, stop = make_stack()
    assert stop.toggle_pause() is True
    assert machine.state is AgentState.PAUSED
    assert released == [1]
    assert modes == [("pause", True)]
    assert stop.toggle_pause() is True
    assert machine.state is AgentState.RUNNING
    assert modes[-1] == ("pause", False)


def test_pause_without_release_config() -> None:
    machine, released, _, stop = make_stack(release_on_pause=False)
    stop.toggle_pause()
    assert machine.state is AgentState.PAUSED
    assert released == []


def test_takeover_toggles_and_releases() -> None:
    machine, released, modes, stop = make_stack()
    assert stop.toggle_takeover() is True
    assert machine.state is AgentState.TAKEOVER
    assert released == [1]
    assert stop.toggle_takeover() is True
    assert machine.state is AgentState.RUNNING
    assert ("takeover", False) in modes


def test_takeover_from_paused_allowed() -> None:
    machine, _, _, stop = make_stack()
    stop.toggle_pause()
    assert stop.toggle_takeover() is True
    assert machine.state is AgentState.TAKEOVER


def test_pause_ignored_when_stopped() -> None:
    machine, _, _, stop = make_stack()
    stop.trigger("x")
    assert stop.toggle_pause() is False
    assert machine.state is AgentState.STOPPED


def test_duplicate_hotkeys_rejected() -> None:
    machine = StateMachine()
    with pytest.raises(ValueError):
        EmergencyStop(
            SafetyConfig(emergency_key="F12", pause_key="F12", takeover_key="F10"),
            machine,
            lambda: None,
        )


def test_binding_names_resolve() -> None:
    cfg = SafetyConfig()
    machine, _, _, stop = make_stack()
    assert set(stop._bindings.values()) == {"emergency", "pause", "takeover"}
    assert stop._vks["f12"] == normalize_key(cfg.emergency_key).vk


def test_dispatch_routes_by_binding() -> None:
    machine, released, modes, stop = make_stack()
    stop._dispatch("f11")
    assert machine.state is AgentState.PAUSED
    stop._dispatch("f10")
    assert machine.state is AgentState.TAKEOVER
    stop._dispatch("f12")
    assert machine.state is AgentState.STOPPED
    assert released == [1, 1, 1]


def test_dispatch_unknown_binding_ignored() -> None:
    _, _, _, stop = make_stack()
    stop._dispatch("a")


def test_hotkey_thread_starts_and_stops() -> None:
    machine = StateMachine(AgentState.RUNNING)
    stop = EmergencyStop(SafetyConfig(hotkey_poll_hz=100), machine, lambda: None)
    stop.start()
    assert stop.active is True
    stop.stop()
    assert stop.active is False
