"""Agent state machine transitions and stop-event semantics."""

from __future__ import annotations

from src.safety.state_machine import AgentState, StateMachine


def test_initial_state_and_transition() -> None:
    sm = StateMachine()
    assert sm.state is AgentState.INIT
    assert sm.request(AgentState.RUNNING, "start") is True
    assert sm.state is AgentState.RUNNING
    assert sm.reason == "start"
    assert sm.stop_event.is_set() is False


def test_invalid_transition_rejected() -> None:
    sm = StateMachine(AgentState.STOPPED)
    assert sm.request(AgentState.RUNNING) is False
    assert sm.state is AgentState.STOPPED
    assert sm.stop_event.is_set() is True


def test_pause_resume_cycle() -> None:
    sm = StateMachine(AgentState.RUNNING)
    assert sm.request(AgentState.PAUSED, "manual") is True
    assert sm.request(AgentState.RUNNING, "resume") is True
    assert sm.active is True


def test_error_is_terminal_and_sets_stop_event() -> None:
    sm = StateMachine(AgentState.RUNNING)
    assert sm.request(AgentState.ERROR, "fault") is True
    assert sm.stop_event.is_set() is True
    assert sm.terminal is True
    assert sm.request(AgentState.RUNNING) is False


def test_same_state_request_is_noop() -> None:
    sm = StateMachine(AgentState.RUNNING)
    assert sm.request(AgentState.RUNNING) is True
    assert sm.state is AgentState.RUNNING


def test_unknown_state_string_rejected() -> None:
    sm = StateMachine(AgentState.RUNNING)
    assert sm.request("warp-speed") is False
    assert sm.state is AgentState.RUNNING


def test_state_string_accepted_case_insensitively() -> None:
    sm = StateMachine(AgentState.INIT)
    assert sm.request("running") is True
    assert sm.state is AgentState.RUNNING


def test_callback_invoked_with_context() -> None:
    seen: list[tuple[str, str, str]] = []
    sm = StateMachine(
        AgentState.INIT, on_transition=lambda o, n, r: seen.append((o, n, r))
    )
    sm.request(AgentState.RUNNING, "go")
    assert seen == [("INIT", "RUNNING", "go")]


def test_callback_exception_does_not_break_transition() -> None:
    def boom(old: str, new: str, reason: str) -> None:
        raise RuntimeError("callback failed")

    sm = StateMachine(AgentState.INIT, on_transition=boom)
    assert sm.request(AgentState.RUNNING, "go") is True
    assert sm.state is AgentState.RUNNING


def test_snapshot_fields() -> None:
    sm = StateMachine(AgentState.RUNNING)
    snap = sm.snapshot()
    assert snap["state"] == "RUNNING"
    assert "seconds_in_state" in snap
    assert sm.seconds_in_state() >= 0.0
