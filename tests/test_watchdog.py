"""Watchdog stall detection and safe-state enforcement."""

from __future__ import annotations

import threading
import time

from src.config import SafetyConfig
from src.safety.state_machine import AgentState, StateMachine
from src.safety.watchdog import HeartbeatRegistry, Watchdog


def fast_cfg(**kwargs) -> SafetyConfig:
    base = dict(
        watchdog_interval_s=0.02,
        heartbeat_stall_s=0.15,
        startup_grace_s=0.0,
    )
    base.update(kwargs)
    return SafetyConfig(**base)


def test_heartbeat_registry_tracks_age() -> None:
    hb = HeartbeatRegistry()
    assert hb.age("main") is None
    hb.beat("main")
    age = hb.age("main")
    assert age is not None and age >= 0.0
    time.sleep(0.05)
    assert hb.age("main") is not None and hb.age("main") >= 0.04  # type: ignore[operator]
    assert hb.names() == ["main"]


def test_fresh_heartbeat_does_not_fault() -> None:
    machine = StateMachine(AgentState.RUNNING)
    hb = HeartbeatRegistry()
    released: list[int] = []
    wd = Watchdog(fast_cfg(), machine, hb, lambda: released.append(1))
    wd.start()
    end = time.monotonic() + 0.3
    while time.monotonic() < end:
        hb.beat("main")
        time.sleep(0.02)
    wd.stop()
    assert machine.state is AgentState.RUNNING
    assert released == []
    assert wd.faults == []


def test_stalled_heartbeat_faults_and_releases() -> None:
    machine = StateMachine(AgentState.RUNNING)
    hb = HeartbeatRegistry()
    hb.beat("main")
    released: list[int] = []
    faults: list[str] = []
    wd = Watchdog(
        fast_cfg(), machine, hb, lambda: released.append(1),
        on_fault=faults.append,
    )
    wd.start()
    end = time.monotonic() + 2.0
    while time.monotonic() < end and not wd.faults:
        time.sleep(0.05)
    wd.stop()
    assert machine.state is AgentState.ERROR
    assert machine.stop_event.is_set()
    assert released == [1]
    assert wd.faults
    assert "main" in wd.faults[0]
    assert faults == wd.faults


def test_dead_registered_thread_faults() -> None:
    machine = StateMachine(AgentState.RUNNING)
    hb = HeartbeatRegistry()
    released: list[int] = []

    def body() -> None:
        return

    thread = threading.Thread(target=body, name="victim")
    thread.start()
    thread.join()
    wd = Watchdog(
        fast_cfg(), machine, hb, lambda: released.append(1),
        threads={"victim": thread},
    )
    wd.start()
    end = time.monotonic() + 2.0
    while time.monotonic() < end and not wd.faults:
        time.sleep(0.05)
    wd.stop()
    assert machine.state is AgentState.ERROR
    assert "victim" in wd.faults[0]


def test_grace_period_prevents_early_fault() -> None:
    machine = StateMachine(AgentState.RUNNING)
    hb = HeartbeatRegistry()
    wd = Watchdog(fast_cfg(startup_grace_s=0.5), machine, hb, lambda: None)
    wd.start()
    time.sleep(0.25)
    assert machine.state is AgentState.RUNNING
    assert wd.faults == []
    wd.stop()


def test_no_fault_when_not_running() -> None:
    machine = StateMachine(AgentState.PAUSED)
    hb = HeartbeatRegistry()
    wd = Watchdog(fast_cfg(), machine, hb, lambda: None)
    wd.start()
    time.sleep(0.3)
    wd.stop()
    assert machine.state is AgentState.PAUSED
    assert wd.faults == []


def test_stopped_watchdog_ignores_faults() -> None:
    machine = StateMachine(AgentState.RUNNING)
    hb = HeartbeatRegistry()
    wd = Watchdog(fast_cfg(), machine, hb, lambda: None)
    wd.start()
    wd.stop()
    time.sleep(0.3)
    assert machine.state is AgentState.RUNNING
    assert not wd.is_alive()
