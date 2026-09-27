"""Tests for the Phase 10 ComponentGuard fault isolation."""

from __future__ import annotations

from src.safety.resilience import ComponentGuard


def test_success_passes_through() -> None:
    guard = ComponentGuard()
    assert guard.call("x", lambda a, b: a + b, 1, 2) == 3
    assert guard.faults == 0
    assert guard.disabled == {}
    assert not guard.is_disabled("x")


def test_fault_disables_and_reports() -> None:
    seen: list[tuple[str, BaseException]] = []
    guard = ComponentGuard(on_fault=lambda name, exc: seen.append((name, exc)))

    def boom() -> None:
        raise ValueError("kaput")

    assert guard.call("planner", boom) is None
    assert guard.faults == 1
    assert guard.is_disabled("planner")
    assert "ValueError" in guard.summary()["planner"]
    assert "kaput" in guard.summary()["planner"]
    assert len(seen) == 1 and seen[0][0] == "planner"
    assert isinstance(seen[0][1], ValueError)


def test_disabled_component_not_called_again() -> None:
    calls: list[int] = []

    def flaky() -> int:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first call fails")
        return 7

    guard = ComponentGuard()
    assert guard.call("demo", flaky) is None
    assert guard.call("demo", flaky) is None
    assert calls == [1]
    assert guard.faults == 1


def test_other_components_unaffected() -> None:
    guard = ComponentGuard()

    def boom() -> None:
        raise OSError("dead")

    guard.call("bad", boom)
    assert guard.call("good", lambda: "ok") == "ok"
    assert set(guard.disabled) == {"bad"}


def test_fault_handler_exception_is_swallowed() -> None:
    def bad_handler(name: str, exc: BaseException) -> None:
        raise KeyError("handler broken")

    guard = ComponentGuard(on_fault=bad_handler)

    def boom() -> None:
        raise ValueError("x")

    assert guard.call("y", boom) is None
    assert guard.is_disabled("y")


def test_no_handler_is_fine() -> None:
    guard = ComponentGuard()

    def boom() -> None:
        raise KeyError("no handler needed")

    assert guard.call("z", boom) is None
    assert guard.faults == 1


def test_kwargs_are_forwarded() -> None:
    guard = ComponentGuard()
    result = guard.call("f", lambda *, scale: scale * 2, scale=5)
    assert result == 10
