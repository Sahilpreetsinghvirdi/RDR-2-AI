"""Fault isolation for long-running sessions (Phase 10).

A component that throws while stepping (planner, mission runner, responder,
assessor, demo) is **disabled for the rest of the session** instead of
crashing the agent: inputs are released, the fault is logged and emitted,
and the loop continues in observation mode. Perception failures still
propagate - the agent must never keep acting on broken vision.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

FaultHandler = Callable[[str, BaseException], None]


class ComponentGuard:
    """Run component step calls; the first exception disables that component."""

    def __init__(self, on_fault: FaultHandler | None = None) -> None:
        self._on_fault = on_fault
        self._disabled: dict[str, str] = {}
        self._faults = 0

    @property
    def faults(self) -> int:
        return self._faults

    @property
    def disabled(self) -> dict[str, str]:
        return dict(self._disabled)

    def is_disabled(self, name: str) -> bool:
        return name in self._disabled

    def summary(self) -> dict[str, str]:
        """Disabled components as ``{name: reason}`` for telemetry."""
        return dict(self._disabled)

    def call(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke ``fn`` unless *name* already faulted; ``None`` on a fault."""
        if name in self._disabled:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            self._disabled[name] = f"{type(exc).__name__}: {exc}"
            self._faults += 1
            log.warning(
                "component %r failed and is disabled for this session: %s",
                name, self._disabled[name], exc_info=True,
            )
            if self._on_fault is not None:
                try:
                    self._on_fault(name, exc)
                except Exception:
                    log.exception("fault handler failed for %r", name)
            return None
