"""Phase 1 input demonstration: verifies keys, holds and mouse movement.

These are low-level plumbing checks (milestone items 13-14 of the roadmap),
not gameplay macros: every step observes whether the input was accepted and
can be interrupted at any moment by the pause/emergency hotkeys.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from src.input.input_controller import InputController
from src.state.agent_status import StatusTracker

log = logging.getLogger(__name__)

Step = tuple[str, Callable[[], bool]]


class Phase1Demo:
    """Executes a fixed verification sequence one step at a time."""

    def __init__(
        self,
        input_controller: InputController,
        ensure_focus: Callable[[], bool],
        on_result: Callable[[str, bool, float], None] | None = None,
        settle_s: float = 0.4,
    ) -> None:
        self._input = input_controller
        self._focus = ensure_focus
        self._on_result = on_result
        self._settle = settle_s
        self._steps: list[Step] = [
            ("focus game window", self._focus),
            ("hold W 600ms", lambda: self._input.hold("w", 0.6)),
            ("tap W 80ms", lambda: self._input.press("w", 80)),
            ("mouse look right +150", lambda: self._input.mouse_move(150, 0)),
            ("mouse look left -150", lambda: self._input.mouse_move(-150, 0)),
            ("tap SHIFT 120ms", lambda: self._input.press("shift", 120)),
        ]
        self._index = 0
        self._next_at = 0.0
        self.results: list[tuple[str, bool, float]] = []
        self.active = True

    @property
    def current(self) -> str:
        if not self.active:
            return "complete"
        return self._steps[self._index][0]

    def step(self, status: StatusTracker) -> bool:
        """Advance one demo step; returns True while the demo is still running."""
        if not self.active:
            return False
        now = time.monotonic()
        if now < self._next_at:
            return True
        label, fn = self._steps[self._index]
        status.update(demo_step=f"{label} ...")
        t0 = time.perf_counter()
        try:
            ok = bool(fn())
        except Exception as exc:
            log.exception("demo step failed: %s", label)
            ok = False
            status.update(error=f"demo: {type(exc).__name__}: {exc}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.results.append((label, ok, elapsed_ms))
        level = logging.INFO if ok else logging.ERROR
        log.log(level, "demo step %-24s %s (%.0fms)", label, "OK" if ok else "FAILED", elapsed_ms)
        if self._on_result is not None:
            self._on_result(label, ok, elapsed_ms)
        self._index += 1
        self._next_at = now + self._settle
        if self._index >= len(self._steps):
            self.active = False
            passed = sum(1 for _, ok, _ in self.results if ok)
            summary = f"{passed}/{len(self.results)} steps passed"
            log.info("Phase 1 demo complete: %s", summary)
            status.update(demo_step="complete: " + summary)
        return self.active
