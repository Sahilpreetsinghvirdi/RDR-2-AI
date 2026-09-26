"""Mission runner: executes validated tasks one control tick at a time (Phase 5).

The runner is duck-compatible with the other planners: ``step(now, status,
frame, game_state)``. It reuses the locomotion controller for turning,
interacting and (via a single-leg :class:`RoutePlanner`) walking, so every
movement goes through the same clamps, focus gates and release paths.
A failed task stops all input and parks the mission as ``MISSION FAILED``;
there is no automatic retry.
"""

from __future__ import annotations

import logging

import numpy as np

from src.config import ControlConfig, MissionConfig
from src.control.locomotion import LocomotionController
from src.mission.tasks import Task, parse_tasks, state_field
from src.planning.route import RoutePlanner
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState

log = logging.getLogger(__name__)

_TURN_EPS = 0.5


class MissionRunner:
    """Steps through ``mission.tasks``; optional restart when ``loop`` is set."""

    def __init__(
        self,
        cfg: MissionConfig,
        control: ControlConfig,
        locomotion: LocomotionController,
        minimap_frac: list[float] | None = None,
    ) -> None:
        self._cfg = cfg
        self._control = control
        self._locomotion = locomotion
        self._minimap_frac = minimap_frac
        self._tasks: list[Task] = parse_tasks(cfg.tasks)
        self._index = 0
        self._round = 0
        self._heading = float(control.nav.start_heading_deg)
        self._task: Task | None = None
        self._turn_remaining = 0.0
        self._wait_until = 0.0
        self._wait_for_since = 0.0
        self._walk: RoutePlanner | None = None
        self._interact_done = False
        self._failed: str | None = None
        self._done = False

    @property
    def failed(self) -> bool:
        return self._failed is not None

    @property
    def done(self) -> bool:
        return self._done

    @property
    def heading_deg(self) -> float:
        return self._heading

    @property
    def tasks_completed(self) -> int:
        return self._index

    @property
    def current(self) -> str:
        if self._failed is not None:
            return "failed"
        if self._done:
            return "complete"
        return f"task {self._index + 1}/{len(self._tasks)}"

    def step(
        self,
        now: float,
        status: StatusTracker,
        frame: np.ndarray | None = None,
        game_state: GameState | None = None,
    ) -> None:
        """Advance one control tick through the current task."""
        self._locomotion.tick()

        if self._failed is not None:
            status.update(
                goal="MISSION FAILED",
                action="none",
                message=self._failed,
            )
            return
        if self._done:
            status.update(goal="MISSION DONE", action="none")
            return
        if self._index >= len(self._tasks):
            self._finish(status)
            return

        if self._task is not self._tasks[self._index]:
            self._begin(self._tasks[self._index], now)

        task = self._task
        assert task is not None
        goal = f"MISSION {self._index + 1}/{len(self._tasks)}"
        status.update(goal=goal, action=task.label)

        if task.kind == "log":
            log.info("mission[%d]: %s", self._index + 1, task.text)
            self._advance()
        elif task.kind == "turn":
            self._step_turn(task, status, goal)
        elif task.kind == "wait":
            if now >= self._wait_until:
                self._advance()
        elif task.kind == "wait_for":
            self._step_wait_for(task, now, status, goal, game_state)
        elif task.kind == "interact":
            if not self._interact_done:
                ok = self._locomotion.tap("interact")
                if not ok:
                    self._fail(f"task {self._index + 1}: interact key refused")
                    status.update(goal="MISSION FAILED", action="none", message=self._failed)
                    return
                self._interact_done = True
                self._advance()
        elif task.kind == "walk":
            self._step_walk(task, now, status, goal, frame)

        if self._index >= len(self._tasks):
            self._finish(status)

    def _begin(self, task: Task, now: float) -> None:
        self._task = task
        self._interact_done = False
        if task.kind == "turn":
            self._turn_remaining = task.deg
        elif task.kind == "wait":
            self._wait_until = now + task.seconds
        elif task.kind == "wait_for":
            self._wait_for_since = now
        elif task.kind == "walk":
            bearing = self._heading if task.bearing is None else task.bearing
            cfg = self._control
            self._walk = RoutePlanner(
                cfg, self._locomotion, self._minimap_frac,
                initial_heading=self._heading,
            )
            self._walk.set_legs([(bearing, task.meters)])
        else:
            self._walk = None
        log.debug("mission[%d]: begin %s", self._index + 1, task.label)

    def _step_turn(self, task: Task, status: StatusTracker, goal: str) -> None:
        remaining = self._turn_remaining
        if abs(remaining) <= _TURN_EPS:
            self._advance()
            return
        applied = max(
            -self._control.max_turn_deg_per_tick,
            min(remaining, self._control.max_turn_deg_per_tick),
        )
        self._locomotion.turn(applied)
        self._heading += applied
        self._turn_remaining = remaining - applied
        status.update(goal=goal, action=f"turn:{remaining:+.0f}deg")
        if abs(self._turn_remaining) <= _TURN_EPS:
            self._advance()

    def _step_wait_for(
        self,
        task: Task,
        now: float,
        status: StatusTracker,
        goal: str,
        game_state: GameState | None,
    ) -> None:
        value = state_field(game_state, task.field) if game_state is not None else None
        if value == task.equals:
            self._advance()
            return
        if task.timeout_s is not None and now - self._wait_for_since >= task.timeout_s:
            self._fail(
                f"task {self._index + 1}: timed out after {task.timeout_s:.0f}s "
                f"waiting for {task.field} == {task.equals!r} (last={value!r})"
            )
            status.update(goal="MISSION FAILED", action="none", message=self._failed)
            return
        elapsed = now - self._wait_for_since
        status.update(goal=goal, action=f"wait_for:{task.field} {elapsed:.0f}s")

    def _step_walk(
        self,
        task: Task,
        now: float,
        status: StatusTracker,
        goal: str,
        frame: np.ndarray | None,
    ) -> None:
        assert self._walk is not None
        self._walk.step(now, status, frame)
        self._heading = self._walk.heading_deg
        if self._walk.failed:
            self._fail(f"task {self._index + 1}: walk blocked on minimap")
            status.update(goal="MISSION FAILED", action="none", message=self._failed)
            return
        if self._walk.arrived:
            self._walk = None
            self._advance()
            status.update(message="")
        else:
            status.update(goal=goal, action=f"walk {task.label}")

    def _advance(self) -> None:
        self._task = None
        self._walk = None
        self._index += 1

    def _finish(self, status: StatusTracker) -> None:
        self._locomotion.stop()
        if self._cfg.loop:
            log.info(
                "mission: round %d complete - restarting (%d tasks)",
                self._round + 1, len(self._tasks),
            )
            self._round += 1
            self._index = 0
            self._task = None
            status.update(goal="MISSION LOOP", action="none",
                          message=f"round {self._round} complete")
            return
        self._done = True
        log.info("mission: complete (%d tasks)", len(self._tasks))
        status.update(
            goal="MISSION DONE",
            action="none",
            message=f"mission complete: {len(self._tasks)} tasks",
        )

    def _fail(self, reason: str) -> None:
        self._failed = reason
        self._locomotion.stop()
        log.error("mission failed: %s", reason)
