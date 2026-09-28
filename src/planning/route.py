"""Route navigation planner: dead-reckoned legs with stall detection (Phase 4).

A route is a list of ``[bearing_deg, meters]`` legs from configuration.
Each leg turns the camera to its bearing (integrating our own clamped mouse
turns - dead reckoning), then walks forward in short bursts, accumulating
distance from the assumed walk speed. While walking, optical phase
correlation on the minimap region measures real screen motion; if the
minimap does not move while walking (wall, cliff, menu), the leg is marked
blocked instead of silently spinning its wheels. Nothing here infers
position beyond what the configured route and our own commands imply.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from src.config import ControlConfig
from src.control.locomotion import LocomotionController
from src.state.agent_status import StatusTracker

log = logging.getLogger(__name__)

_MAX_DT_S = 1.0


def angle_delta(target_deg: float, current_deg: float) -> float:
    """Shortest signed turn from *current* to *target*, in -180..180."""
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


class RoutePlanner:
    """Follows configured legs; duck-types ScriptedPlanner.step()."""

    def __init__(
        self,
        cfg: ControlConfig,
        locomotion: LocomotionController,
        minimap_frac: list[float] | None = None,
        initial_heading: float | None = None,
    ) -> None:
        self._cfg = cfg
        self._nav = cfg.nav
        self._locomotion = locomotion
        self._legs = [(float(b), float(m)) for b, m in self._nav.legs]
        self._heading = (
            float(self._nav.start_heading_deg) if initial_heading is None
            else float(initial_heading)
        )
        self._leg = 0
        self._leg_phase = "turn"      # turn | walk | blocked
        self._progress_m = 0.0
        self._done = False
        self._last_step: float | None = None
        self._no_flow_since: float | None = None
        self._minimap_frac = minimap_frac
        self._prev_roi: np.ndarray | None = None
        self._flow: float | None = None
        self._legs_completed = 0
        self._blocked = False

    @property
    def legs_completed(self) -> int:
        return self._legs_completed

    @property
    def arrived(self) -> bool:
        return self._done

    @property
    def failed(self) -> bool:
        """True when the current leg is blocked (no minimap motion)."""
        return self._blocked

    @property
    def blocked(self) -> bool:
        return self._blocked

    def set_legs(self, legs: list[tuple[float, float]]) -> None:
        """Install a fresh leg list (used by the mission runner) and reset."""
        self._legs = [(float(b), float(m)) for b, m in legs]
        self._leg = 0
        self._leg_phase = "turn"
        self._progress_m = 0.0
        self._done = False
        self._blocked = False
        self._no_flow_since = None
        self._legs_completed = 0
        self._locomotion.stop()

    @property
    def heading_deg(self) -> float:
        return self._heading

    def nudge_heading(self, delta_deg: float) -> None:
        """Shift the dead-reckoned heading after an external camera move."""
        self._heading += float(delta_deg)

    @property
    def current(self) -> str:
        if self._done:
            return "arrived"
        if self._leg >= len(self._legs):
            return "arrived"
        return f"leg {self._leg + 1}/{len(self._legs)}"

    def step(
        self,
        now: float,
        status: StatusTracker,
        frame: np.ndarray | None = None,
        game_state: object | None = None,
    ) -> None:
        """Advance one control tick; *frame* drives the stall check."""
        self._locomotion.tick()
        self._measure_flow(frame)
        dt = 0.0
        if self._last_step is not None:
            dt = min(max(now - self._last_step, 0.0), _MAX_DT_S)
        self._last_step = now

        if self._done or self._leg >= len(self._legs):
            self._finish(status)
            return

        bearing, meters = self._legs[self._leg]
        goal = f"ROUTE {self._leg + 1}/{len(self._legs)}"

        if self._leg_phase == "turn":
            self._locomotion.stop()
            err = angle_delta(bearing, self._heading)
            if abs(err) <= self._nav.heading_tol_deg:
                self._leg_phase = "walk"
                self._progress_m = 0.0
                self._no_flow_since = None
                status.update(goal=goal, action="walk:start")
            else:
                applied = max(
                    -self._cfg.max_turn_deg_per_tick,
                    min(err, self._cfg.max_turn_deg_per_tick),
                )
                if self._locomotion.turn(applied):
                    self._heading += applied
                status.update(goal=goal, action=f"turn:{err:+.0f}deg")
            return

        # walk phase
        if self._leg_phase == "blocked":
            if self._flow is None or self._flow >= self._nav.stall_flow_eps:
                log.info("route: motion resumed, leg %d unblocked", self._leg + 1)
                self._leg_phase = "walk"
                self._no_flow_since = None
                self._blocked = False
            else:
                status.update(
                    goal=goal,
                    action="blocked",
                    message=f"route leg {self._leg + 1}: no minimap motion - held",
                )
                return

        remaining = meters - self._progress_m
        if remaining <= 0.0:
            self._complete_leg(status)
            return

        driving = self._locomotion.active
        if not driving:
            sprint = self._nav.sprint_after_m > 0 and remaining > self._nav.sprint_after_m
            driving = self._locomotion.move("forward", self._nav.step_s, sprint=sprint)

        if driving:
            self._progress_m += self._nav.walk_speed_mps * dt

        if self._check_stall(now):
            status.update(
                goal=goal,
                action="blocked",
                message=f"route leg {self._leg + 1}: no minimap motion - held",
            )
        else:
            status.update(
                goal=goal,
                action=f"walk {self._progress_m:.1f}/{meters:.1f}m",
            )
        if meters - self._progress_m <= 0.0:
            self._complete_leg(status)

    def _complete_leg(self, status: StatusTracker) -> None:
        self._locomotion.stop()
        self._legs_completed += 1
        log.info(
            "route: leg %d/%d complete (%.1fm)",
            self._leg + 1, len(self._legs), self._legs[self._leg][1],
        )
        self._leg += 1
        self._leg_phase = "turn"
        self._progress_m = 0.0
        self._no_flow_since = None
        if self._leg >= len(self._legs):
            self._finish(status)

    def _finish(self, status: StatusTracker) -> None:
        if not self._done:
            self._done = True
            self._locomotion.stop()
            log.info("route: arrived (%d legs)", self._legs_completed)
            status.update(
                goal="ROUTE ARRIVED",
                action="none",
                message=f"route complete: {self._legs_completed} legs",
            )
        else:
            status.update(goal="ROUTE ARRIVED", action="none")

    def _check_stall(self, now: float) -> bool:
        """True while the walk is held because the minimap is not moving."""
        if self._flow is None:
            return False
        if self._flow >= self._nav.stall_flow_eps:
            self._no_flow_since = None
            return False
        if self._no_flow_since is None:
            self._no_flow_since = now
            return False
        if now - self._no_flow_since >= self._nav.stall_timeout_s:
            if not self._blocked:
                self._blocked = True
                self._leg_phase = "blocked"
                self._locomotion.stop()
                log.warning("route: leg %d blocked (no minimap motion)", self._leg + 1)
            return True
        return False

    def _measure_flow(self, frame: np.ndarray | None) -> None:
        """Optical phase correlation magnitude (px/frame) over the minimap."""
        if frame is None or self._minimap_frac is None:
            self._flow = None
            self._prev_roi = None
            return
        h, w = frame.shape[:2]
        fx, fy, fw, fh = self._minimap_frac
        x0, y0 = int(fx * w), int(fy * h)
        x1, y1 = int((fx + fw) * w), int((fy + fh) * h)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            self._flow = None
            self._prev_roi = None
            return
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev_roi is None or self._prev_roi.shape != gray.shape:
            self._prev_roi = gray
            self._flow = None
            return
        (dx, dy), _resp = cv2.phaseCorrelate(self._prev_roi, gray)
        self._prev_roi = gray
        self._flow = math.hypot(dx, dy)
