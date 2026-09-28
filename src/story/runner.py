"""Story mode director: runs configured missions end to end (NOTES Mode 1).

Episode stages per mission: travel (:class:`RoutePlanner` legs) -> objective
(:class:`MissionRunner` tasks) -> complete (completion-keyword watch) -> next
mission. Failures and stage timeouts skip forward instead of parking the
session; the combat/survival gates in the main loop simply pause the director
and the pause-aware stage timers do not count that time.

Honor is handled outside this class: ``story.auto_greet`` arms the existing
prompt responder to greet on every opportunity (the honor value itself is not
readable from the HUD).
"""

from __future__ import annotations

import logging

import numpy as np

from src.config import ControlConfig, MissionConfig, StoryConfig
from src.control.locomotion import LocomotionController
from src.mission.runner import MissionRunner
from src.planning.route import RoutePlanner
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState
from src.vision.ocr import OcrEngine

log = logging.getLogger(__name__)

_STAGES = ("travel", "objective", "complete", "done")


class StoryRunner:
    """Primary-planner duck type: ``step(now, status, frame, game_state)``."""

    def __init__(
        self,
        cfg: StoryConfig,
        control: ControlConfig,
        locomotion: LocomotionController,
        minimap_frac: list[float] | None = None,
        ocr: OcrEngine | None = None,
        ocr_min_confidence: float = 40.0,
    ) -> None:
        self._cfg = cfg
        self._control = control
        self._locomotion = locomotion
        self._minimap_frac = minimap_frac
        self._ocr = ocr
        self._ocr_min_confidence = ocr_min_confidence
        self._profiles = [p for p in cfg.missions if isinstance(p, dict)]
        self._index = 0
        self._stage = "travel"
        self._pending_begin = True
        self._name = ""
        self._legs: list[tuple[float, float]] = []
        self._tasks: list[dict[str, object]] = []
        self._route: RoutePlanner | None = None
        self._mission: MissionRunner | None = None
        self._stage_started = 0.0
        self._last_step: float | None = None
        self._banner_counter = 0
        self._round = 0
        self._note = ""
        if not self._profiles:
            self._stage = "done"
            self._note = "no story missions configured"

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def index(self) -> int:
        return self._index

    @property
    def done(self) -> bool:
        return self._stage == "done"

    @property
    def current(self) -> str:
        if self._stage == "done":
            return "done"
        return f"mission {self._index + 1}/{len(self._profiles)} {self._stage}"

    def step(
        self,
        now: float,
        status: StatusTracker,
        frame: np.ndarray | None = None,
        game_state: GameState | None = None,
    ) -> None:
        """Advance one control tick through the current story stage."""
        self._rebase_timers(now)
        if self._stage == "done":
            status.update(goal="STORY DONE", action="none", message=self._note)
            return
        if self._pending_begin:
            self._begin_current(now)
        if self._stage == "travel":
            self._step_travel(now, status, frame)
        elif self._stage == "objective":
            self._step_objective(now, status, frame, game_state)
        elif self._stage == "complete":
            self._step_complete(now, status, frame, game_state)
        if self._stage != "done":
            status.update(
                goal=f"STORY {self._index + 1}/{len(self._profiles)} "
                f"{self._stage.upper()}",
                message=self._name,
            )

    def _begin_current(self, now: float) -> None:
        self._pending_begin = False
        self._locomotion.stop()
        self._route = None
        self._mission = None
        profile = self._profiles[self._index]
        self._name = str(profile.get("name") or f"mission {self._index + 1}")
        self._legs = [
            (float(leg[0]), float(leg[1]))
            for leg in (profile.get("legs") or [])
            if isinstance(leg, (list, tuple)) and len(leg) == 2
        ]
        raw_tasks = profile.get("tasks") or []
        self._tasks = [t for t in raw_tasks if isinstance(t, dict)]
        if self._legs:
            self._goto("travel", now)
        elif self._tasks:
            self._goto("objective", now)
        else:
            self._goto("complete", now)
        log.info(
            "story: mission %d/%d '%s' - stage %s (%d legs, %d tasks)",
            self._index + 1, len(self._profiles), self._name, self._stage,
            len(self._legs), len(self._tasks),
        )

    def _goto(self, stage: str, now: float) -> None:
        self._stage = stage
        self._stage_started = now
        self._banner_counter = 0

    def _step_travel(self, now: float, status: StatusTracker,
                     frame: np.ndarray | None) -> None:
        if self._route is None:
            self._route = RoutePlanner(
                self._control, self._locomotion, self._minimap_frac
            )
            self._route.set_legs(self._legs)
        self._route.step(now, status, frame)
        if self._route.failed:
            self._fail_advance(f"route blocked ({self._route.current})", now)
            return
        if self._route.arrived:
            self._route = None
            if self._tasks:
                self._goto("objective", now)
            else:
                self._goto("complete", now)
            log.info("story: mission '%s' - stage %s", self._name, self._stage)
            return
        if now - self._stage_started >= self._cfg.travel_timeout_s:
            self._fail_advance(
                f"travel timed out after {self._cfg.travel_timeout_s:.0f}s", now
            )

    def _step_objective(
        self,
        now: float,
        status: StatusTracker,
        frame: np.ndarray | None,
        game_state: GameState | None,
    ) -> None:
        if self._mission is None:
            self._mission = MissionRunner(
                MissionConfig(enabled=True, loop=False, tasks=list(self._tasks)),
                self._control,
                self._locomotion,
                self._minimap_frac,
            )
        self._mission.step(now, status, frame, game_state)
        if self._mission.failed:
            self._fail_advance(f"tasks failed ({self._mission.current})", now)
            return
        if self._mission.done:
            self._mission = None
            if self._tasks_done_should_watch():
                self._goto("complete", now)
                log.info("story: mission '%s' - stage complete", self._name)
            else:
                self._advance(now, "tasks complete (no completion watch)")
            return
        if now - self._stage_started >= self._cfg.objective_timeout_s:
            self._fail_advance(
                f"objectives timed out after {self._cfg.objective_timeout_s:.0f}s",
                now,
            )

    def _tasks_done_should_watch(self) -> bool:
        profile = self._profiles[self._index]
        wait = profile.get("wait_completion", True)
        return bool(wait)

    def _step_complete(
        self,
        now: float,
        status: StatusTracker,
        frame: np.ndarray | None,
        game_state: GameState | None,
    ) -> None:
        status.update(action="watch:completion")
        hit = self._find_completion(game_state, frame)
        if hit is not None:
            self._advance(now, f"completion seen ({hit!r})")
            return
        if now - self._stage_started >= self._cfg.completion_timeout_s:
            self._advance(
                now,
                f"completion not seen after {self._cfg.completion_timeout_s:.0f}s",
            )

    def _find_completion(
        self, game_state: GameState | None, frame: np.ndarray | None
    ) -> str | None:
        parts: list[str] = []
        if game_state is not None:
            texts = (
                game_state.mission.objective_text,
                game_state.mission.prompt_text,
                game_state.dialogue.text,
            )
            parts.extend(t for t in texts if t)
        banner = self._read_banner(frame)
        if banner:
            parts.append(banner)
        if not parts:
            return None
        joined = " ".join(parts).lower()
        for keyword in self._cfg.completion_keywords:
            needle = str(keyword).strip().lower()
            if needle and needle in joined:
                return str(keyword)
        return None

    def _read_banner(self, frame: np.ndarray | None) -> str | None:
        region = self._cfg.completion_region
        if frame is None or not region or self._ocr is None:
            return None
        if not self._ocr.available:
            return None
        self._banner_counter += 1
        if self._banner_counter % self._cfg.completion_every_n_frames != 0:
            return None
        fh, fw = frame.shape[:2]
        x = max(0, int(float(region[0]) * fw))
        y = max(0, int(float(region[1]) * fh))
        w = max(4, int(float(region[2]) * fw))
        h = max(4, int(float(region[3]) * fh))
        x, y = min(x, fw - 4), min(y, fh - 4)
        w, h = min(w, fw - x), min(h, fh - y)
        text, conf = self._ocr.read(frame[y:y + h, x:x + w])
        if text and conf * 100.0 >= self._ocr_min_confidence:
            return text
        return None

    def _advance(self, now: float, reason: str) -> None:
        self._locomotion.stop()
        log.info(
            "story: mission %d/%d '%s' - %s",
            self._index + 1, len(self._profiles), self._name, reason,
        )
        self._index += 1
        if self._index >= len(self._profiles):
            if self._cfg.loop and self._profiles:
                self._round += 1
                self._index = 0
                self._note = f"round {self._round} complete"
                log.info("story: round %d complete - restarting", self._round)
                self._pending_begin = True
                return
            self._stage = "done"
            self._note = f"all {len(self._profiles)} missions complete"
            log.info("story: %s", self._note)
            return
        self._pending_begin = True

    def _fail_advance(self, reason: str, now: float) -> None:
        log.warning(
            "story: mission %d/%d '%s' - %s - moving on",
            self._index + 1, len(self._profiles), self._name, reason,
        )
        self._note = reason
        self._advance(now, reason)

    def _rebase_timers(self, now: float) -> None:
        """Shift stage timers across pause/gaps so suppression time is free."""
        if self._last_step is not None:
            gap = now - self._last_step
            if gap > self._cfg.pause_grace_s:
                self._stage_started += gap
        self._last_step = now
