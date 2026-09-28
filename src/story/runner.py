"""Story mode director: runs configured missions end to end (NOTES Mode 1).

Episode stages per mission: mount (optional whistle-and-ride) -> travel
(:class:`RoutePlanner` legs, optionally homing toward a minimap marker) ->
objective (:class:`MissionRunner` tasks) -> complete (completion-keyword
watch) -> next mission. Failures and stage timeouts skip forward instead of
parking the session; the combat/survival gates in the main loop simply pause
the director and the pause-aware stage timers do not count that time.

Stranger (side) missions declare ``kind: stranger``; with
``story.opportunistic`` the director detours to one when its marker is
visible, then resumes the story mission it left.

Honor is handled outside this class: ``story.auto_greet`` arms the existing
prompt responder to greet on every opportunity (the honor value itself is not
readable from the HUD).
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import numpy as np

from src.config import ControlConfig, MissionConfig, StoryConfig
from src.control.locomotion import LocomotionController
from src.mission.runner import MissionRunner
from src.planning.route import RoutePlanner, angle_delta
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState
from src.vision.ocr import OcrEngine

log = logging.getLogger(__name__)

_STAGES = ("mount", "travel", "dismount", "objective", "roam", "identify",
            "complete", "done")


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
        self._mounted: ControlConfig | None = None
        self._detour_from: int | None = None
        self._marker_frames = 0
        self._marker_seen = False
        self._identify_until = 0.0
        self._identified = False
        self._quiet_since = 0.0
        self._mount_whistled = False
        self._mount_settle_until = 0.0
        self._heading = float(control.nav.start_heading_deg)
        self._dismount_faced = False
        self._dismount_tapped = False
        self._dismount_until = 0.0
        self._roam_walked = 0.0
        self._roam_turn_remaining = 0.0
        self._roam_prev: float | None = None
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
            self._begin_current(now, game_state)
        if self._stage == "mount":
            self._step_mount(now, status, game_state)
        elif self._stage == "travel":
            self._step_travel(now, status, frame, game_state)
        elif self._stage == "dismount":
            self._step_dismount(now, status)
        elif self._stage == "objective":
            self._step_objective(now, status, frame, game_state)
        elif self._stage == "roam":
            self._step_roam(now, status)
        elif self._stage == "identify":
            self._step_identify(now, status, frame, game_state)
        elif self._stage == "complete":
            self._step_complete(now, status, frame, game_state)
        if self._stage != "done":
            status.update(
                goal=f"STORY {self._index + 1}/{len(self._profiles)} "
                f"{self._stage.upper()}",
                message=self._name,
            )

    @property
    def _driver_control(self) -> ControlConfig:
        return self._mounted if self._mounted is not None else self._control

    def _begin_current(self, now: float, game_state: GameState | None) -> None:
        self._locomotion.stop()
        self._route = None
        self._mission = None
        self._mounted = None
        self._marker_frames = 0
        self._marker_seen = False
        self._mount_whistled = False
        self._mount_settle_until = 0.0
        self._heading = float(self._control.nav.start_heading_deg)
        self._dismount_faced = False
        self._dismount_tapped = False
        self._dismount_until = 0.0
        self._roam_walked = 0.0
        self._roam_turn_remaining = 0.0
        self._roam_prev = None
        detour = None
        if not self._identified:
            detour = self._opportunistic_target(game_state)
        self._identified = False
        if detour is not None:
            self._detour_from = self._index
            self._index = detour
            log.info(
                "story: opportunistic detour to stranger mission %d/%d",
                detour + 1, len(self._profiles),
            )
        profile = self._profiles[self._index]
        self._name = str(profile.get("name") or f"mission {self._index + 1}")
        parsed_legs: list[tuple[float, float]] = []
        for leg in profile.get("legs") or []:
            if not isinstance(leg, (list, tuple)) or len(leg) != 2:
                raise ValueError(
                    f"story profile {self._name!r}: bad leg {leg!r}"
                )
            parsed_legs.append((float(leg[0]), float(leg[1])))
        self._legs = parsed_legs
        raw_tasks = profile.get("tasks") or []
        self._tasks = [t for t in raw_tasks if isinstance(t, dict)]
        if profile.get("mount"):
            self._goto("mount", now)
        elif profile.get("roam"):
            self._goto("roam", now)
        else:
            self._after_mount(now)
        log.info(
            "story: mission %d/%d '%s' - stage %s (%d legs, %d tasks)",
            self._index + 1, len(self._profiles), self._name, self._stage,
            len(self._legs), len(self._tasks),
        )
        self._pending_begin = False

    def _opportunistic_target(self, game_state: GameState | None) -> int | None:
        """Later stranger profile to detour to when its marker is visible."""
        if not self._cfg.opportunistic or game_state is None:
            return None
        current = self._profiles[self._index]
        if current.get("kind", "story") == "stranger":
            return None
        if game_state.mission.objective_location_estimate is None:
            return None
        for i in range(self._index + 1, len(self._profiles)):
            if self._profiles[i].get("kind", "story") == "stranger":
                return i
        return None

    def _after_mount(self, now: float) -> None:
        self._locomotion.stop()
        profile = self._profiles[self._index]
        if self._legs or profile.get("nearest"):
            self._goto("travel", now)
        elif self._tasks:
            self._goto("objective", now)
        else:
            self._goto("complete", now)
        log.info("story: mission '%s' - stage %s", self._name, self._stage)

    def _goto(self, stage: str, now: float) -> None:
        self._stage = stage
        self._stage_started = now
        self._banner_counter = 0
        if stage == "complete":
            self._quiet_since = now

    def _step_mount(
        self, now: float, status: StatusTracker, game_state: GameState | None
    ) -> None:
        if self._mount_settle_until and now >= self._mount_settle_until:
            self._finish_mount(now)
            return
        encounter = (
            game_state.dialogue.encounter if game_state is not None else None
        )
        if encounter == "mount" and not self._mount_settle_until:
            if self._locomotion.tap("interact"):
                self._mount_settle_until = now + self._cfg.mount_wait_s
                status.update(action="mount:board")
            else:
                log.warning("story: mount tap refused - continuing on foot")
                self._after_mount(now)
            return
        if not self._mount_whistled:
            self._mount_whistled = True
            if self._locomotion.tap("whistle"):
                status.update(action="mount:whistle")
            else:
                log.warning("story: whistle refused - continuing on foot")
                self._after_mount(now)
            return
        if now - self._stage_started >= self._cfg.mount_timeout_s:
            log.warning("story: horse never came - continuing on foot")
            self._after_mount(now)
            return
        status.update(action="mount:wait")

    def _finish_mount(self, now: float) -> None:
        self._mounted = replace(
            self._control,
            nav=replace(
                self._control.nav, walk_speed_mps=self._cfg.mounted_speed_mps
            ),
        )
        log.info(
            "story: mounted - riding at %.1f m/s", self._cfg.mounted_speed_mps
        )
        self._after_mount(now)

    def _step_travel(
        self, now: float, status: StatusTracker,
        frame: np.ndarray | None, game_state: GameState | None,
    ) -> None:
        profile = self._profiles[self._index]
        if profile.get("nearest"):
            self._step_nearest(now, status, game_state)
            return
        if self._route is None:
            self._route = RoutePlanner(
                self._driver_control, self._locomotion, self._minimap_frac
            )
            self._route.set_legs(self._legs)
        self._apply_seek_correction(status, game_state)
        self._route.step(now, status, frame)
        if self._route.failed:
            self._fail_advance(f"route blocked ({self._route.current})", now)
            return
        if self._route.arrived:
            self._heading = self._route.heading_deg
            self._route = None
            if self._should_dismount():
                self._goto("dismount", now)
            elif self._should_identify():
                self._enter_identify(now)
            elif self._tasks:
                self._goto("objective", now)
            else:
                self._goto("complete", now)
            log.info("story: mission '%s' - stage %s", self._name, self._stage)
            return
        if now - self._stage_started >= self._cfg.travel_timeout_s:
            self._fail_advance(
                f"travel timed out after {self._cfg.travel_timeout_s:.0f}s", now
            )

    def _apply_seek_correction(
        self, status: StatusTracker, game_state: GameState | None
    ) -> None:
        """Nudge the route heading toward a persistent minimap marker."""
        profile = self._profiles[self._index]
        if not profile.get("seek_marker") or self._route is None:
            self._marker_frames = 0
            return
        estimate = (
            game_state.mission.objective_location_estimate
            if game_state is not None else None
        )
        if not estimate or len(estimate) != 2:
            self._marker_frames = 0
            return
        self._marker_frames += 1
        if self._marker_frames < self._cfg.marker_persist_frames:
            return
        dx, dy = float(estimate[0]), float(estimate[1])
        angle = math.degrees(math.atan2(dx, -dy))
        nav = self._driver_control.nav
        if abs(angle) <= nav.heading_tol_deg:
            return
        cap = self._driver_control.max_turn_deg_per_tick
        delta = max(-cap, min(angle, cap))
        if self._locomotion.turn(delta):
            self._route.nudge_heading(delta)
            status.update(action=f"seek:{angle:+.0f}deg")

    def _should_dismount(self) -> bool:
        if self._mounted is None:
            return False
        profile = self._profiles[self._index]
        return (
            bool(profile.get("dismount"))
            or profile.get("kind", "story") == "camp"
        )

    def _step_dismount(self, now: float, status: StatusTracker) -> None:
        profile = self._profiles[self._index]
        bearing = profile.get("dismount_bearing")
        if bearing is not None and not self._dismount_faced:
            err = angle_delta(float(bearing), self._heading)
            if abs(err) <= self._control.nav.heading_tol_deg:
                self._dismount_faced = True
            else:
                cap = self._control.max_turn_deg_per_tick
                delta = max(-cap, min(err, cap))
                if self._locomotion.turn(delta):
                    self._heading += delta
                status.update(action=f"dismount:turn{err:+.0f}deg")
                return
        if not self._dismount_tapped:
            self._dismount_tapped = True
            if self._locomotion.tap("interact"):
                self._dismount_until = now + self._cfg.dismount_wait_s
                status.update(action="dismount:wait")
            else:
                log.warning("story: dismount tap refused - continuing")
            return
        if now >= self._dismount_until:
            self._mounted = None
            self._locomotion.stop()
            log.info("story: dismounted")
            if self._should_identify():
                self._enter_identify(now)
            elif self._tasks:
                self._goto("objective", now)
            else:
                self._goto("complete", now)
            log.info("story: mission '%s' - stage %s", self._name, self._stage)

    def _step_roam(self, now: float, status: StatusTracker) -> None:
        nav = self._driver_control.nav
        dt = 0.0
        if self._roam_prev is not None:
            dt = min(max(now - self._roam_prev, 0.0), 1.0)
        self._roam_prev = now
        if abs(self._roam_turn_remaining) > 0.5:
            cap = self._driver_control.max_turn_deg_per_tick
            delta = max(-cap, min(self._roam_turn_remaining, cap))
            if self._locomotion.turn(delta):
                self._roam_turn_remaining -= delta
            status.update(
                action=f"roam:turn{self._roam_turn_remaining:+.0f}deg"
            )
            return
        driving = self._locomotion.active
        if not driving:
            driving = self._locomotion.move("forward", nav.step_s)
        if driving:
            self._roam_walked += nav.walk_speed_mps * dt
        if self._roam_walked >= self._cfg.roam_leg_m:
            self._roam_walked = 0.0
            self._roam_turn_remaining = self._cfg.roam_turn_deg
            status.update(action="roam:leg done")
        else:
            status.update(action=f"roam:walk {self._roam_walked:.1f}m")

    def _step_nearest(
        self, now: float, status: StatusTracker, game_state: GameState | None
    ) -> None:
        """Chase the nearest minimap marker instead of following legs."""
        if now - self._stage_started >= self._cfg.travel_timeout_s:
            self._fail_advance(
                f"no marker reached after {self._cfg.travel_timeout_s:.0f}s",
                now,
            )
            return
        estimate = (
            game_state.mission.objective_location_estimate
            if game_state is not None else None
        )
        if estimate and len(estimate) == 2:
            self._marker_frames += 1
        else:
            self._marker_frames = 0
            estimate = None
        if estimate is None:
            if self._marker_seen:
                self._arrive_from_nearest(now, "marker consumed")
                return
            self._locomotion.turn(self._control.max_turn_deg_per_tick)
            status.update(action="nearest:scan")
            return
        if self._marker_frames < self._cfg.marker_persist_frames:
            status.update(action="nearest:acquire")
            return
        self._marker_seen = True
        dx, dy = float(estimate[0]), float(estimate[1])
        if math.hypot(dx, dy) <= self._cfg.marker_arrival_radius:
            self._arrive_from_nearest(now, "marker reached")
            return
        angle = math.degrees(math.atan2(dx, -dy))
        cap = self._control.max_turn_deg_per_tick
        delta = max(-cap, min(angle, cap))
        self._locomotion.turn(delta)
        if not self._locomotion.active:
            self._locomotion.move("forward", self._control.nav.step_s)
        status.update(action=f"nearest:{angle:+.0f}deg")

    def _arrive_from_nearest(self, now: float, reason: str) -> None:
        log.info("story: mission '%s' - nearest marker %s", self._name, reason)
        self._locomotion.stop()
        self._marker_seen = False
        if self._library_has_titles():
            self._enter_identify(now)
        else:
            self._enter_post_identify(now)

    def _library_has_titles(self) -> bool:
        for profile in self._profiles:
            titles = profile.get("titles") or []
            if isinstance(titles, list) and any(
                isinstance(t, str) and t.strip() for t in titles
            ):
                return True
        return False

    def _enter_identify(self, now: float) -> None:
        self._goto("identify", now)
        self._identify_until = now + self._cfg.title_wait_s

    def _should_identify(self) -> bool:
        titles = self._profiles[self._index].get("titles") or []
        return isinstance(titles, list) and len(titles) > 0

    def _step_identify(
        self, now: float, status: StatusTracker,
        frame: np.ndarray | None, game_state: GameState | None,
    ) -> None:
        status.update(action="identify:title")
        hit = self._match_title(game_state, frame)
        if hit is not None:
            index, title = hit
            if index != self._index:
                log.info(
                    "story: title %r matched mission '%s'",
                    title, self._profiles[index].get("name"),
                )
                self._index = index
                self._detour_from = None
                self._identified = True
                self._pending_begin = True
                return
            log.info("story: title %r confirmed mission '%s'", title, self._name)
            self._enter_post_identify(now)
            return
        if now >= self._identify_until:
            log.info(
                "story: title not recognized - continuing '%s'", self._name
            )
            self._enter_post_identify(now)

    def _match_title(
        self, game_state: GameState | None, frame: np.ndarray | None
    ) -> tuple[int, str] | None:
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
        for i, profile in enumerate(self._profiles):
            titles = profile.get("titles") or []
            if not isinstance(titles, list):
                continue
            for title in titles:
                needle = str(title).strip().lower()
                if needle and needle in joined:
                    return i, str(title)
        return None

    def _enter_post_identify(self, now: float) -> None:
        if self._tasks:
            self._goto("objective", now)
        else:
            self._goto("complete", now)
        log.info("story: mission '%s' - stage %s", self._name, self._stage)

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
                self._driver_control,
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
        if self._anything_on_screen(game_state):
            self._quiet_since = now
        elif now - self._quiet_since >= self._cfg.completion_idle_timeout_s:
            self._advance(now, "nothing happening - moving on")
            return
        if now - self._stage_started >= self._cfg.completion_timeout_s:
            self._advance(
                now,
                f"completion not seen after {self._cfg.completion_timeout_s:.0f}s",
            )

    @staticmethod
    def _anything_on_screen(game_state: GameState | None) -> bool:
        if game_state is None:
            return False
        return bool(
            game_state.mission.objective_text
            or game_state.mission.prompt_text
            or game_state.dialogue.text
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
        finished = self._profiles[self._index]
        self._index += 1
        if (
            finished.get("kind", "story") == "stranger"
            and self._detour_from is not None
        ):
            self._index = self._detour_from
            self._detour_from = None
            log.info(
                "story: detour complete - resuming mission %d/%d",
                self._index + 1, len(self._profiles),
            )
            self._pending_begin = True
            return
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
                self._identify_until += gap
                self._mount_settle_until += gap
                self._dismount_until += gap
        self._last_step = now
