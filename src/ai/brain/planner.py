"""Vision-brain driver: a background thread asks the model, the loop acts.

The model is slow (seconds per decision on CPU), so it never runs on the
main loop: a daemon thread polls the latest observed frame every
``interval_s``, and :meth:`step` only executes a cached decision while it
is fresh. Combat/survival suppression in the main loop applies unchanged,
and every locomotion call goes through the same focus gates and clamps.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from src.ai.brain.client import ask_ollama
from src.ai.brain.prompt import ACTIONS, build_prompt, parse_decision
from src.config import BrainConfig
from src.control.locomotion import LocomotionController
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState

log = logging.getLogger(__name__)


def _downscale(frame: np.ndarray, max_px: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_px:
        return frame
    scale = max_px / float(w)
    return cv2.resize(frame, (max_px, max(1, int(h * scale))))


def _encode(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _summarize(game_state: GameState | None) -> dict[str, object]:
    if game_state is None:
        return {"threat": "unknown"}
    threat = game_state.threat
    player = game_state.player
    mission = game_state.mission
    dialogue = game_state.dialogue
    summary: dict[str, object] = {
        "threat": threat.level,
        "wanted": threat.wanted_level,
        "enemies": threat.enemies_detected,
    }
    if player.health is not None:
        summary["health"] = round(player.health, 2)
    if player.stamina is not None:
        summary["stamina"] = round(player.stamina, 2)
    if player.dead_eye is not None:
        summary["dead_eye"] = round(player.dead_eye, 2)
    if mission.objective_text:
        summary["objective"] = mission.objective_text[:160]
    if mission.prompt_text:
        summary["prompt"] = mission.prompt_text[:160]
    if dialogue.text:
        summary["dialogue"] = dialogue.text[:160]
    if player.weapon_ammo is not None:
        summary["ammo"] = player.weapon_ammo
    return summary


class BrainPlanner:
    """Primary-planner duck type: ``step(now, status, frame, game_state)``."""

    def __init__(
        self,
        cfg: BrainConfig,
        locomotion: LocomotionController,
        ask_fn: Callable[[str, str], str] | None = None,
    ) -> None:
        self._cfg = cfg
        self._locomotion = locomotion
        self._ask_fn = ask_fn
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: np.ndarray | None = None
        self._summary: dict[str, object] = {"threat": "unknown"}
        self._action = "noop"
        self._reason = "waiting for the first decision"
        self._decided_at = 0.0
        self._executed = True
        self._asked = 0
        self._failures = 0
        self._latency_ms = 0.0
        self._note = "starting"
        self._thread = threading.Thread(
            target=self._loop, name="brain", daemon=True
        )
        self._thread.start()

    @property
    def pending(self) -> bool:
        with self._lock:
            return not self._executed

    @property
    def decisions(self) -> int:
        with self._lock:
            return self._asked

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def note(self) -> str:
        with self._lock:
            return self._note

    def observe(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        with self._lock:
            self._frame = frame.copy()

    def step(
        self,
        now: float,
        status: StatusTracker,
        frame: np.ndarray | None = None,
        game_state: GameState | None = None,
    ) -> None:
        """Feed the frame to the thinker; execute one fresh cached decision."""
        self._locomotion.tick()
        if frame is not None:
            self.observe(frame)
        with self._lock:
            self._summary = _summarize(game_state)
            action, reason = self._action, self._reason
            age = now - self._decided_at
            fresh = not self._executed and age <= self._cfg.stale_after_s
            if fresh:
                self._executed = True
        if not fresh:
            status.update(
                goal=f"BRAIN (phase {status.phase})",
                action="brain:wait",
                message=self.note,
            )
            return
        self._execute(action)
        status.update(
            goal=f"BRAIN (phase {status.phase})",
            action=f"brain:{action}",
            message=reason,
        )

    def _execute(self, action: str) -> None:
        cfg = self._cfg
        loco = self._locomotion
        if action == "forward":
            loco.move("forward", cfg.step_s)
        elif action == "sprint":
            loco.move("forward", cfg.step_s, sprint=True)
        elif action == "back":
            loco.move("back", cfg.step_s)
        elif action in ("strafe_left", "strafe_right"):
            loco.move(action, cfg.step_s)
        elif action == "turn_left":
            loco.turn(-cfg.turn_step_deg)
        elif action == "turn_right":
            loco.turn(cfg.turn_step_deg)
        elif action == "interact":
            loco.tap("interact")
        elif action == "whistle":
            loco.tap("whistle")
        elif action in ACTIONS:
            pass
        else:
            log.warning("brain: dropping unknown action %r", action)

    def close(self) -> None:
        """Stop the thinker thread; safe to call repeatedly."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._cfg.interval_s):
            try:
                self._cycle()
            except Exception as exc:  # noqa: BLE001 - thread must survive
                log.warning("brain decision failed: %s", exc)
                with self._lock:
                    self._failures += 1
                    self._note = f"model error: {exc}"

    def _cycle(self) -> None:
        with self._lock:
            frame = self._frame
            summary = dict(self._summary)
        if frame is None:
            with self._lock:
                self._note = "waiting for the first frame"
            return
        image_b64 = _encode(_downscale(frame, self._cfg.max_image_px))
        prompt = build_prompt(summary)
        start = time.monotonic()
        if self._ask_fn is not None:
            text = self._ask_fn(prompt, image_b64)
        else:
            text = ask_ollama(
                self._cfg.host, self._cfg.model, prompt, image_b64,
                self._cfg.timeout_s,
            )
        latency_ms = (time.monotonic() - start) * 1000.0
        action, reason, clean = parse_decision(text)
        if not clean:
            log.warning("brain: repaired model reply (%s)", reason)
        with self._lock:
            self._action = action
            self._reason = reason or action
            self._decided_at = time.monotonic()
            self._executed = False
            self._asked += 1
            self._latency_ms = round(latency_ms, 1)
            self._note = f"{action} ({self._latency_ms:.0f}ms)"
        log.info("brain: %s - %s", action, reason)
