"""Policy planner: act from a trained checkpoint or explore for data (Phase 9).

Duck-typed like the other planners: ``step(now, status, frame, game_state)``.
In ``policy`` mode it stays **inactive** (and sends nothing) until a
checkpoint loads; in ``random`` mode it explores the scripted-skill action
space to bootstrap a transition buffer. Rewards are computed only from
known state deltas between action ticks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.ai.rl.actions import ACTIONS, execute, n_actions
from src.ai.rl.buffer import TransitionBuffer
from src.ai.rl.features import feature_dim, featurize
from src.ai.rl.policy import Policy
from src.ai.rl.reward import reward_delta
from src.config import RlConfig
from src.control.locomotion import LocomotionController
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState

log = logging.getLogger(__name__)


def _snapshot(state: GameState) -> GameState:
    """Cores + threat level only - enough for the reward function."""
    snap = GameState()
    snap.player.health = state.player.health
    snap.player.stamina = state.player.stamina
    snap.player.dead_eye = state.player.dead_eye
    snap.threat.level = state.threat.level
    return snap


def rl_summary(planner: PolicyPlanner | None) -> dict[str, object] | None:
    """Compact policy-planner snapshot for telemetry events."""
    if planner is None:
        return None
    return {
        "mode": planner.mode,
        "active": planner.active,
        "steps": planner.steps,
        "buffer": planner.buffer_len,
        "note": planner.reason,
    }


class PolicyPlanner:
    """Learned-policy driver over the scripted locomotion skills."""

    def __init__(
        self,
        cfg: RlConfig,
        locomotion: LocomotionController,
        *,
        checkpoint_path: Path,
        buffer_path: Path,
    ) -> None:
        self._cfg = cfg
        self._locomotion = locomotion
        self._checkpoint_path = Path(checkpoint_path)
        self._buffer_path = Path(buffer_path)
        self._policy: Policy | None = None
        self._reason = ""
        self._buffer = (
            TransitionBuffer(max_rows=cfg.buffer_max) if cfg.record else None
        )
        self._rng = np.random.default_rng()
        self._last: tuple[np.ndarray, int, bool] | None = None
        self._prev: GameState | None = None
        self._prev_action_at: float | None = None
        self._next_at = 0.0
        self._steps = 0
        self._closed = False
        if cfg.mode == "policy":
            self._load_policy()
        else:
            log.info(
                "rl: random exploration mode (%s recording)",
                "with" if self._buffer is not None else "without",
            )

    def _load_policy(self) -> None:
        try:
            policy = Policy.load(self._checkpoint_path)
        except FileNotFoundError:
            self._reason = f"no policy checkpoint at {self._checkpoint_path.name}"
            log.warning("rl: %s (train one with src.ai.rl.train)", self._reason)
            return
        except (OSError, ValueError) as exc:
            self._reason = f"bad policy checkpoint: {exc}"
            log.warning("rl: %s", self._reason)
            return
        if policy.dim_in != feature_dim() or policy.dim_out != n_actions():
            self._reason = (
                f"checkpoint dims {policy.dim_in}->{policy.dim_out} != "
                f"expected {feature_dim()}->{n_actions()}"
            )
            log.warning("rl: %s", self._reason)
            return
        self._policy = policy
        log.info(
            "rl: policy loaded from %s (hidden=%d, explore=%.2f)",
            self._checkpoint_path, policy.dim_hidden, self._cfg.explore,
        )

    @property
    def active(self) -> bool:
        if self._cfg.mode == "random":
            return True
        return self._policy is not None

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def mode(self) -> str:
        return self._cfg.mode

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def buffer_len(self) -> int:
        if self._buffer is None:
            return 0
        return len(self._buffer)

    def _choose(self, obs: np.ndarray) -> tuple[int, float]:
        uniform = 1.0 / n_actions()
        if self._policy is None:
            return int(self._rng.integers(n_actions())), uniform
        if self._rng.random() < self._cfg.explore:
            return int(self._rng.integers(n_actions())), uniform
        return self._policy.act(
            obs, temperature=self._cfg.temperature, rng=self._rng
        )

    def step(
        self,
        now: float,
        status: StatusTracker,
        frame: object | None = None,
        game_state: GameState | None = None,
    ) -> bool:
        """Choose and execute one scripted skill per action interval."""
        self._locomotion.tick()
        if game_state is None:
            return False
        if not self.active:
            status.update(
                rl_mode="", rl_action="", rl_prob=0.0, rl_steps=self._steps,
                rl_buffer=self.buffer_len, rl_note=self._reason,
            )
            return False
        if now < self._next_at:
            return True
        obs = featurize(game_state)
        index, prob = self._choose(obs)
        ok = execute(
            ACTIONS[index],
            self._locomotion,
            step_s=self._cfg.step_s,
            turn_step_deg=self._cfg.turn_step_deg,
        )
        if not ok:
            return True  # guard refused; retry next tick
        if self._buffer is not None and self._last is not None \
                and self._prev is not None:
            last_obs, last_index, last_boundary = self._last
            reward = reward_delta(self._prev, game_state, self._cfg.reward)
            self._buffer.add(last_obs, last_index, reward, boundary=last_boundary)
        boundary = False
        if self._buffer is not None:
            gap = (
                float("inf") if self._prev_action_at is None
                else now - self._prev_action_at
            )
            boundary = gap > self._cfg.boundary_gap_s
        self._last = (obs, index, boundary)
        self._prev = _snapshot(game_state)
        self._prev_action_at = now
        self._next_at = now + self._cfg.action_interval_s
        self._steps += 1
        status.update(
            goal="RL POLICY",
            action=f"rl:{ACTIONS[index]}",
            rl_mode=self._cfg.mode,
            rl_action=ACTIONS[index],
            rl_prob=round(prob, 3),
            rl_steps=self._steps,
            rl_buffer=self.buffer_len,
            rl_note="",
        )
        return True

    def stop(self) -> None:
        """Clear pending transition bookkeeping (pause / takeover / fault).

        The half-recorded action pair is dropped rather than rewarded across
        a pause, so the next action starts a fresh episode boundary. Cheap
        and idempotent - the loop calls it every tick while not RUNNING.
        """
        if self._last is None and self._prev is None and self._prev_action_at is None:
            return
        self._last = None
        self._prev = None
        self._prev_action_at = None
        self._next_at = 0.0
        log.debug("rl: cleared pending transition on stop")

    def close(self) -> Path | None:
        """Persist the recorded buffer (returns its path, if anything saved)."""
        if self._closed or self._buffer is None or len(self._buffer) == 0:
            return None
        self._closed = True
        path = self._buffer.save(self._buffer_path)
        log.info("rl: saved %d transitions to %s", len(self._buffer), path)
        return path
