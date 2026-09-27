"""Transition buffer for the RL scaffold (Phase 9).

A flat sequence of ``(observation, action, reward, boundary)`` rows saved as
``.npz``. ``boundary=True`` marks the start of a new episode so discounted
returns reset there - without it, returns bleed across sessions and the
REINFORCE advantage becomes position noise. Rows from multiple sessions
concatenate; within an episode returns are computed over the whole segment
(a known limitation of this best-effort scaffold).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TransitionBuffer:
    """In-memory transition store with ``.npz`` round-trip.

    ``max_rows`` caps memory on long runs: the oldest row is dropped and the
    new head is promoted to an episode boundary.
    """

    def __init__(self, *, max_rows: int | None = None) -> None:
        self._obs: list[np.ndarray] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._boundary: list[bool] = []
        self._max_rows = max_rows

    def __len__(self) -> int:
        return len(self._actions)

    @property
    def feature_dim(self) -> int | None:
        if not self._obs:
            return None
        return int(self._obs[0].shape[0])

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        *,
        boundary: bool = False,
    ) -> None:
        row = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self._obs and row.shape != self._obs[0].shape:
            raise ValueError(
                f"observation shape {row.shape} != buffer shape {self._obs[0].shape}"
            )
        if action < 0:
            raise ValueError(f"action index must be >= 0, got {action}")
        if not self._obs:
            boundary = True  # the first row always starts an episode
        self._obs.append(row)
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._boundary.append(bool(boundary))
        if self._max_rows is not None and len(self._actions) > self._max_rows:
            self._drop_oldest()

    def _drop_oldest(self) -> None:
        self._obs.pop(0)
        self._actions.pop(0)
        self._rewards.pop(0)
        self._boundary.pop(0)
        if self._boundary:
            self._boundary[0] = True  # the new head starts an episode

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._obs:
            return (
                np.zeros((0, 0), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
            )
        obs = np.stack(self._obs).astype(np.float32, copy=False)
        actions = np.asarray(self._actions, dtype=np.int64)
        rewards = np.asarray(self._rewards, dtype=np.float32)
        return obs, actions, rewards

    def boundary_flags(self) -> np.ndarray:
        return np.asarray(self._boundary, dtype=bool)

    def discounted_returns(self, gamma: float) -> np.ndarray:
        """Monte-Carlo returns, resetting at every episode boundary."""
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        _, _, rewards = self.as_arrays()
        returns = np.zeros_like(rewards, dtype=np.float64)
        running = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            returns[i] = rewards[i] + running
            running = 0.0 if self._boundary[i] else gamma * float(returns[i])
        return returns

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        obs, actions, rewards = self.as_arrays()
        np.savez(
            out, obs=obs, actions=actions, rewards=rewards,
            boundary=self.boundary_flags(),
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> TransitionBuffer:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"buffer not found: {src}")
        buffer = cls()
        with np.load(src, allow_pickle=False) as data:
            required = {"obs", "actions", "rewards"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"buffer missing arrays: {sorted(missing)}")
            obs = np.asarray(data["obs"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.int64)
            rewards = np.asarray(data["rewards"], dtype=np.float32)
            boundary = (
                np.asarray(data["boundary"], dtype=bool)
                if "boundary" in data.files
                else np.zeros(actions.shape[0], dtype=bool)
            )
        if obs.ndim != 2 or actions.shape != (obs.shape[0],) \
                or rewards.shape != (obs.shape[0],) \
                or boundary.shape != (obs.shape[0],):
            raise ValueError("buffer arrays have inconsistent shapes")
        for row, action, reward, flag in zip(
            obs, actions, rewards, boundary, strict=True
        ):
            buffer.add(row, int(action), float(reward), boundary=bool(flag))
        return buffer
