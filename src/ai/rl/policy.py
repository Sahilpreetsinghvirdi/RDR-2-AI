"""Tiny numpy MLP policy: forward pass, sampling, save/load (Phase 9).

Best-effort research scaffold: one hidden layer (tanh) trained offline by
``src.ai.rl.train`` (behavior cloning or REINFORCE). No GPU frameworks - the
whole policy is four arrays in an ``.npz`` file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class Policy:
    """Input -> tanh hidden -> logits over :data:`src.ai.rl.actions.ACTIONS`."""

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        *,
        rng: np.random.Generator | None = None,
        init_scale: float = 0.1,
    ) -> None:
        if dim_in < 1 or dim_hidden < 1 or dim_out < 1:
            raise ValueError("policy dimensions must be >= 1")
        gen = rng if rng is not None else np.random.default_rng()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.w1 = gen.normal(0.0, init_scale, (dim_hidden, dim_in))
        self.b1 = np.zeros(dim_hidden)
        self.w2 = gen.normal(0.0, init_scale, (dim_out, dim_hidden))
        self.b2 = np.zeros(dim_out)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        hidden = np.tanh(self.w1 @ obs + self.b1)
        return self.w2 @ hidden + self.b2

    def probs(self, obs: np.ndarray) -> np.ndarray:
        return softmax(self.logits(obs))

    def act(
        self,
        obs: np.ndarray,
        *,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float]:
        """Sample an action; returns ``(index, probability_of_chosen)``."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        probs = softmax(self.logits(obs) / temperature)
        gen = rng if rng is not None else np.random.default_rng()
        index = int(gen.choice(probs.size, p=probs))
        return index, float(probs[index])

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            version=FORMAT_VERSION,
            w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"policy checkpoint not found: {src}")
        with np.load(src, allow_pickle=False) as data:
            required = {"version", "w1", "b1", "w2", "b2"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"policy checkpoint missing arrays: {sorted(missing)}")
            version = int(data["version"])
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"policy checkpoint version {version}, expected {FORMAT_VERSION}"
                )
            w1 = np.asarray(data["w1"], dtype=np.float64)
            b1 = np.asarray(data["b1"], dtype=np.float64)
            w2 = np.asarray(data["w2"], dtype=np.float64)
            b2 = np.asarray(data["b2"], dtype=np.float64)
        if w1.ndim != 2 or w2.ndim != 2:
            raise ValueError("policy checkpoint arrays have wrong ranks")
        if b1.shape != (w1.shape[0],) or b2.shape != (w2.shape[0],):
            raise ValueError("policy checkpoint bias shapes do not match weights")
        if w2.shape[1] != w1.shape[0]:
            raise ValueError("policy checkpoint layer sizes do not match")
        policy = cls(w1.shape[1], w1.shape[0], w2.shape[0])
        policy.w1, policy.b1, policy.w2, policy.b2 = w1, b1, w2, b2
        return policy
