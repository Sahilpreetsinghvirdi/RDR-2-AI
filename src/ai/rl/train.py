"""Offline training for the RL scaffold (Phase 9).

Usage::

    .\\.venv\\Scripts\\python.exe -m src.ai.rl.train \\
        --data learning/buffer.npz --out learning/policy.npz

Methods:
    ``reinforce`` (default) - policy gradient on recorded rewards
        (REINFORCE with a mean baseline, std-normalized advantages and an
        optional entropy bonus). High-variance by nature: episode boundaries
        come from ``rl.boundary_gap_s`` so returns do not bleed across
        sessions.
    ``bc`` - behavior cloning on the recorded action labels (for data that
        came from scripted demonstrations rather than random exploration).

Pure numpy gradients over the small MLP policy; no GPU frameworks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.ai.rl.actions import n_actions
from src.ai.rl.buffer import TransitionBuffer
from src.ai.rl.policy import Policy, softmax


def _forward(policy: Policy, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(obs @ policy.w1.T + policy.b1)
    logits = hidden @ policy.w2.T + policy.b2
    return hidden, logits


def _backward(
    policy: Policy,
    obs: np.ndarray,
    hidden: np.ndarray,
    dlogits: np.ndarray,
    lr: float,
) -> None:
    n = obs.shape[0]
    dlogits = dlogits / n
    dw2 = dlogits.T @ hidden
    db2 = dlogits.sum(axis=0)
    dhidden = dlogits @ policy.w2
    dz = dhidden * (1.0 - hidden * hidden)
    dw1 = dz.T @ obs
    db1 = dz.sum(axis=0)
    policy.w2 -= lr * dw2
    policy.b2 -= lr * db2
    policy.w1 -= lr * dw1
    policy.b1 -= lr * db1


def train_bc(
    policy: Policy,
    buffer: TransitionBuffer,
    *,
    epochs: int,
    lr: float,
) -> list[float]:
    """Cross-entropy on recorded actions; returns per-epoch mean losses."""
    obs, actions, _ = buffer.as_arrays()
    if obs.shape[0] == 0:
        raise ValueError("buffer is empty - record transitions first")
    if obs.shape[1] != policy.dim_in:
        raise ValueError(
            f"buffer features {obs.shape[1]} != policy input {policy.dim_in}"
        )
    losses: list[float] = []
    rows = np.arange(obs.shape[0])
    for _ in range(epochs):
        hidden, logits = _forward(policy, obs)
        probs = np.vstack([softmax(row) for row in logits])
        chosen = np.clip(probs[rows, actions], 1e-12, 1.0)
        losses.append(float(-np.log(chosen).mean()))
        dlogits = probs.copy()
        dlogits[rows, actions] -= 1.0
        _backward(policy, obs, hidden, dlogits, lr)
    return losses


def train_reinforce(
    policy: Policy,
    buffer: TransitionBuffer,
    *,
    epochs: int,
    lr: float,
    gamma: float,
    entropy: float,
) -> list[float]:
    """REINFORCE with a mean baseline; returns per-epoch mean losses."""
    obs, actions, _ = buffer.as_arrays()
    if obs.shape[0] == 0:
        raise ValueError("buffer is empty - record transitions first")
    if obs.shape[1] != policy.dim_in:
        raise ValueError(
            f"buffer features {obs.shape[1]} != policy input {policy.dim_in}"
        )
    returns = buffer.discounted_returns(gamma)
    adv = returns - returns.mean()
    std = float(adv.std())
    if std > 1e-8:
        adv = adv / std
    losses: list[float] = []
    rows = np.arange(obs.shape[0])
    for _ in range(epochs):
        hidden, logits = _forward(policy, obs)
        probs = np.vstack([softmax(row) for row in logits])
        chosen = np.clip(probs[rows, actions], 1e-12, 1.0)
        logp = np.log(chosen)
        entropy_vals = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        losses.append(float(-(logp * adv).mean() - entropy * entropy_vals.mean()))
        dlogits = adv[:, None] * probs
        dlogits[rows, actions] -= adv
        if entropy > 0.0:
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            cross = np.sum(probs * log_probs, axis=1, keepdims=True)
            dlogits -= entropy * probs * (cross - log_probs)
        _backward(policy, obs, hidden, dlogits, lr)
    return losses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.ai.rl.train",
        description="Train an RDR2 AI policy from a recorded transition buffer.",
    )
    parser.add_argument("--data", required=True, help="transition buffer .npz")
    parser.add_argument("--out", required=True, help="policy checkpoint .npz to write")
    parser.add_argument(
        "--method", choices=["reinforce", "bc"], default="reinforce",
        help="reinforce (rewards) or bc (action labels)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--resume", action="store_true",
        help="continue training the existing --out checkpoint",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1 or args.hidden < 1:
        print("train error: --epochs and --hidden must be >= 1", file=sys.stderr)
        return 2
    if args.lr <= 0 or not 0.0 < args.gamma <= 1.0 or args.entropy < 0:
        print("train error: --lr > 0, --gamma in (0,1], --entropy >= 0", file=sys.stderr)
        return 2
    try:
        buffer = TransitionBuffer.load(args.data)
    except (OSError, ValueError) as exc:
        print(f"train error: {exc}", file=sys.stderr)
        return 2
    if len(buffer) == 0:
        print("train error: buffer has no transitions", file=sys.stderr)
        return 2
    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    try:
        if args.resume and out_path.exists():
            policy = Policy.load(out_path)
            if policy.dim_in != buffer.feature_dim or policy.dim_out != n_actions():
                print(
                    "train error: resumed checkpoint dims do not match buffer/actions",
                    file=sys.stderr,
                )
                return 2
        else:
            policy = Policy(buffer.feature_dim or 0, args.hidden, n_actions(), rng=rng)
        if args.method == "bc":
            losses = train_bc(policy, buffer, epochs=args.epochs, lr=args.lr)
        else:
            losses = train_reinforce(
                policy, buffer, epochs=args.epochs, lr=args.lr,
                gamma=args.gamma, entropy=args.entropy,
            )
        policy.save(out_path)
    except (OSError, ValueError) as exc:
        print(f"train error: {exc}", file=sys.stderr)
        return 2
    print(
        f"trained {args.method}: {len(buffer)} transitions, {args.epochs} epochs, "
        f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    print(f"saved policy: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
