"""Reward shaping from known state deltas (Phase 9).

Core improvements add weighted deltas; the current threat level is a
per-step penalty. Unknown cores contribute nothing - the reward never
invents data.
"""

from __future__ import annotations

from src.config import RewardConfig
from src.state.game_state import GameState

THREAT_VALUES: dict[str, int] = {
    "none": 0, "enemy": 1, "wanted": 2, "under_fire": 3,
}

_CORE_WEIGHTS = (("health", "health"), ("stamina", "stamina"),
                 ("dead_eye", "dead_eye"))


def threat_value(level: str) -> int:
    return THREAT_VALUES.get(level, 0)


def reward_delta(prev: GameState, cur: GameState, cfg: RewardConfig) -> float:
    """Scalar reward for the interval between the two snapshots."""
    reward = 0.0
    for attr, weight_name in _CORE_WEIGHTS:
        before = getattr(prev.player, attr, None)
        after = getattr(cur.player, attr, None)
        if before is None or after is None:
            continue
        reward += getattr(cfg, weight_name) * (after - before)
    reward -= cfg.threat * threat_value(cur.threat.level)
    return float(reward)
