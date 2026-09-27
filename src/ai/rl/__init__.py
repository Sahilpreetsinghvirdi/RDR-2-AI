"""Learned/optimized policies (Phase 9): best-effort research scaffold.

Record transitions while acting -> train offline with numpy -> act from the
checkpoint. Disabled by default; safety guards are unchanged.
"""

from src.ai.rl.buffer import TransitionBuffer
from src.ai.rl.features import FEATURE_NAMES, featurize
from src.ai.rl.planner import PolicyPlanner, rl_summary
from src.ai.rl.policy import Policy

__all__ = [
    "FEATURE_NAMES",
    "Policy",
    "PolicyPlanner",
    "TransitionBuffer",
    "featurize",
    "rl_summary",
]
