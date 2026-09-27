"""Discrete action space mapped onto the scripted locomotion skills (Phase 9).

The policy never touches raw keys: every action reuses the Phase 4
locomotion planner (clamped, focus-gated, released by ``stop()``).
"""

from __future__ import annotations

from src.control.locomotion import LocomotionController

ACTIONS: tuple[str, ...] = (
    "noop",
    "forward",
    "back",
    "strafe_left",
    "strafe_right",
    "turn_left",
    "turn_right",
    "sprint",
)


def n_actions() -> int:
    return len(ACTIONS)


def action_index(name: str) -> int:
    try:
        return ACTIONS.index(name)
    except ValueError:
        raise KeyError(f"unknown action {name!r}") from None


def execute(
    action: str,
    locomotion: LocomotionController,
    *,
    step_s: float,
    turn_step_deg: float,
) -> bool:
    """Run one scripted skill; ``False`` only when the input guard refused it."""
    if action == "noop":
        return True
    if action == "sprint":
        return locomotion.move("forward", step_s, sprint=True)
    if action == "turn_left":
        return locomotion.turn(-turn_step_deg)
    if action == "turn_right":
        return locomotion.turn(turn_step_deg)
    if action in ("forward", "back", "strafe_left", "strafe_right"):
        return locomotion.move(action, step_s)
    raise KeyError(f"unknown action {action!r}")
