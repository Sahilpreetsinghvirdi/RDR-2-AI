"""Prompt building and decision parsing for the vision brain.

The model chooses exactly one skill action per decision and explains it in
one line. Parsing is defensive: code fences are stripped, the first JSON
object wins, and anything unparseable or out of vocabulary falls back to a
harmless ``noop`` so a confused model can never inject input.
"""

from __future__ import annotations

import json

ACTIONS = (
    "noop",
    "forward",
    "back",
    "strafe_left",
    "strafe_right",
    "turn_left",
    "turn_right",
    "sprint",
    "interact",
    "whistle",
)


def build_prompt(summary: dict[str, object]) -> str:
    """Render the game-state summary into a strict single-action prompt."""
    lines = [
        "You are playing Red Dead Redemption 2 as Arthur Morgan.",
        "You see the current gameplay screenshot. Read this state, then act:",
    ]
    for key in (
        "threat", "wanted", "enemies", "health", "stamina", "dead_eye",
        "objective", "prompt", "dialogue", "ammo",
    ):
        value = summary.get(key)
        if value is None or value == "":
            continue
        lines.append(f"- {key}: {value}")
    lines.extend([
        "Choose EXACTLY one action from: " + ", ".join(ACTIONS) + ".",
        "Guidance: ride or walk toward mission markers and objectives;",
        "face what you approach before moving; use interact on prompts;",
        "whistle only to call your horse; noop when waiting or unsure.",
        'Reply with JSON only, no other text: {"action": "...", "reason": "..."}',
    ])
    return "\n".join(lines)


def parse_decision(text: str) -> tuple[str, str, bool]:
    """Parse a model reply into (action, reason, clean).

    *clean* is False when the reply needed repair or fell back to ``noop``.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return "noop", "unparseable reply", False
    try:
        data = json.loads(cleaned[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return "noop", "invalid JSON", False
    if not isinstance(data, dict):
        return "noop", "reply was not an object", False
    action = data.get("action")
    reason = data.get("reason", "")
    if action not in ACTIONS:
        return "noop", f"unknown action {action!r}", False
    if not isinstance(reason, str):
        reason = str(reason)
    return str(action), reason.strip()[:160], True
