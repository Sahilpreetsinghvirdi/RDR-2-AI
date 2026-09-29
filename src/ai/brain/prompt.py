"""Prompt building and decision parsing for the vision brain.

The model chooses exactly one skill action per decision and explains it in
one line. Parsing is defensive: code fences are stripped, the first JSON
object wins, and anything unparseable or out of vocabulary falls back to a
harmless ``noop`` so a confused model can never inject input.
"""

from __future__ import annotations

import json
import re

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

ACTION_ALIASES = {
    "walk": "forward",
    "go": "forward",
    "run": "sprint",
    "left": "turn_left",
    "right": "turn_right",
    "stop": "noop",
    "wait": "noop",
}


def _clean_action(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    name = raw.strip().lower()
    if name in ACTIONS:
        return name
    return ACTION_ALIASES.get(name)


def build_prompt(summary: dict[str, object]) -> str:
    """Render the game-state summary into a strict single-action prompt."""
    lines = [
        "You are playing Red Dead Redemption 2 as Arthur Morgan.",
        "You see the current gameplay screenshot. Read this state, then act:",
    ]
    for key in (
        "threat", "wanted", "enemies", "health", "stamina", "dead_eye",
        "objective", "prompt", "dialogue", "ammo", "previous",
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
        "Do not undo your previous action unless the scene changed -",
        "keep riding or walking in the same direction across decisions.",
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
    if start >= 0 and end > start:
        try:
            data = json.loads(cleaned[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            data = None
        if isinstance(data, dict):
            action = _clean_action(data.get("action"))
            reason = data.get("reason", "")
            if action is not None:
                if not isinstance(reason, str):
                    reason = str(reason)
                return action, reason.strip()[:160], True
    fallback = re.search(r'"action"\s*:\s*"([A-Za-z_]+)"', cleaned)
    if fallback is not None:
        action = _clean_action(fallback.group(1))
        if action is not None:
            reason = re.search(r'"reason"\s*:\s*"([^"]{0,160})', cleaned)
            return (
                action,
                reason.group(1).strip() if reason else "repaired reply",
                False,
            )
    return "noop", "unparseable reply", False
