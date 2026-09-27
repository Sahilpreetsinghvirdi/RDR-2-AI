"""Fixed observation featurizer for the RL scaffold (Phase 9).

Maps :class:`GameState` onto a deterministic, versioned float vector.
Unknown fields become ``0.0`` with a presence flag of ``0.0`` - they are
never guessed. The layout is the contract between recording and training:
``FEATURE_NAMES`` must not change order between a recorded buffer and the
policy trained from it.
"""

from __future__ import annotations

import numpy as np

from src.state.game_state import GameState

ENCOUNTER_KEYS: tuple[str, ...] = (
    "greet", "antagonize", "talk", "rob", "mount", "skin", "loot",
)
TIME_KEYS: tuple[str, ...] = ("day", "dusk", "night")
THREAT_KEYS: tuple[str, ...] = ("none", "enemy", "wanted", "under_fire")

ENV_KEYS: tuple[str, ...] = (
    "road", "water", "town", "building", "wilderness", "cover",
)

FEATURE_NAMES: tuple[str, ...] = (
    "health", "health_present",
    "stamina", "stamina_present",
    "dead_eye", "dead_eye_present",
    *(f"threat_{key}" for key in THREAT_KEYS),
    "enemies", "enemies_present",
    "wanted",
    "incoming_fire",
    "prompt_present",
    "dialogue_active",
    *(f"enc_{key}" for key in ENCOUNTER_KEYS),
    *(f"env_{key}" for key in ENV_KEYS),
    "env_present",
    *(f"time_{key}" for key in TIME_KEYS),
    "horse_detected",
    "mounted", "mounted_present",
)

FEATURE_DIM = len(FEATURE_NAMES)


def feature_dim() -> int:
    return FEATURE_DIM


def _norm(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(min(1.0, max(0.0, value)))


def featurize(state: GameState) -> np.ndarray:
    """Project *state* onto the fixed observation vector (float32)."""
    values: list[float] = []

    for attr in ("health", "stamina", "dead_eye"):
        core = getattr(state.player, attr, None)
        values.append(_norm(core))
        values.append(1.0 if core is not None else 0.0)

    level = state.threat.level
    for key in THREAT_KEYS:
        values.append(1.0 if level == key else 0.0)

    enemies = state.threat.enemies_detected
    values.append(0.0 if enemies is None else min(float(enemies), 10.0) / 10.0)
    values.append(1.0 if enemies is not None else 0.0)
    values.append(min(float(state.threat.wanted_level), 5.0) / 5.0)
    values.append(1.0 if state.threat.incoming_fire else 0.0)

    values.append(1.0 if state.mission.prompt_text else 0.0)
    values.append(1.0 if state.dialogue.active else 0.0)
    encounter = state.dialogue.encounter
    for key in ENCOUNTER_KEYS:
        values.append(1.0 if encounter == key else 0.0)

    env = state.environment
    env_flags = (
        env.road_detected, env.water_detected, env.town_detected,
        env.building_detected, env.wilderness, env.cover_available,
    )
    for flag in env_flags:
        values.append(1.0 if flag else 0.0)
    values.append(1.0 if state.confidence.navigation > 0.0 else 0.0)

    tod = env.time_of_day
    for key in TIME_KEYS:
        values.append(1.0 if tod == key else 0.0)

    values.append(1.0 if state.horse.detected else 0.0)
    mounted = state.player.mounted
    values.append(1.0 if mounted else 0.0)
    values.append(1.0 if mounted is not None else 0.0)

    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (FEATURE_DIM,):
        raise RuntimeError(
            f"featurizer produced {arr.shape}, expected ({FEATURE_DIM},)"
        )
    return arr
