"""Phase 7: minimap threat detection, threat assessment, self-defense combat."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.ai.threat import ThreatAssessor
from src.config import AppConfig, CombatConfig, ConfigError, VisionConfig
from src.control.locomotion import LocomotionController
from src.planning.combat import CombatPlanner
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState
from src.vision.perception import Perception, threat_summary
from src.vision.world import WorldConfig, _count_enemy_dots, _read_minimap
from tests.test_hud import _boxes, _frame

FRAME_W, FRAME_H = 1920, 1080


class FakeInput:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.fail_buttons: set[str] = set()

    def key_down(self, key: str) -> bool:
        self.events.append(("down", key))
        return True

    def key_up(self, key: str) -> bool:
        self.events.append(("up", key))
        return True

    def mouse_move(self, dx: int, dy: int = 0) -> bool:
        self.events.append(("mouse", dx, dy))
        return True

    def press(self, key: str, hold_ms: int | None = None) -> bool:
        self.events.append(("press", key))
        return True

    def mouse_down(self, button: str = "left") -> bool:
        if button in self.fail_buttons:
            return False
        self.events.append(("down_btn", button))
        return True

    def mouse_up(self, button: str = "left") -> bool:
        self.events.append(("up_btn", button))
        return True


def make_combat(
    **overrides: object,
) -> tuple[CombatPlanner, FakeInput, StatusTracker, LocomotionController]:
    cfg = CombatConfig(**overrides)  # type: ignore[arg-type]
    fake = FakeInput()
    loco = LocomotionController(AppConfig().control, fake)
    planner = CombatPlanner(cfg, loco, fake)  # type: ignore[arg-type]
    return planner, fake, StatusTracker(phase=7), loco


def threat_state(
    gs: GameState, *, wanted: int = 0, enemies: int | None = 0,
    fire: bool = False, offset: list[float] | None = None,
) -> GameState:
    gs.threat.wanted_level = wanted
    gs.threat.enemies_detected = enemies
    gs.threat.incoming_fire = fire
    gs.threat.enemy_offset = offset
    return gs


def events_of(fake: FakeInput, kind: str) -> list[tuple[object, ...]]:
    return [e for e in fake.events if e[0] == kind]


class TestMinimapEnemyDots:
    def test_red_blips_counted(self) -> None:
        crop = np.full((80, 80, 3), 20, dtype=np.uint8)
        crop[10:16, 10:16] = (0, 0, 255)
        crop[60:66, 40:46] = (0, 0, 255)
        count, offset = _count_enemy_dots(crop, WorldConfig())
        assert count == 2
        assert offset is not None

    def test_offset_points_to_nearest_blip(self) -> None:
        crop = np.full((80, 80, 3), 20, dtype=np.uint8)
        crop[36:44, 60:68] = (0, 0, 255)  # right of centre
        count, offset = _count_enemy_dots(crop, WorldConfig())
        assert count == 1
        assert offset is not None
        assert offset[0] > 0.2
        assert abs(offset[1]) < 0.1

    def test_no_red_returns_zero(self) -> None:
        crop = np.full((80, 80, 3), 20, dtype=np.uint8)
        assert _count_enemy_dots(crop, WorldConfig()) == (0, None)

    def test_tiny_crop_returns_unknown(self) -> None:
        crop = np.full((8, 8, 3), 20, dtype=np.uint8)
        assert _count_enemy_dots(crop, WorldConfig()) == (None, None)

    def test_low_chroma_red_ignored(self) -> None:
        crop = np.full((80, 80, 3), 20, dtype=np.uint8)
        crop[30:40, 30:40] = (140, 130, 160)  # brownish, not a blip
        assert _count_enemy_dots(crop, WorldConfig()) == (0, None)

    def test_oversized_blob_ignored(self) -> None:
        crop = np.full((80, 80, 3), 20, dtype=np.uint8)
        crop[10:70, 10:70] = (0, 0, 255)
        assert _count_enemy_dots(crop, WorldConfig()) == (0, None)

    def test_minimap_reading_exposes_enemy_dots(self) -> None:
        crop = np.full((80, 80, 3), 20, dtype=np.uint8)
        crop[12:18, 12:18] = (0, 0, 255)
        reading = _read_minimap(crop, WorldConfig())
        assert reading.enemy_dots == 1
        assert reading.enemy_offset is not None

    def test_perception_wires_dots_into_threat(self) -> None:
        cfg = VisionConfig()
        perception = Perception(cfg, ocr=None)
        frame = _frame()
        x, y, w, h = _boxes(cfg.hud)["minimap"]
        cx, cy = x + w // 3, y + h // 3
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        state = GameState()
        perception.apply_to_state(state, perception.process(frame, 1))
        assert state.threat.enemies_detected == 1
        assert state.threat.enemy_offset is not None


class TestThreatAssessor:
    def make(self, **overrides: object) -> tuple[ThreatAssessor, GameState]:
        cfg = CombatConfig(**overrides)  # type: ignore[arg-type]
        return ThreatAssessor(cfg), GameState()

    def test_health_drop_sets_incoming_fire(self) -> None:
        assessor, gs = self.make(health_drop_frac=0.05, health_drop_window_s=2.0)
        gs.player.health = 1.0
        assessor.assess(100.0, gs)
        gs.player.health = 0.8
        assessor.assess(100.5, gs)
        assert gs.threat.incoming_fire is True
        assert gs.threat.level == "under_fire"

    def test_steady_health_is_not_fire(self) -> None:
        assessor, gs = self.make()
        gs.player.health = 1.0
        assessor.assess(100.0, gs)
        assessor.assess(100.5, gs)
        assert gs.threat.incoming_fire is False
        assert gs.threat.level == "none"

    def test_incoming_fire_decays_after_hold(self) -> None:
        assessor, gs = self.make(incoming_fire_hold_s=2.0, health_drop_window_s=2.0)
        gs.player.health = 1.0
        assessor.assess(100.0, gs)
        gs.player.health = 0.5
        assessor.assess(100.5, gs)
        assert gs.threat.incoming_fire is True
        gs.player.health = 0.5
        assessor.assess(103.0, gs)
        assert gs.threat.incoming_fire is False

    def test_old_drop_outside_window_not_counted(self) -> None:
        assessor, gs = self.make(health_drop_window_s=1.0, incoming_fire_hold_s=0.1)
        gs.player.health = 1.0
        assessor.assess(100.0, gs)
        gs.player.health = 0.5
        assessor.assess(102.0, gs)  # drop happened outside the window
        assert gs.threat.incoming_fire is False

    def test_level_ladder(self) -> None:
        assessor, gs = self.make()
        assessor.assess(10.0, gs)
        assert gs.threat.level == "none"
        threat_state(gs, enemies=2)
        assessor.assess(10.1, gs)
        assert gs.threat.level == "enemy"
        threat_state(gs, wanted=2, enemies=2)
        assessor.assess(10.2, gs)
        assert gs.threat.level == "wanted"

    def test_engage_flag_can_disable_fire_detection(self) -> None:
        assessor, gs = self.make(engage_on_incoming_fire=False)
        gs.player.health = 1.0
        assessor.assess(100.0, gs)
        gs.player.health = 0.5
        assessor.assess(100.5, gs)
        assert gs.threat.incoming_fire is False
        assert gs.threat.level != "under_fire"

    def test_combat_confidence_boosted_by_signal(self) -> None:
        assessor, gs = self.make()
        threat_state(gs, enemies=1)
        assessor.assess(10.0, gs)
        assert gs.confidence.combat >= 0.5

    def test_summary_shape(self) -> None:
        summary = threat_summary(GameState())
        assert set(summary) == {"level", "enemies", "wanted", "incoming_fire"}


class TestCombatPlanner:
    def test_no_threat_does_nothing(self) -> None:
        planner, fake, status, _ = make_combat()
        gs = threat_state(GameState())
        assert planner.step(100.0, status, None, gs) is None
        assert planner.engaged is False
        assert fake.events == []

    def test_wanted_stars_engage_and_fire(self) -> None:
        planner, fake, status, _ = make_combat(burst_s=0.3, burst_interval_s=1.0)
        gs = threat_state(GameState(), wanted=1)
        gs.player.weapon_ammo = 6
        label = planner.step(100.0, status, None, gs)
        assert planner.engaged is True
        assert label == "combat:fire"
        assert ("down_btn", "left") in fake.events
        assert status.snapshot().goal == "COMBAT (phase 7)"

    def test_burst_deadline_releases_button(self) -> None:
        planner, fake, status, _ = make_combat(burst_s=0.3, burst_interval_s=0.05)
        gs = threat_state(GameState(), wanted=1)
        planner.step(100.0, status, None, gs)
        planner.step(100.4, status, None, gs)  # past burst deadline
        assert ("up_btn", "left") in fake.events

    def test_burst_interval_limits_rate(self) -> None:
        planner, fake, status, _ = make_combat(burst_s=0.05, burst_interval_s=10.0)
        gs = threat_state(GameState(), wanted=1)
        planner.step(100.0, status, None, gs)
        planner.step(100.5, status, None, gs)
        planner.step(100.6, status, None, gs)
        downs = events_of(fake, "down_btn")
        assert len(downs) == 1

    def test_zero_ammo_never_fires(self) -> None:
        planner, fake, status, _ = make_combat()
        gs = threat_state(GameState(), wanted=1)
        gs.player.weapon_ammo = 0
        assert planner.step(100.0, status, None, gs) == "combat:cover"
        assert events_of(fake, "down_btn") == []
        assert planner.engaged is True

    def test_aims_toward_nearest_blip(self) -> None:
        planner, fake, status, _ = make_combat(
            aim_interval_s=0.0, aim_dead_zone_deg=1.0
        )
        gs = threat_state(GameState(), enemies=2, offset=[0.4, 0.0])
        gs.player.weapon_ammo = 0  # isolate the aim path from firing
        label = planner.step(100.0, status, None, gs)
        assert label == "combat:aim"
        moves = events_of(fake, "mouse")
        assert moves and moves[0][1] > 0

    def test_aim_dead_zone_skips_small_error(self) -> None:
        planner, fake, status, _ = make_combat(
            aim_interval_s=0.0, aim_dead_zone_deg=30.0
        )
        gs = threat_state(GameState(), enemies=1, offset=[0.01, -0.4])
        planner.step(100.0, status, None, gs)
        assert events_of(fake, "mouse") == []

    def test_low_health_retreats_and_releases_fire(self) -> None:
        planner, fake, status, _ = make_combat(retreat_health_frac=0.3)
        gs = threat_state(GameState(), wanted=1)
        gs.player.health = 0.2
        gs.player.weapon_ammo = 6
        planner.step(100.0, status, None, gs)
        planner.step(100.1, status, None, gs)
        assert status.snapshot().action == "combat:retreat"
        assert ("down", "s") in fake.events
        assert events_of(fake, "down_btn") == []

    def test_incoming_fire_engages(self) -> None:
        planner, fake, status, _ = make_combat()
        gs = threat_state(GameState(), fire=True)
        planner.step(100.0, status, None, gs)
        assert planner.engaged is True
        assert status.snapshot().goal == "COMBAT (phase 7)"

    def test_disengages_after_clear_window(self) -> None:
        planner, fake, status, _ = make_combat(disengage_clear_s=1.0, burst_s=0.3)
        gs = threat_state(GameState(), wanted=1)
        planner.step(100.0, status, None, gs)
        assert planner.engaged is True
        calm = threat_state(GameState())
        planner.step(100.4, status, None, calm)  # releases the burst first
        assert planner.engaged is True
        planner.step(101.0, status, None, calm)
        planner.step(101.6, status, None, calm)
        assert planner.engaged is False
        assert ("up_btn", "left") in fake.events

    def test_refused_mouse_down_does_not_count_burst(self) -> None:
        planner, fake, status, _ = make_combat()
        fake.fail_buttons.add("left")
        gs = threat_state(GameState(), wanted=1)
        planner.step(100.0, status, None, gs)
        assert planner.bursts == 0

    def test_stop_releases_everything(self) -> None:
        planner, fake, status, _ = make_combat(burst_s=10.0)
        gs = threat_state(GameState(), wanted=1)
        planner.step(100.0, status, None, gs)
        planner.stop()
        assert ("up_btn", "left") in fake.events


class TestCombatConfigValidation:
    def test_default_config_valid(self) -> None:
        AppConfig().validate()

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("engage_min_wanted", 0, "engage_min_wanted"),
            ("engage_enemies", 0, "engage_enemies"),
            ("health_drop_frac", 1.5, "health_drop_frac"),
            ("retreat_health_frac", -0.1, "retreat_health_frac"),
            ("aim_dead_zone_deg", -1.0, "aim_dead_zone_deg"),
            ("fire_button", "trigger", "fire_button"),
            ("burst_s", 0.0, "burst_s"),
            ("disengage_clear_s", -1.0, "disengage_clear_s"),
        ],
    )
    def test_invalid_values_rejected(
        self, field: str, value: object, match: str
    ) -> None:
        cfg = AppConfig()
        setattr(cfg.combat, field, value)
        with pytest.raises(ConfigError, match=match):
            cfg.validate()

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("enemy_red_min", 300, "enemy_red_min"),
            ("enemy_red_chroma", -5, "enemy_red_chroma"),
            ("enemy_min_blob_px", 0, "enemy_min_blob_px"),
            ("enemy_max_dots", 0, "enemy_max_dots"),
        ],
    )
    def test_world_enemy_params_invalid(
        self, field: str, value: object, match: str
    ) -> None:
        cfg = AppConfig()
        setattr(cfg.vision.world, field, value)
        with pytest.raises(ConfigError, match=match):
            cfg.validate()
