"""World state tests (Phase 3): minimap, sky, weapon, horse, GameState wiring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import AppConfig, ConfigError, HudConfig, VisionConfig, WorldConfig
from src.state.agent_status import AgentStatus
from src.state.game_state import GameState
from src.ui.debug_overlay import build_lines
from src.vision.hud import scale_regions
from src.vision.ocr import OcrEngine
from src.vision.perception import Perception, world_summary
from src.vision.world import WorldReader
from tests.test_hud import _boxes, _frame, draw_gauge

FRAME_W, FRAME_H = 1920, 1080


def _world_boxes(cfg: WorldConfig | None = None) -> dict[str, tuple[int, int, int, int]]:
    return scale_regions((cfg or WorldConfig()).regions, FRAME_W, FRAME_H)


class DigitOcr(OcrEngine):
    name = "digit"
    available = True

    def __init__(self, text: str = "24/120") -> None:
        self.text = text
        self.calls = 0

    def read(self, image: np.ndarray) -> tuple[str, float]:
        self.calls += 1
        return self.text, 0.9


def _reader() -> WorldReader:
    return WorldReader(WorldConfig(), HudConfig())


def _minimap_box() -> tuple[int, int, int, int]:
    return _boxes(HudConfig())["minimap"]


def test_world_disabled_returns_skipped() -> None:
    reading = WorldReader(WorldConfig(enabled=False), HudConfig()).process(
        _frame(), _minimap_box()
    )
    assert reading.skipped
    assert reading.sky is None and reading.weapon is None
    cfg = VisionConfig()
    cfg.world.enabled = False
    pres = Perception(cfg, ocr=None).process(_frame(), 1)
    assert pres.world.skipped
    assert world_summary(pres) == {"enabled": False}


def test_sky_night_on_dark_frame() -> None:
    sky = _reader().process(_frame(world=10), None).sky
    assert sky is not None and sky.time_of_day == "night"
    assert sky.present


def test_sky_day_on_bright_frame() -> None:
    frame = np.full((FRAME_H, FRAME_W, 3), 200, dtype=np.uint8)
    sky = _reader().process(frame, None).sky
    assert sky is not None and sky.time_of_day == "day"


def test_sky_dusk_on_warm_frame() -> None:
    frame = np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:] = (100, 120, 200)  # B, G, R: red clearly dominant
    sky = _reader().process(frame, None).sky
    assert sky is not None and sky.time_of_day == "dusk"


def test_sky_clear_uniform_blue() -> None:
    frame = np.empty((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:] = (220, 200, 150)  # blue-dominant and uniform
    sky = _reader().process(frame, None).sky
    assert sky is not None
    assert sky.weather == "clear"
    assert sky.time_of_day == "day"


def test_sky_cloudy_when_noisy() -> None:
    frame = _frame(world=200)
    top = frame[0:int(FRAME_H * 0.12)]
    rng = np.random.default_rng(3)
    top[:] = np.clip(
        rng.integers(60, 240, size=top.shape, dtype=np.int16), 0, 255
    ).astype(np.uint8)
    sky = _reader().process(frame, None).sky
    assert sky is not None and sky.weather == "cloudy"


def test_minimap_marker_detected_at_center() -> None:
    frame = _frame(world=120)
    x, y, w, h = _minimap_box()
    cv2.circle(frame, (x + w // 2, y + h // 2), 6, (255, 255, 255), -1)
    minimap = _reader().process(frame, _minimap_box()).minimap
    assert minimap is not None and minimap.marker_present
    assert minimap.marker_offset is not None
    assert abs(minimap.marker_offset[0]) <= 0.05
    assert abs(minimap.marker_offset[1]) <= 0.05
    assert minimap.confidence >= 0.7


def test_minimap_no_marker_on_plain_map() -> None:
    frame = _frame(world=120)
    minimap = _reader().process(frame, _minimap_box()).minimap
    assert minimap is not None and not minimap.marker_present
    assert minimap.marker_offset is None


def test_minimap_reports_all_marker_blobs() -> None:
    frame = _frame(world=120)
    x, y, w, h = _minimap_box()
    cv2.circle(frame, (x + w // 2, y + h // 2), 6, (255, 255, 255), -1)
    cv2.circle(frame, (x + w // 2 + 40, y + h // 2), 8, (255, 255, 255), -1)
    minimap = _reader().process(frame, _minimap_box()).minimap
    assert minimap is not None and minimap.marker_present
    assert len(minimap.marker_blobs) == 2
    assert minimap.marker_offset is not None
    assert abs(minimap.marker_offset[0] - 40 / w) <= 0.02


def test_minimap_water_fraction() -> None:
    frame = _frame(world=120)
    x, y, w, h = _minimap_box()
    frame[y:y + h, x:x + w] = (220, 80, 60)  # blue dominant
    minimap = _reader().process(frame, _minimap_box()).minimap
    assert minimap is not None
    assert minimap.water_frac > 0.5
    assert minimap.road_frac < 0.1


def test_minimap_road_fraction() -> None:
    frame = _frame(world=20)
    x, y, w, h = _minimap_box()
    frame[y:y + h, x:x + w] = (100, 200, 230)  # bright and warm
    minimap = _reader().process(frame, _minimap_box()).minimap
    assert minimap is not None
    assert minimap.road_frac > 0.5
    assert minimap.water_frac < 0.1


def test_minimap_green_wilderness() -> None:
    frame = _frame(world=20)
    x, y, w, h = _minimap_box()
    frame[y:y + h, x:x + w] = (60, 160, 60)  # green dominant
    minimap = _reader().process(frame, _minimap_box()).minimap
    assert minimap is not None
    assert minimap.green_frac > 0.5
    assert minimap.road_frac < 0.1


def test_weapon_visible_only_when_bright() -> None:
    cfg = WorldConfig()
    boxes = _world_boxes(cfg)
    plain = _reader().process(_frame(), None).weapon
    assert plain is not None and not plain.ammo_visible
    frame = _frame()
    x, y, w, h = boxes["weapon"]
    frame[y:y + h, x:x + w] = 255
    lit = _reader().process(frame, None).weapon
    assert lit is not None and lit.ammo_visible
    assert lit.confidence > 0.0


def test_horse_gauge_detected() -> None:
    boxes = _world_boxes()
    frame = _frame()
    draw_gauge(frame, boxes["horse_health"], 0.8)
    draw_gauge(frame, boxes["horse_stamina"], 0.6)
    horse = _reader().process(frame, None).horse
    assert horse is not None and horse.detected
    assert horse.health is not None and abs(horse.health - 0.8) <= 0.15
    assert horse.stamina is not None and abs(horse.stamina - 0.6) <= 0.15


def test_horse_absent_on_plain_frame() -> None:
    horse = _reader().process(_frame(), None).horse
    assert horse is not None
    assert not horse.detected
    assert horse.health is None and horse.stamina is None


def test_boxes_and_labels_parallel() -> None:
    boxes = _world_boxes()
    frame = _frame()
    draw_gauge(frame, boxes["horse_health"], 0.8)
    x, y, w, h = boxes["weapon"]
    frame[y:y + h, x:x + w] = 255
    reading = _reader().process(frame, None)
    assert len(reading.boxes) == len(reading.labels) == 2


def test_perception_fills_environment_state() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=None)
    frame = _frame(world=10)
    x, y, w, h = _minimap_box()
    frame[y:y + h, x:x + w] = (220, 80, 60)  # water
    state = GameState()
    perception.apply_to_state(state, perception.process(frame, frame_id=3))
    assert state.environment.water_detected is True
    assert state.environment.road_detected is False
    assert state.environment.wilderness is False
    assert state.environment.time_of_day == "night"
    assert state.confidence.navigation > 0.0
    assert state.confidence.overall > 0.0


def test_perception_fills_horse_state() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=None)
    frame = _frame()
    draw_gauge(frame, _world_boxes()["horse_health"], 0.75)
    state = GameState()
    perception.apply_to_state(state, perception.process(frame, frame_id=4))
    assert state.horse.detected is True
    assert state.horse.health is not None
    assert state.confidence.horse > 0.0


def test_ammo_ocr_flows_into_state() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    fake = DigitOcr("24/120")
    perception = Perception(cfg, ocr=fake)
    frame = _frame()
    x, y, w, h = _world_boxes()["weapon"]
    frame[y:y + h, x:x + w] = 255
    pres = perception.process(frame, frame_id=1)
    assert fake.calls == 1
    assert pres.ammo == 24
    state = GameState()
    perception.apply_to_state(state, pres)
    assert state.player.weapon_ammo == 24


def test_ammo_ocr_skipped_without_weapon_visible() -> None:
    cfg = VisionConfig()
    cfg.ocr.every_n_frames = 1
    fake = DigitOcr()
    perception = Perception(cfg, ocr=fake)
    for i in range(3):
        pres = perception.process(_frame(), frame_id=i)
    assert fake.calls == 0
    assert pres.ammo is None


def test_world_summary_shape() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=None)
    summary = world_summary(perception.process(_frame(), 1))
    assert set(summary) >= {
        "enabled", "sky", "weather", "road", "water", "marker", "ammo",
        "horse_detected", "latency_ms",
    }
    assert summary["enabled"] is True


def test_overlay_boxes_include_world_labels() -> None:
    cfg = VisionConfig()
    perception = Perception(cfg, ocr=None)
    frame = _frame()
    draw_gauge(frame, _world_boxes()["horse_health"], 0.8)
    x, y, w, h = _world_boxes()["weapon"]
    frame[y:y + h, x:x + w] = 255
    boxes, labels = perception.overlay_boxes(perception.process(frame, 1))
    assert len(boxes) == len(labels)
    assert any(label.startswith("HP") for label in labels)
    assert any(label.startswith("ammo") for label in labels)


def test_world_line_in_overlay() -> None:
    status = AgentStatus(
        world_time="day", world_weather="clear", world_ammo=24,
        world_horse_detected=True, world_ms=1.5,
    )
    text = "\n".join(build_lines(status))
    assert "WORLD" in text
    assert "day" in text and "clear" in text
    assert "ammo 24" in text and "horse Y" in text


def test_world_defaults_render_as_dashes() -> None:
    text = "\n".join(build_lines(AgentStatus()))
    assert "--" in text
    assert "ammo --" in text and "horse N" in text


def test_invalid_world_region_rejected() -> None:
    cfg = AppConfig()
    cfg.vision.world.regions["horse_health"] = [2.0, 0.0, 0.1, 0.1]
    with pytest.raises(ConfigError, match="vision.world.regions.horse_health"):
        cfg.validate()


def test_invalid_world_fraction_rejected() -> None:
    cfg = AppConfig()
    cfg.vision.world.water_min_frac = 1.5
    with pytest.raises(ConfigError, match="vision.world.water_min_frac"):
        cfg.validate()


def test_default_world_config_valid() -> None:
    AppConfig().validate()
