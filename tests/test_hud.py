"""HUD region detection tests on synthetic frames (no game required)."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import HudConfig
from src.vision.hud import HudReader, scale_regions

FRAME_W, FRAME_H = 1920, 1080


def _cfg() -> HudConfig:
    return HudConfig()


def _boxes(cfg: HudConfig) -> dict[str, tuple[int, int, int, int]]:
    return scale_regions(cfg.regions, FRAME_W, FRAME_H)


def _frame(world: int = 10) -> np.ndarray:
    return np.full((FRAME_H, FRAME_W, 3), world, dtype=np.uint8)


def draw_gauge(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    fill: float,
    arc_bgr: tuple[int, int, int] = (60, 60, 255),
    track: tuple[int, int, int] = (35, 35, 35),
    icon: tuple[int, int, int] = (210, 210, 210),
) -> None:
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    r = int(min(w, h) * 0.35)
    thickness = max(4, r // 5)
    cv2.circle(frame, (cx, cy), r, track, thickness)
    cv2.circle(frame, (cx, cy), max(3, r // 3), icon, -1)
    if fill > 0:
        cv2.ellipse(frame, (cx, cy), (r, r), 0, 0, 360 * fill, arc_bgr, thickness)


def draw_prompt(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]
    crop[:] = (25, 25, 25)
    cv2.putText(crop, "PRESS E TO MOUNT", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(crop, "HOLD F TO FEED", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)


def test_scale_regions_maps_fractions_to_pixels() -> None:
    boxes = scale_regions({"health": [0.25, 0.5, 0.1, 0.2]}, 1920, 1080)
    assert boxes["health"] == (480, 540, 192, 216)


def test_scale_regions_clamps_out_of_bounds() -> None:
    boxes = scale_regions({"weird": [0.95, 0.95, 0.2, 0.2]}, 1000, 1000)
    x, y, w, h = boxes["weird"]
    assert x + w <= 1000
    assert y + h <= 1000
    assert w >= 4 and h >= 4


def test_scale_regions_rejects_tiny_frames() -> None:
    assert scale_regions({"health": [0.1, 0.1, 0.2, 0.2]}, 4, 4) == {}


def test_gauge_partial_fill() -> None:
    cfg = _cfg()
    frame = _frame()
    draw_gauge(frame, _boxes(cfg)["health"], 0.75)
    reader = HudReader(cfg)
    detection = reader.process(frame)
    gauge = detection.gauges["health"]
    assert gauge.value is not None
    assert abs(gauge.value - 0.75) <= 0.12
    assert gauge.confidence >= cfg.gauge_min_confidence
    assert gauge.present
    assert "%" in gauge.label


def test_gauge_full_fill() -> None:
    cfg = _cfg()
    frame = _frame()
    draw_gauge(frame, _boxes(cfg)["stamina"], 1.0)
    gauge = HudReader(cfg).process(frame).gauges["stamina"]
    assert gauge.value is not None
    assert gauge.value >= 0.9


def test_gauge_empty_reads_unknown_not_zero() -> None:
    cfg = _cfg()
    frame = _frame()
    draw_gauge(frame, _boxes(cfg)["health"], 0.0)
    gauge = HudReader(cfg).process(frame).gauges["health"]
    assert gauge.value is None
    assert gauge.present is False


def test_gauge_colours_do_not_matter() -> None:
    cfg = _cfg()
    for key, arc in (("health", (60, 60, 255)), ("dead_eye", (255, 120, 0)),
                     ("stamina", (0, 255, 240))):
        frame = _frame()
        draw_gauge(frame, _boxes(cfg)[key], 0.5, arc_bgr=arc)
        gauge = HudReader(cfg).process(frame).gauges[key]
        assert gauge.value is not None, key
        assert abs(gauge.value - 0.5) <= 0.15, key


def test_flat_region_reads_unknown() -> None:
    cfg = _cfg()
    detection = HudReader(cfg).process(_frame())
    for key in ("health", "stamina", "dead_eye"):
        assert detection.gauges[key].value is None
    assert detection.minimap is not None and not detection.minimap.present
    assert detection.prompt is not None and not detection.prompt.present
    assert detection.wanted is not None and detection.wanted.value is None


def test_gauge_on_bright_background_degrades_to_unknown() -> None:
    cfg = _cfg()
    frame = _frame(world=150)
    draw_gauge(frame, _boxes(cfg)["health"], 0.75, track=(10, 10, 10))
    gauge = HudReader(cfg).process(frame).gauges["health"]
    assert gauge.value is None or abs(gauge.value - 0.75) <= 0.3


def test_prompt_detection() -> None:
    cfg = _cfg()
    box = _boxes(cfg)["prompt"]
    frame = _frame()
    draw_prompt(frame, box)
    present = HudReader(cfg).process(frame).prompt
    assert present is not None and present.present
    assert present.confidence > 0.0
    empty = HudReader(cfg).process(_frame()).prompt
    assert empty is not None and not empty.present


def test_wanted_star_count() -> None:
    cfg = _cfg()
    x, y, w, h = _boxes(cfg)["wanted"]
    frame = _frame(world=20)
    crop = frame[y:y + h, x:x + w]
    for i in range(3):
        cv2.circle(crop, (30 + i * 40, h // 2), 7, (255, 255, 255), -1)
    wanted = HudReader(cfg).process(frame).wanted
    assert wanted is not None
    assert wanted.value == 3.0
    assert wanted.present


def test_minimap_presence_via_histogram() -> None:
    cfg = _cfg()
    box = _boxes(cfg)["minimap"]
    frame = _frame()
    x, y, w, h = box
    rng = np.random.default_rng(1)
    frame[y:y + h, x:x + w] = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    present = HudReader(cfg).process(frame).minimap
    assert present is not None and present.present
    absent = HudReader(cfg).process(_frame()).minimap
    assert absent is not None and not absent.present


def test_disabled_reader_skips_everything() -> None:
    cfg = HudConfig(enabled=False)
    detection = HudReader(cfg).process(_frame())
    assert detection.skipped
    assert detection.gauges == {}
    assert detection.minimap is None


def test_boxes_and_labels_stay_parallel() -> None:
    cfg = _cfg()
    frame = _frame()
    draw_gauge(frame, _boxes(cfg)["health"], 0.75)
    draw_prompt(frame, _boxes(cfg)["prompt"])
    detection = HudReader(cfg).process(frame)
    assert len(detection.boxes) == len(detection.labels)
    assert len(detection.boxes) == 2


def test_tiny_frame_does_not_crash() -> None:
    detection = HudReader(_cfg()).process(np.zeros((10, 10, 3), dtype=np.uint8))
    assert detection.gauges["health"].value is None


def test_latency_recorded() -> None:
    detection = HudReader(_cfg()).process(_frame())
    assert detection.latency_ms > 0.0
    assert not detection.skipped
