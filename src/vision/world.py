"""World state estimation (Phase 3): minimap, sky, weapon, horse cores.

Everything here is region-based and heuristic, and every reading carries a
confidence - fields stay ``None`` when the signal is not there rather than
being guessed:

- **Minimap**: player marker (bright blob near centre) plus colour fractions
  classified as water / road / vegetation, feeding environment hints.
- **Sky**: top strip luminance and warmth -> time of day, uniformity -> weather.
- **Weapon**: ammo counter visibility; the numeric value needs OCR.
- **Horse**: the two cores that appear above the player's when the horse is
  nearby - health/stamina values via the same ring-fill gauge reader as Phase 2.

The minimap region is shared with ``vision.hud.regions.minimap``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import HudConfig, WorldConfig
from src.vision.hud import read_gauge, scale_regions


@dataclass
class MinimapReading:
    bbox: list[int] = field(default_factory=list)
    marker_present: bool = False
    marker_bbox: list[int] | None = None
    marker_offset: list[float] | None = None  # dx, dy in -0.5..0.5
    # Every bright blob: [dx, dy, area_px]. Used to pick the nearest
    # marker; the largest one stays the legacy marker_present reading.
    marker_blobs: list[list[float]] = field(default_factory=list)
    enemy_dots: int | None = None     # None = minimap too small to read
    enemy_offset: list[float] | None = None  # nearest red blip, dx/dy in -0.5..0.5
    water_frac: float = 0.0
    road_frac: float = 0.0
    green_frac: float = 0.0
    confidence: float = 0.0
    present: bool = False


@dataclass
class SkyReading:
    bbox: list[int] = field(default_factory=list)
    time_of_day: str | None = None  # day | dusk | night
    weather: str | None = None      # clear | cloudy
    confidence: float = 0.0
    present: bool = False


@dataclass
class WeaponReading:
    bbox: list[int] = field(default_factory=list)
    ammo_visible: bool = False
    ammo: int | None = None
    confidence: float = 0.0
    present: bool = False


@dataclass
class HorseReading:
    health_bbox: list[int] = field(default_factory=list)
    stamina_bbox: list[int] = field(default_factory=list)
    detected: bool = False
    health: float | None = None
    stamina: float | None = None
    confidence: float = 0.0
    present: bool = False


@dataclass
class WorldReading:
    minimap: MinimapReading | None = None
    sky: SkyReading | None = None
    weapon: WeaponReading | None = None
    horse: HorseReading | None = None
    latency_ms: float = 0.0
    skipped: bool = False

    @classmethod
    def empty(cls, skipped: bool = True) -> WorldReading:
        return cls(skipped=skipped)

    @property
    def boxes(self) -> list[list[int]]:
        out: list[list[int]] = []
        if self.horse is not None and self.horse.present:
            if self.horse.health_bbox:
                out.append(list(self.horse.health_bbox))
            if self.horse.stamina_bbox:
                out.append(list(self.horse.stamina_bbox))
        if self.weapon is not None and self.weapon.present and self.weapon.bbox:
            out.append(list(self.weapon.bbox))
        return out

    @property
    def labels(self) -> list[str]:
        out: list[str] = []
        if self.horse is not None and self.horse.present:
            if self.horse.health_bbox:
                out.append(f"HP {_pct(self.horse.health)}")
            if self.horse.stamina_bbox:
                out.append(f"ST {_pct(self.horse.stamina)}")
        if self.weapon is not None and self.weapon.present and self.weapon.bbox:
            ammo = "?" if self.weapon.ammo is None else str(self.weapon.ammo)
            out.append(f"ammo {ammo}")
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "minimap": (
                {
                    "marker": self.minimap.marker_present,
                    "enemies": self.minimap.enemy_dots,
                    "water": round(self.minimap.water_frac, 3),
                    "road": round(self.minimap.road_frac, 3),
                    "green": round(self.minimap.green_frac, 3),
                }
                if self.minimap else None
            ),
            "sky": (
                {"time": self.sky.time_of_day, "weather": self.sky.weather}
                if self.sky else None
            ),
            "ammo": self.weapon.ammo if self.weapon else None,
            "ammo_visible": self.weapon.ammo_visible if self.weapon else False,
            "horse": (
                {"detected": self.horse.detected, "health": self.horse.health,
                 "stamina": self.horse.stamina}
                if self.horse else None
            ),
            "latency_ms": round(self.latency_ms, 2),
            "skipped": self.skipped,
        }


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.0f}%"


def _read_minimap(
    crop: np.ndarray, cfg: WorldConfig
) -> MinimapReading:
    reading = MinimapReading()
    if crop.size == 0:
        return reading
    h, w = crop.shape[:2]
    if min(h, w) < 16:
        return reading
    reading.present = True

    b = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    r = crop[:, :, 2].astype(np.int16)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    water = (b > r + 20) & (b > 100)
    road = (gray > cfg.road_luma) & (b <= r)
    green = (g > r + 10) & (g > b + 10) & (g > 80)
    total = float(h * w)
    reading.water_frac = round(float(water.sum()) / total, 3)
    reading.road_frac = round(float(road.sum()) / total, 3)
    reading.green_frac = round(float(green.sum()) / total, 3)

    sh0, sh1 = h // 4, max(h // 4 + 1, 3 * h // 4)
    sw0, sw1 = w // 4, max(w // 4 + 1, 3 * w // 4)
    sub = gray[sh0:sh1, sw0:sw1]
    _, binary = cv2.threshold(sub, cfg.marker_bright, 255, cv2.THRESH_BINARY)
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    blobs: list[tuple[int, float, float, float]] = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if cfg.marker_min_blob_px <= area <= cfg.marker_max_blob_px:
            blobs.append((
                i,
                float(centroids[i][0]) + sw0,
                float(centroids[i][1]) + sh0,
                float(area),
            ))
    for _, cx, cy, area in blobs:
        reading.marker_blobs.append([
            round(cx / w - 0.5, 3), round(cy / h - 0.5, 3), round(area, 1),
        ])
    best = max(blobs, key=lambda b: b[3], default=None)
    if best is not None:
        label, cx, cy, _ = best
        reading.marker_present = True
        reading.marker_offset = [
            round(cx / w - 0.5, 3), round(cy / h - 0.5, 3),
        ]
        reading.marker_bbox = [
            int(stats[label, cv2.CC_STAT_LEFT]) + sw0,
            int(stats[label, cv2.CC_STAT_TOP]) + sh0,
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        ]
        reading.confidence = 0.7
    else:
        reading.confidence = 0.4

    reading.enemy_dots, reading.enemy_offset = _count_enemy_dots(crop, cfg)
    return reading


def _count_enemy_dots(
    crop: np.ndarray, cfg: WorldConfig
) -> tuple[int | None, list[float] | None]:
    """Count small red blips (enemy markers) on the minimap; nearest one gives a direction."""
    h, w = crop.shape[:2]
    if min(h, w) < 16:
        return None, None
    b = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    r = crop[:, :, 2].astype(np.int16)
    red = (
        (r >= cfg.enemy_red_min)
        & (r >= g + cfg.enemy_red_chroma)
        & (r >= b + cfg.enemy_red_chroma)
    ).astype(np.uint8)
    if int(red.sum()) == 0:
        return 0, None
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(red, 8)
    dots: list[tuple[float, float]] = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if cfg.enemy_min_blob_px <= area <= cfg.enemy_max_blob_px:
            dots.append((float(centroids[i][0]), float(centroids[i][1])))
    if not dots:
        return 0, None
    dots = dots[: cfg.enemy_max_dots]
    cx, cy = min(
        dots, key=lambda p: (p[0] - w / 2.0) ** 2 + (p[1] - h / 2.0) ** 2
    )
    return len(dots), [round(cx / w - 0.5, 3), round(cy / h - 0.5, 3)]


def _read_sky(crop: np.ndarray, cfg: WorldConfig) -> SkyReading:
    reading = SkyReading()
    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return reading
    reading.present = True
    b = float(crop[:, :, 0].mean())
    r = float(crop[:, :, 2].mean())
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    luma = float(gray.mean())
    std = float(gray.std())

    if luma < cfg.night_luma:
        reading.time_of_day = "night"
        reading.confidence = 0.7
    elif r > b + cfg.dusk_warmth:
        reading.time_of_day = "dusk"
        reading.confidence = 0.6
    else:
        reading.time_of_day = "day"
        reading.confidence = 0.7

    if std <= cfg.sky_max_std:
        reading.weather = "clear" if b > r + 5 else "cloudy"
    else:
        reading.weather = "cloudy"
    return reading


def _read_ammo(crop: np.ndarray, cfg: WorldConfig) -> WeaponReading:
    reading = WeaponReading()
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return reading
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright = float((gray > cfg.ammo_min_luma).mean())
    reading.ammo_visible = (
        bright >= cfg.ammo_min_bright_frac and float(gray.mean()) >= 40.0
    )
    reading.present = reading.ammo_visible
    reading.confidence = round(min(1.0, bright / (2.0 * cfg.ammo_min_bright_frac)), 3) \
        if reading.ammo_visible else 0.0
    return reading


def _read_horse_gauge(crop: np.ndarray, hud_cfg: HudConfig) -> tuple[float | None, float]:
    value, conf = read_gauge(crop, hud_cfg)
    if conf < hud_cfg.gauge_min_confidence:
        value = None
    return value, conf


class WorldReader:
    """Runs the Phase 3 world detectors over one frame."""

    def __init__(self, cfg: WorldConfig, hud_cfg: HudConfig) -> None:
        self._cfg = cfg
        self._hud_cfg = hud_cfg

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def process(
        self, frame: np.ndarray, minimap_region: tuple[int, int, int, int] | None
    ) -> WorldReading:
        if not self._cfg.enabled:
            return WorldReading.empty(skipped=True)
        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]
        boxes = scale_regions(self._cfg.regions, fw, fh)
        reading = WorldReading()

        if minimap_region is not None:
            x, y, w, h = minimap_region
            reading.minimap = _read_minimap(frame[y:y + h, x:x + w], self._cfg)

        box = boxes.get("sky")
        if box is not None:
            x, y, w, h = box
            sky = _read_sky(frame[y:y + h, x:x + w], self._cfg)
            sky.bbox = list(box)
            reading.sky = sky

        box = boxes.get("weapon")
        if box is not None:
            x, y, w, h = box
            ammo = _read_ammo(frame[y:y + h, x:x + w], self._cfg)
            ammo.bbox = list(box)
            reading.weapon = ammo

        horse = HorseReading()
        any_gauge = False
        for key, attr in (("horse_health", "health"), ("horse_stamina", "stamina")):
            box = boxes.get(key)
            if box is None:
                continue
            x, y, w, h = box
            value, conf = _read_horse_gauge(frame[y:y + h, x:x + w], self._hud_cfg)
            if value is None:
                continue
            setattr(horse, attr, value)
            any_gauge = True
            horse.confidence = max(horse.confidence, conf)
            if attr == "health":
                horse.health_bbox = list(box)
            else:
                horse.stamina_bbox = list(box)
        horse.detected = any_gauge
        horse.present = any_gauge
        reading.horse = horse

        reading.latency_ms = (time.perf_counter() - t0) * 1000.0
        return reading
