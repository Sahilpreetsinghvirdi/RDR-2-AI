"""HUD perception (Phase 2): core gauges, minimap, prompts, wanted stars.

Detection is region-based: ``vision.hud.regions`` gives fractional rectangles
(frame-relative) that are scaled to the actual frame. Gauges are read with a
polar luminance profile (ring fill), the minimap by histogram distance to its
surroundings, prompts by edge/brightness statistics, and wanted stars by
bright-blob counting. Everything reports a confidence; below
``gauge_min_confidence`` the value stays ``None`` rather than being guessed.

The default region layout assumes a 1920x1080 reference - tune it against the
real game with ``tools/calibrate_hud.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import HudConfig

GAUGE_LABELS: dict[str, str] = {"health": "HP", "stamina": "ST", "dead_eye": "DE"}
GAUGE_KEYS = ("health", "stamina", "dead_eye")


@dataclass
class RegionReading:
    name: str
    bbox: list[int] = field(default_factory=list)  # x, y, w, h in pixels
    value: float | None = None
    present: bool = False
    confidence: float = 0.0
    label: str = ""


@dataclass
class HudDetection:
    gauges: dict[str, RegionReading] = field(default_factory=dict)
    minimap: RegionReading | None = None
    prompt: RegionReading | None = None
    wanted: RegionReading | None = None
    latency_ms: float = 0.0
    skipped: bool = False

    @classmethod
    def empty(cls, skipped: bool = True) -> HudDetection:
        return cls(skipped=skipped)

    @property
    def boxes(self) -> list[list[int]]:
        """Pixel boxes of every detected region (overlay drawing order)."""
        out: list[list[int]] = []
        for key in GAUGE_KEYS:
            reading = self.gauges.get(key)
            if reading is not None and reading.present and reading.bbox:
                out.append(list(reading.bbox))
        for reading in (self.minimap, self.prompt, self.wanted):
            if reading is not None and reading.present and reading.bbox:
                out.append(list(reading.bbox))
        return out

    @property
    def labels(self) -> list[str]:
        out: list[str] = []
        for key in GAUGE_KEYS:
            reading = self.gauges.get(key)
            if reading is not None and reading.present:
                out.append(reading.label)
        for reading in (self.minimap, self.prompt, self.wanted):
            if reading is not None and reading.present:
                out.append(reading.label)
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "gauges": {
                k: {"value": r.value, "confidence": r.confidence, "present": r.present}
                for k, r in self.gauges.items()
            },
            "minimap": self.minimap.present if self.minimap else None,
            "prompt": self.prompt.present if self.prompt else None,
            "wanted": self.wanted.value if self.wanted else None,
            "latency_ms": round(self.latency_ms, 2),
            "skipped": self.skipped,
        }


def scale_regions(
    regions: dict[str, list[float]], frame_w: int, frame_h: int
) -> dict[str, tuple[int, int, int, int]]:
    """Map fractional regions onto pixel boxes, clamped inside the frame."""
    out: dict[str, tuple[int, int, int, int]] = {}
    if frame_w < 8 or frame_h < 8:
        return out
    for name, frac in regions.items():
        if not isinstance(frac, (list, tuple)) or len(frac) != 4:
            continue
        x = int(round(float(frac[0]) * frame_w))
        y = int(round(float(frac[1]) * frame_h))
        w = int(round(float(frac[2]) * frame_w))
        h = int(round(float(frac[3]) * frame_h))
        x = max(0, min(x, frame_w - 4))
        y = max(0, min(y, frame_h - 4))
        w = max(4, min(w, frame_w - x))
        h = max(4, min(h, frame_h - y))
        out[name] = (x, y, w, h)
    return out


def _to_gray(crop: np.ndarray) -> np.ndarray | None:
    if crop.ndim == 2:
        return crop.astype(np.uint8, copy=False)
    if crop.ndim != 3:
        return None
    if crop.shape[2] == 4:
        return cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
    if crop.shape[2] == 3:
        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return None


def _read_gauge(crop: np.ndarray, cfg: HudConfig) -> tuple[float | None, float]:
    """Ring fill of a circular core gauge: (value 0..1 | None, confidence)."""
    gray = _to_gray(crop)
    if gray is None:
        return None, 0.0
    h, w = gray.shape[:2]
    if min(h, w) < 16:
        return None, 0.0
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = min(cx, cy) - 1.0
    if rmax < 7:
        return None, 0.0

    r_i = np.rint(dist).astype(np.int32)
    r_i = np.clip(r_i, 0, int(rmax))
    counts = np.bincount(r_i.ravel(), minlength=int(rmax) + 1)
    sums = np.bincount(
        r_i.ravel(), weights=gray.ravel().astype(np.float64),
        minlength=int(rmax) + 1,
    )
    profile = sums / np.maximum(counts, 1)
    max_profile = np.zeros_like(profile)
    np.maximum.at(max_profile, r_i.ravel(), gray.ravel().astype(np.float64))
    if profile.size < 8:
        return None, 0.0

    # bg from the low end so a bright outdoor background does not hide the
    # ring (the dark track is the reference); peak from the bright end.
    bg = float(np.percentile(profile, 10))
    peak = float(np.percentile(max_profile, 95))
    contrast = peak - bg
    confidence = float(np.clip(contrast / 100.0, 0.0, 1.0))
    if contrast < cfg.gauge_contrast_min:
        return None, confidence

    # Band = outermost radii with any bright content. The ring is dimmer than
    # the icon, so band detection uses a low factor against the max profile;
    # the outer-half/edge rules reject background and the centre icon.
    band_thresh = bg + 0.15 * (peak - bg)
    active = max_profile >= band_thresh
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))

    band: tuple[int, int] | None = None
    for s, e in reversed(runs):
        # The ring sits in the outer half and must not touch the crop edge
        # (a run touching the edge is background, not the gauge).
        if s >= 0.5 * rmax and e <= rmax * 0.97:
            band = (s, e)
            break
    if band is None:
        return None, confidence

    band_mask = (dist >= band[0] - 0.5) & (dist <= band[1] + 0.5)
    if int(band_mask.sum()) < 30:
        return None, confidence
    n_bins = 180
    ang = np.arctan2(yy - cy, xx - cx)
    bin_i = np.clip(((ang + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int32),
                    0, n_bins - 1)
    max_per_bin = np.full(n_bins, -1.0, dtype=np.float64)
    np.maximum.at(max_per_bin, bin_i[band_mask], gray[band_mask].astype(np.float64))
    samples = max_per_bin[max_per_bin >= 0]
    if samples.size < n_bins * 0.8:
        return None, confidence
    # Bimodal split between lit arc and empty track inside the band itself,
    # so the reading does not depend on the icon's absolute brightness.
    low = float(np.percentile(samples, 20))
    high = float(np.percentile(samples, 95))
    thresh_fill = low + 0.5 * (high - low)
    fill = float((samples >= thresh_fill).mean())
    return round(fill, 3), round(confidence, 3)


def _read_minimap(
    frame: np.ndarray, box: tuple[int, int, int, int], cfg: HudConfig
) -> tuple[bool, float]:
    """Present if the region's colour distribution differs from its surround."""
    fh, fw = frame.shape[:2]
    x, y, w, h = box
    pad = max(6, min(w, h) // 8)
    ox, oy = max(0, x - pad), max(0, y - pad)
    ow = min(fw, x + w + pad) - ox
    oh = min(fh, y + h + pad) - oy
    inner = frame[y:y + h, x:x + w]
    outer_ring = frame[oy:oy + oh, ox:ox + ow].copy()
    if inner.size == 0 or outer_ring.size == 0 or ow * oh < 300:
        return False, 0.0
    ry0, rx0 = y - oy, x - ox
    outer_ring[ry0:ry0 + h, rx0:rx0 + w] = 0
    if int(np.count_nonzero(outer_ring)) < 100:
        return False, 0.0

    def _hist(img: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    distance = float(cv2.compareHist(_hist(inner), _hist(outer_ring),
                                     cv2.HISTCMP_BHATTACHARYYA))
    if distance < cfg.minimap_hist_threshold:
        return False, round(min(distance, 1.0), 3)
    conf = float(np.clip(0.5 + 0.5 * (distance - cfg.minimap_hist_threshold)
                         / max(1e-6, 1.0 - cfg.minimap_hist_threshold), 0.0, 1.0))
    return True, round(conf, 3)


def _read_prompt(crop: np.ndarray, cfg: HudConfig) -> tuple[bool, float]:
    """Prompt text: near-white glyphs with enough edge density."""
    gray = _to_gray(crop)
    if gray is None or min(gray.shape[:2]) < 8:
        return False, 0.0
    edges = cv2.Canny(gray, 60, 150)
    density = float((edges > 0).mean())
    bright = float((gray > 190).mean())
    ok = density >= cfg.prompt_min_edge_density and bright >= cfg.prompt_min_bright_frac
    conf = min(
        density / (2.0 * cfg.prompt_min_edge_density),
        bright / (2.0 * cfg.prompt_min_bright_frac),
        1.0,
    )
    return ok, round(max(conf, 0.0), 3)


def _read_wanted(crop: np.ndarray, cfg: HudConfig) -> tuple[float | None, float]:
    """Count small near-white blobs (wanted stars)."""
    gray = _to_gray(crop)
    if gray is None or min(gray.shape[:2]) < 8:
        return None, 0.0
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    count = 0
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if cfg.wanted_min_blob_px <= area <= cfg.wanted_max_blob_px:
            count += 1
    if count == 0:
        return None, 0.0
    conf = float(min(1.0, 0.4 + 0.15 * count))
    return float(count), round(conf, 3)


def _pct_label(prefix: str, value: float | None) -> str:
    if value is None:
        return f"{prefix} --"
    return f"{prefix} {value * 100:.0f}%"


class HudReader:
    """Runs the region-based HUD detectors over one frame."""

    def __init__(self, cfg: HudConfig) -> None:
        self._cfg = cfg

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def process(self, frame: np.ndarray) -> HudDetection:
        if not self._cfg.enabled:
            return HudDetection.empty(skipped=True)
        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]
        boxes = scale_regions(self._cfg.regions, fw, fh)
        detection = HudDetection()

        for key in GAUGE_KEYS:
            box = boxes.get(key)
            if box is None:
                continue
            x, y, w, h = box
            crop = frame[y:y + h, x:x + w]
            value, conf = _read_gauge(crop, self._cfg)
            if conf < self._cfg.gauge_min_confidence:
                value = None
            reading = RegionReading(
                name=key, bbox=list(box), value=value,
                present=value is not None, confidence=conf,
                label=_pct_label(GAUGE_LABELS[key], value),
            )
            detection.gauges[key] = reading

        box = boxes.get("minimap")
        if box is not None:
            x, y, w, h = box
            present, conf = _read_minimap(frame, box, self._cfg)
            detection.minimap = RegionReading(
                name="minimap", bbox=list(box), present=present, confidence=conf,
                label="map" if present else "map --",
            )

        box = boxes.get("prompt")
        if box is not None:
            x, y, w, h = box
            present, conf = _read_prompt(frame[y:y + h, x:x + w], self._cfg)
            detection.prompt = RegionReading(
                name="prompt", bbox=list(box), present=present, confidence=conf,
                label="prompt" if present else "prompt --",
            )

        box = boxes.get("wanted")
        if box is not None:
            x, y, w, h = box
            value, conf = _read_wanted(frame[y:y + h, x:x + w], self._cfg)
            detection.wanted = RegionReading(
                name="wanted", bbox=list(box), value=value,
                present=value is not None, confidence=conf,
                label=f"stars x{int(value)}" if value is not None else "stars --",
            )

        detection.latency_ms = (time.perf_counter() - t0) * 1000.0
        return detection
