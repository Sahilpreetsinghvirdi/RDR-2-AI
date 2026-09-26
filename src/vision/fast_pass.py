"""Fast vision layer: cheap per-frame statistics (brightness, motion).

This is the always-on layer of the vision hierarchy. Object detection (medium)
and scene reasoning (slow) arrive in later phases; they must never run on every
frame at full rate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import FastPassConfig


@dataclass
class FastPassResult:
    gray: np.ndarray
    brightness: float
    motion: float
    latency_ms: float
    skipped: bool = False
    frame_count: int = 0


class FastPass:
    """Downscaled grayscale statistics with frame-skipping support."""

    def __init__(self, cfg: FastPassConfig) -> None:
        self._cfg = cfg
        self._prev: np.ndarray | None = None
        self._last: FastPassResult | None = None
        self._counter = 0

    def reset(self) -> None:
        self._prev = None
        self._last = None
        self._counter = 0

    def process(self, image: np.ndarray) -> FastPassResult:
        self._counter += 1
        every = max(1, self._cfg.every_n_frames)
        if not self._cfg.enabled:
            return self._empty(skipped=True)
        if every > 1 and self._counter % every != 0 and self._last is not None:
            return FastPassResult(
                gray=self._last.gray,
                brightness=self._last.brightness,
                motion=self._last.motion,
                latency_ms=0.0,
                skipped=True,
                frame_count=self._counter,
            )

        t0 = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray,
            (self._cfg.width, self._cfg.height),
            interpolation=cv2.INTER_AREA,
        )
        brightness = float(small.mean())
        if self._prev is None or self._prev.shape != small.shape:
            motion = 0.0
        else:
            diff = cv2.absdiff(small, self._prev)
            motion = float(diff.mean())
        self._prev = small
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = FastPassResult(
            gray=small,
            brightness=brightness,
            motion=motion,
            latency_ms=latency_ms,
            skipped=False,
            frame_count=self._counter,
        )
        self._last = result
        return result

    def _empty(self, skipped: bool) -> FastPassResult:
        zeros = np.zeros((self._cfg.height, self._cfg.width), dtype=np.uint8)
        result = FastPassResult(
            gray=zeros, brightness=0.0, motion=0.0, latency_ms=0.0,
            skipped=skipped, frame_count=self._counter,
        )
        self._last = result
        return result
