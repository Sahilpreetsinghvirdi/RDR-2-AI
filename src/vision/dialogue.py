"""Subtitle/dialogue detection and encounter keywords (Phase 6).

Detects when the subtitle band carries bright text (RDR2 shows white
subtitles on a dark strip), optionally OCRs it, and matches configured
encounter keywords against prompt/subtitle text. Nothing is inferred beyond
those detections: no speaker, tone or meaning is guessed.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.config import DialogueConfig


def match_keyword(text: str | None, keywords: list[str]) -> str | None:
    """First configured keyword found in *text* (case-insensitive), else None."""
    if not text:
        return None
    low = text.lower()
    for keyword in keywords:
        if keyword and keyword.lower() in low:
            return keyword
    return None


class DialogueReading:
    """Result of one subtitle-band pass."""

    __slots__ = ("present", "text", "confidence", "latency_ms", "skipped")

    def __init__(self) -> None:
        self.present = False
        self.text: str | None = None
        self.confidence = 0.0
        self.latency_ms = 0.0
        self.skipped = True

    def as_dict(self) -> dict[str, object]:
        return {
            "present": self.present,
            "text": self.text or "",
            "confidence": round(self.confidence, 3),
            "skipped": self.skipped,
            "latency_ms": round(self.latency_ms, 2),
        }


class DialogueReader:
    """Measures bright-text density in the configured subtitle band."""

    def __init__(self, cfg: DialogueConfig) -> None:
        self._cfg = cfg

    def process(self, frame: np.ndarray) -> DialogueReading:
        """Measure the subtitle band of *frame*; OCR is orchestrated by perception."""
        t0 = time.perf_counter()
        reading = DialogueReading()
        if not self._cfg.enabled:
            reading.latency_ms = (time.perf_counter() - t0) * 1000.0
            return reading

        fh, fw = frame.shape[:2]
        x, y, w, h = self._cfg.region
        x0, y0 = int(x * fw), int(y * fh)
        x1, y1 = int((x + w) * fw), int((y + h) * fh)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            reading.latency_ms = (time.perf_counter() - t0) * 1000.0
            return reading

        reading.skipped = False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bright_frac = float(np.mean(gray >= self._cfg.bright_luma))
        reading.present = bright_frac >= self._cfg.min_text_frac
        reading.confidence = (
            min(1.0, bright_frac / max(self._cfg.min_text_frac, 1e-6))
            if reading.present else 0.0
        )
        reading.latency_ms = (time.perf_counter() - t0) * 1000.0
        return reading
