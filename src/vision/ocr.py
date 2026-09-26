"""Optional OCR layer for HUD text (prompts, objectives).

Tesseract is the only engine (Phase 2); when neither the binary nor the
``pytesseract`` wrapper is installed the factory returns a null engine and
text fields simply stay unknown. Nothing here ever raises into the agent
loop.
"""

from __future__ import annotations

import logging

import numpy as np

from src.config import OcrConfig

log = logging.getLogger(__name__)


class OcrEngine:
    """Reads a text crop; returns (text, confidence 0..1)."""

    name = "base"
    available = False

    def read(self, image: np.ndarray) -> tuple[str, float]:  # pragma: no cover
        raise NotImplementedError


class NullOcr(OcrEngine):
    name = "off"
    available = False

    def read(self, image: np.ndarray) -> tuple[str, float]:
        return "", 0.0


class TesseractOcr(OcrEngine):
    """Tesseract via pytesseract; construction fails if unavailable."""

    def __init__(self, cfg: OcrConfig) -> None:
        import pytesseract  # noqa: PLC0415 - lazy: only when the engine is wanted

        pytesseract.get_tesseract_version()
        self._pt = pytesseract
        self._cfg = cfg
        self.name = "tesseract"
        self.available = True

    def read(self, image: np.ndarray) -> tuple[str, float]:
        if image is None or image.size == 0:
            return "", 0.0
        try:
            if image.ndim == 2:
                rgb = np.stack([image] * 3, axis=-1)
            else:
                rgb = image[:, :, ::-1]  # BGR -> RGB
            whitelist = (
                "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz .:$%/-'"
            )
            config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
            data = self._pt.image_to_data(
                rgb, lang=self._cfg.languages, config=config,
                output_type=self._pt.Output.DICT,
            )
        except Exception as exc:
            log.debug("ocr read failed: %s", exc)
            return "", 0.0
        words: list[str] = []
        best = 0.0
        for text, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
            try:
                score = float(conf)
            except (TypeError, ValueError):
                continue
            if text and text.strip() and score >= 0:
                words.append(text.strip())
                best = max(best, score / 100.0)
        return " ".join(words), best


def create_ocr(cfg: OcrConfig) -> OcrEngine:
    """Build the configured engine; never raises - degrades to :class:`NullOcr`."""
    if not cfg.enabled or cfg.engine == "off":
        return NullOcr()
    if cfg.engine in {"auto", "tesseract"}:
        try:
            engine = TesseractOcr(cfg)
            log.info("ocr engine: tesseract (%s)", cfg.languages)
            return engine
        except Exception as exc:
            if cfg.engine == "tesseract":
                log.warning(
                    "vision.ocr.engine=tesseract but unavailable (%s); "
                    "text reads disabled - install Tesseract + pytesseract", exc,
                )
            else:
                log.info("ocr: tesseract not installed, text reads disabled")
            return NullOcr()
    return NullOcr()
