"""OCR engine selection and graceful degradation tests."""

from __future__ import annotations

import numpy as np

from src.config import OcrConfig
from src.vision.ocr import NullOcr, OcrEngine, TesseractOcr, create_ocr


def test_null_engine_returns_nothing() -> None:
    engine = NullOcr()
    text, conf = engine.read(np.zeros((20, 60, 3), dtype=np.uint8))
    assert text == ""
    assert conf == 0.0
    assert engine.name == "off"
    assert engine.available is False


def test_engine_off_returns_null() -> None:
    engine = create_ocr(OcrConfig(engine="off"))
    assert isinstance(engine, NullOcr)


def test_disabled_returns_null() -> None:
    engine = create_ocr(OcrConfig(enabled=False, engine="auto"))
    assert isinstance(engine, NullOcr)


def test_auto_never_raises_and_reports_consistently() -> None:
    engine = create_ocr(OcrConfig(engine="auto"))
    assert isinstance(engine, (NullOcr, TesseractOcr))
    if isinstance(engine, NullOcr):
        assert engine.name == "off"
        assert engine.available is False
    else:
        assert engine.name == "tesseract"
        assert engine.available is True


def test_explicit_tesseract_degrades_gracefully_when_missing() -> None:
    engine = create_ocr(OcrConfig(engine="tesseract"))
    assert isinstance(engine, (NullOcr, TesseractOcr))
    text, conf = engine.read(np.zeros((20, 60, 3), dtype=np.uint8))
    assert isinstance(text, str)
    assert 0.0 <= conf <= 1.0


def test_protocol_contract() -> None:
    engine: OcrEngine = NullOcr()
    assert callable(engine.read)
