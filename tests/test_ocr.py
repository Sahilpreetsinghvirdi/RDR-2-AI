"""OCR engine selection and graceful degradation tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import OcrConfig
from src.vision.ocr import (
    NullOcr,
    OcrEngine,
    TesseractOcr,
    create_ocr,
    ensure_tesseract,
)


class FakePytesseract:
    """Mimics the pytesseract module: fails while tesseract_cmd is default."""

    def __init__(self, always_fail: bool = False) -> None:
        self.calls = 0
        self.always_fail = always_fail
        self.pytesseract = SimpleNamespace(tesseract_cmd="tesseract")

    def get_tesseract_version(self) -> str:
        self.calls += 1
        if self.always_fail or self.pytesseract.tesseract_cmd == "tesseract":
            raise RuntimeError("tesseract not found")
        return "5.4.0"


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


def test_ensure_tesseract_points_at_standard_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    exe = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setattr(
        "src.vision.ocr._tesseract_candidates", lambda: [exe]
    )
    fake = FakePytesseract()
    ensure_tesseract(fake)
    assert fake.pytesseract.tesseract_cmd == str(exe)


def test_ensure_tesseract_raises_when_nothing_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.vision.ocr._tesseract_candidates", lambda: [])
    fake = FakePytesseract(always_fail=True)
    with pytest.raises(RuntimeError):
        ensure_tesseract(fake)


def test_ensure_tesseract_accepts_path_hit_immediately() -> None:
    fake = FakePytesseract()
    fake.pytesseract.tesseract_cmd = "found-on-path"
    ensure_tesseract(fake)
    assert fake.pytesseract.tesseract_cmd == "found-on-path"
