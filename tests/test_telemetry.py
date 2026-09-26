"""Event log and logging setup tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import TelemetryConfig
from src.telemetry.logger import EventLog, setup_logging


def test_setup_logging_creates_file(tmp_path: Path) -> None:
    cfg = TelemetryConfig(dir=str(tmp_path), console=False, file=True, log_level="DEBUG")
    path = setup_logging(cfg)
    assert path is not None and path.exists()
    logging.getLogger("test").debug("hello")
    for handler in logging.getLogger().handlers:
        handler.flush()
    content = path.read_text(encoding="utf-8")
    assert "hello" in content
    assert "DEBUG" in content


def test_setup_logging_without_file(tmp_path: Path) -> None:
    cfg = TelemetryConfig(dir=str(tmp_path), console=False, file=False)
    assert setup_logging(cfg) is None


def test_event_log_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path, flush_every=1)
    log.emit("state", old="INIT", new="RUNNING")
    log.emit("action", key="w", ok=True)
    log.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["kind"] == "state"
    assert rows[1]["key"] == "w"
    assert "ts" in rows[0]
    assert log.count == 2


def test_event_log_none_path_is_noop() -> None:
    log = EventLog(None)
    log.emit("state", value=1)
    log.close()
    assert log.count == 0


def test_event_log_append_mode(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    a = EventLog(path, flush_every=1)
    a.emit("one")
    a.close()
    b = EventLog(path, flush_every=1)
    b.emit("two")
    b.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
