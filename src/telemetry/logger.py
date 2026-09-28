"""Logging setup and structured JSONL event logging."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
import time
from pathlib import Path
from typing import Any

from src.config import TelemetryConfig

LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(cfg: TelemetryConfig) -> Path | None:
    """Configure console + rotating file logging; returns the log file path."""
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if cfg.console and sys.stdout is not None:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(level)
        root.addHandler(console)

    log_path: Path | None = None
    if cfg.file:
        log_dir = Path(cfg.dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"rdr2ai_{stamp}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=cfg.log_max_bytes,
            backupCount=cfg.log_backups,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    for noisy in ("dxcam", "mss", "comtypes"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


class EventLog:
    """Append-only JSONL stream for state/vision/decision/action events."""

    def __init__(self, path: Path | None, flush_every: int = 20) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._handle = None
        self._pending = 0
        self._flush_every = flush_every
        self.count = 0
        self.errors = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path | None:
        return self._path

    def emit(self, kind: str, **fields: Any) -> None:
        if self._handle is None:
            return
        record = {"ts": round(time.time(), 3), "kind": kind, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self._handle.write(line + "\n")
                self.count += 1
                self._pending += 1
                if self._pending >= self._flush_every:
                    self._handle.flush()
                    self._pending = 0
            except Exception:
                self.errors += 1

    def close(self) -> None:
        with self._lock:
            if self._handle is None:
                return
            try:
                self._handle.flush()
                self._handle.close()
            except Exception:
                pass
            self._handle = None
