"""Session recording: frames + state snapshots for later dataset building."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.capture.screen_capture import FramePacket
from src.config import RecordingConfig

log = logging.getLogger(__name__)


class SessionRecorder:
    """Rate-limited frame/state recorder that disables itself on I/O errors."""

    def __init__(self, cfg: RecordingConfig, base_dir: Path) -> None:
        self._cfg = cfg
        self._base = Path(base_dir)
        self._lock = threading.Lock()
        self._session_dir: Path | None = None
        self._frames_dir: Path | None = None
        self._states_handle = None
        self._last_saved = 0.0
        self._saved = 0
        self._disabled_reason = ""
        self._t0 = time.time()
        self._meta: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self._session_dir is not None and not self._disabled_reason

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def saved(self) -> int:
        return self._saved

    def start(self, metadata: dict[str, Any] | None = None) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session = self._base / f"session_{stamp}"
        counter = 1
        while session.exists():
            counter += 1
            session = self._base / f"session_{stamp}_{counter}"
        frames = session / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        self._session_dir = session
        self._frames_dir = frames
        if self._cfg.save_state_jsonl:
            self._states_handle = (session / "states.jsonl").open("a", encoding="utf-8")
        self._meta = {
            "started_at": time.time(),
            "target_fps": self._cfg.fps,
            "image_format": self._cfg.image_format,
            **(metadata or {}),
        }
        (session / "metadata.json").write_text(
            json.dumps(self._meta, indent=2, default=str), encoding="utf-8"
        )
        log.info("recording session: %s", session)
        return session

    def _should_save(self) -> bool:
        if not self.enabled:
            return False
        if self._saved >= self._cfg.max_frames:
            return False
        interval = 1.0 / max(1, self._cfg.fps)
        now = time.monotonic()
        if now - self._last_saved < interval:
            return False
        self._last_saved = now
        return True

    def record(self, packet: FramePacket, state: dict[str, Any] | None = None) -> bool:
        with self._lock:
            if not self._should_save():
                return False
            try:
                name = f"{packet.frame_id:06d}.{self._cfg.image_format}"
                path = self._frames_dir / name  # type: ignore[operator]
                params = []
                if self._cfg.image_format == "jpg":
                    params = [cv2.IMWRITE_JPEG_QUALITY, int(self._cfg.jpeg_quality)]
                ok = cv2.imwrite(str(path), packet.image, params)
                if not ok:
                    raise OSError(f"cv2.imwrite failed for {path}")
                if self._states_handle is not None:
                    row = {
                        "frame_id": packet.frame_id,
                        "wall_time": packet.wall_time,
                        "source": packet.source,
                        "capture_latency_ms": round(packet.capture_latency_ms, 2),
                        "image": name,
                        "state": state or {},
                    }
                    self._states_handle.write(json.dumps(row, default=str) + "\n")
                    self._states_handle.flush()
                self._saved += 1
                if self._saved >= self._cfg.max_frames:
                    log.warning(
                        "recording limit reached (%d frames); further frames skipped",
                        self._cfg.max_frames,
                    )
                return True
            except Exception as exc:
                self._disabled_reason = str(exc)
                log.error("recording disabled after error: %s", exc)
                return False

    def record_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            if self._states_handle is None:
                return
            try:
                row = {"event": event, "wall_time": time.time()}
                self._states_handle.write(json.dumps(row, default=str) + "\n")
            except Exception as exc:
                log.debug("event record failed: %s", exc)

    def close(self) -> None:
        with self._lock:
            if self._session_dir is None:
                return
            try:
                if self._states_handle is not None:
                    self._states_handle.flush()
                    self._states_handle.close()
                    self._states_handle = None
                self._meta.update(
                    {
                        "ended_at": time.time(),
                        "frames_saved": self._saved,
                        "disabled_reason": self._disabled_reason,
                    }
                )
                (self._session_dir / "metadata.json").write_text(
                    json.dumps(self._meta, indent=2, default=str), encoding="utf-8"
                )
                log.info(
                    "recording closed: %d frames -> %s", self._saved, self._session_dir
                )
            except Exception:
                log.debug("recorder close failed", exc_info=True)

    @staticmethod
    def save_frame(image: np.ndarray, path: Path, quality: int = 95) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return path
