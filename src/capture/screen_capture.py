"""High-rate screen capture thread with frame buffering and latency metrics."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from src.capture.backends import (
    CaptureBackend,
    CaptureBackendError,
    SyntheticBackend,
    create_backend,
)
from src.capture.window_manager import Rect
from src.config import CaptureConfig

log = logging.getLogger(__name__)


@dataclass(slots=True)
class FramePacket:
    """One captured frame with identity and timing metadata."""

    image: np.ndarray
    frame_id: int
    wall_time: float
    timestamp: float
    source: str
    capture_latency_ms: float
    region: Rect | None = None

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.timestamp) * 1000.0


class FrameBuffer:
    """Latest frame plus a small ring; writers never block readers for long."""

    def __init__(self, ring_size: int = 4) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: FramePacket | None = None
        self._ring: deque[FramePacket] = deque(maxlen=max(1, ring_size))

    def put(self, packet: FramePacket) -> None:
        with self._cond:
            self._latest = packet
            self._ring.append(packet)
            self._cond.notify_all()

    def latest(self) -> FramePacket | None:
        with self._lock:
            return self._latest

    def recent(self, n: int = 4) -> list[FramePacket]:
        with self._lock:
            return list(self._ring)[-n:]

    def wait_for_frame(self, timeout_s: float = 1.0) -> FramePacket | None:
        with self._cond:
            self._cond.wait(timeout_s)
            return self._latest

    def clear(self) -> None:
        with self._cond:
            self._latest = None
            self._ring.clear()


@dataclass
class CaptureStats:
    backend: str = "none"
    fps: float = 0.0
    latency_ms_avg: float = 0.0
    latency_ms_max: float = 0.0
    frames: int = 0
    failures: int = 0
    region: tuple[int, int, int, int] | None = None
    width: int = 0
    height: int = 0
    waiting_for_window: bool = False
    last_error: str = ""
    started_at: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "fps": round(self.fps, 1),
            "latency_ms_avg": round(self.latency_ms_avg, 2),
            "latency_ms_max": round(self.latency_ms_max, 2),
            "frames": self.frames,
            "failures": self.failures,
            "size": [self.width, self.height],
            "waiting_for_window": self.waiting_for_window,
            "last_error": self.last_error,
        }


class ScreenCapture:
    """Owns the capture backend and a worker thread publishing FramePackets."""

    def __init__(
        self,
        cfg: CaptureConfig,
        region_provider: Callable[[], Rect | None],
        on_beat: Callable[[str], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._region_provider = region_provider
        self._on_beat = on_beat
        self._buffer = FrameBuffer(cfg.ring_size)
        self._stats = CaptureStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backend: CaptureBackend | None = None
        self._applied_region: Rect | None = None
        self._failures = 0
        self._frame_id = 0
        self._fps_ema = 0.0
        self._lat_ema = 0.0
        self._last_frame_ts = 0.0

    @property
    def buffer(self) -> FrameBuffer:
        return self._buffer

    @property
    def stats(self) -> CaptureStats:
        return self._stats

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def thread(self) -> threading.Thread | None:
        return self._thread

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                log.warning("capture thread did not stop within %.1fs", timeout_s)
        self._close_backend()
        self._thread = None

    def _close_backend(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            try:
                backend.close()
            except Exception:
                log.debug("backend close raised", exc_info=True)

    def _try_fallback(self, reason: str) -> bool:
        current = self._backend.name if self._backend is not None else "?"
        self._close_backend()
        self._applied_region = None
        self._failures = 0
        if self._cfg.backend != "auto" or current != "dxcam":
            return False
        log.warning("falling back from dxcam to mss: %s", reason)
        self._stats.last_error = f"fallback: {reason}"
        try:
            self._backend = create_backend(replace(self._cfg, backend="mss"))
        except Exception as exc:
            self._stats.last_error = f"fallback failed: {exc}"
            log.error("fallback backend creation failed: %s", exc)
            return False
        self._stats.backend = self._backend.name
        self._applied_region = None
        return True

    def _apply_region(self, region: Rect | None) -> None:
        backend = self._backend
        if backend is None:
            return
        if region == self._applied_region:
            return
        backend.set_region(region)
        self._applied_region = region
        self._stats.region = region.as_tuple() if region is not None else None

    def _needs_window(self) -> bool:
        return self._backend is not None and not isinstance(self._backend, SyntheticBackend)

    def _update_fps(self, now: float) -> None:
        if self._last_frame_ts > 0.0:
            dt = now - self._last_frame_ts
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (
                    0.85 * self._fps_ema + 0.15 * inst
                )
        self._last_frame_ts = now

    def _run(self) -> None:
        log.info("capture thread started (target_fps=%d)", self._cfg.target_fps)
        period = 1.0 / max(self._cfg.target_fps, 1)
        try:
            while not self._stop.is_set():
                if self._on_beat is not None:
                    self._on_beat("capture")
                try:
                    if self._backend is None:
                        self._backend = create_backend(self._cfg)
                        self._stats.backend = self._backend.name
                        self._applied_region = None
                    region = self._region_provider()
                    if region is None and self._needs_window():
                        self._stats.waiting_for_window = True
                        self._stop.wait(0.25)
                        continue
                    self._stats.waiting_for_window = False
                    if self._cfg.roi_mode == "full":
                        region = None
                    self._apply_region(region)
                    self._capture_one(period)
                    self._failures = 0
                except CaptureBackendError as exc:
                    self._failures += 1
                    self._stats.failures += 1
                    self._stats.last_error = str(exc)
                    log.error("capture backend error (%d): %s", self._failures, exc)
                    if self._failures >= self._cfg.backend_failures_before_fallback:
                        if not self._try_fallback(str(exc)):
                            log.critical("capture failed permanently: %s", exc)
                            break
                    self._stop.wait(min(0.5, 0.05 * self._failures))
                except Exception as exc:
                    self._failures += 1
                    self._stats.failures += 1
                    self._stats.last_error = f"{type(exc).__name__}: {exc}"
                    log.exception("unexpected capture failure")
                    if self._failures >= self._cfg.backend_failures_before_fallback:
                        if not self._try_fallback(self._stats.last_error):
                            log.critical("capture failed permanently: %s", exc)
                            break
                    self._stop.wait(min(0.5, 0.05 * self._failures))
        finally:
            self._close_backend()
            log.info("capture thread stopped")

    def _capture_one(self, period: float) -> None:
        backend = self._backend
        if backend is None:
            return
        t0 = time.perf_counter()
        frame = backend.grab()
        now = time.monotonic()
        if frame is None:
            self._stop.wait(min(period, 0.005))
            return
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise CaptureBackendError("backend produced an empty frame")
        self._frame_id += 1
        if self._frame_id == 1:
            log.info("first frame: %dx%d from %s", frame.shape[1], frame.shape[0], backend.name)
        self._update_fps(now)
        self._lat_ema = latency_ms if self._lat_ema == 0.0 else (
            0.9 * self._lat_ema + 0.1 * latency_ms
        )
        self._stats.frames = self._frame_id
        self._stats.fps = self._fps_ema
        self._stats.latency_ms_avg = self._lat_ema
        self._stats.latency_ms_max = max(self._stats.latency_ms_max, latency_ms)
        self._stats.width = frame.shape[1]
        self._stats.height = frame.shape[0]
        self._buffer.put(
            FramePacket(
                image=frame,
                frame_id=self._frame_id,
                wall_time=time.time(),
                timestamp=now,
                source=backend.name,
                capture_latency_ms=latency_ms,
                region=self._applied_region,
            )
        )
        elapsed = time.perf_counter() - t0
        if elapsed < period:
            self._stop.wait(period - elapsed)
