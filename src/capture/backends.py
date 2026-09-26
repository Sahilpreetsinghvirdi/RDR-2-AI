"""Frame acquisition backends: DXGI Desktop Duplication, GDI (mss), synthetic."""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol, runtime_checkable

import numpy as np

from src.capture.window_manager import Rect
from src.config import CaptureConfig

log = logging.getLogger(__name__)

BGRA = "bgra"
BGR = "bgr"
RGB = "rgb"


class CaptureBackendError(Exception):
    """Raised when a capture backend cannot start or has failed permanently."""


def normalize_color(frame: np.ndarray, order: str) -> np.ndarray:
    """Return an 8-bit 3-channel BGR image from a raw backend frame."""
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise CaptureBackendError(f"unexpected frame shape {frame.shape}")
    channels = frame.shape[2]
    if channels == 4:
        bgr = frame[:, :, :3]
        if order == RGB:
            bgr = bgr[:, :, ::-1]
        return np.ascontiguousarray(bgr)
    if order == RGB:
        return np.ascontiguousarray(frame[:, :, ::-1])
    return np.ascontiguousarray(frame)


@runtime_checkable
class CaptureBackend(Protocol):
    name: str

    def set_region(self, region: Rect | None) -> None: ...

    def grab(self) -> np.ndarray | None: ...

    def close(self) -> None: ...


class DxcamBackend:
    """DXGI Desktop Duplication via dxcam (GPU-side copy, fastest option).

    dxcam 0.3 returns RGB frames; verified empirically on this machine.
    Regions are supplied in screen coordinates and translated into the
    coordinate space of the selected DXGI output.
    """

    name = "dxcam"

    def __init__(self, cfg: CaptureConfig) -> None:
        self._cfg = cfg
        self._order = cfg.color_order if cfg.color_order != "auto" else RGB
        self._region: Rect | None = None
        self._applied: Rect | None = None
        self._cam = None
        self._video_mode = False
        self._lock = threading.Lock()
        try:
            import dxcam
        except Exception as exc:  # pragma: no cover - depends on install
            raise CaptureBackendError(f"dxcam unavailable: {exc}") from exc
        self._dxcam = dxcam

    def _create(self) -> None:
        cam = self._dxcam.create(
            output_idx=self._cfg.monitor_index,
            max_buffer_len=self._cfg.dxcam_max_buffer,
        )
        if cam is None:
            raise CaptureBackendError("dxcam.create returned None (no DXGI output?)")
        self._cam = cam
        self._video_mode = False
        self._applied = None

    def _output_bounds(self) -> Rect | None:
        cam = self._cam
        if cam is None:
            return None
        try:
            dc = cam._output.desc.DesktopCoordinates
            return Rect(int(dc.left), int(dc.top), int(dc.right), int(dc.bottom))
        except Exception:
            log.debug("DXGI output desktop coordinates unavailable", exc_info=True)
            return None

    def _translate(self, region: Rect) -> Rect:
        bounds = self._output_bounds()
        if bounds is None:
            return region
        shifted = Rect(
            region.left - bounds.left,
            region.top - bounds.top,
            region.right - bounds.left,
            region.bottom - bounds.top,
        )
        if shifted.right <= 0 or shifted.bottom <= 0 or shifted.left >= bounds.width or \
                shifted.top >= bounds.height:
            raise CaptureBackendError(
                f"capture region {region.as_tuple()} is outside output "
                f"{bounds.as_tuple()}; set capture.monitor_index to the monitor "
                "showing the game"
            )
        return shifted.clamp(Rect(0, 0, bounds.width, bounds.height))

    def set_region(self, region: Rect | None) -> None:
        with self._lock:
            if region == self._applied and self._cam is not None:
                return
            if self._cam is None:
                self._create()
            self._region = region
            target = None
            if self._cfg.roi_mode == "window" and region is not None:
                target = self._translate(region)
                if not target.is_valid:
                    raise CaptureBackendError(f"invalid capture region {region.as_tuple()}")
            try:
                self._cam.region = target.as_tuple() if target is not None else None
            except Exception as exc:
                log.warning("dxcam region update failed (%s); recreating", exc)
                self._recreate(target)
            self._applied = region

    def _recreate(self, target: Rect | None) -> None:
        try:
            self._cam.stop()
            self._cam.release()
        except Exception:
            pass
        self._cam = None
        self._create()
        try:
            self._cam.region = target.as_tuple() if target is not None else None
        except Exception as exc:
            raise CaptureBackendError(f"dxcam region rejected: {exc}") from exc

    def _ensure_started(self) -> None:
        if self._cam is None:
            self._create()
        if self._video_mode:
            return
        try:
            self._cam.start(target_fps=self._cfg.target_fps)
            self._video_mode = True
            log.info("dxcam video mode started at target_fps=%d", self._cfg.target_fps)
        except Exception as exc:
            log.warning("dxcam start() failed (%s); falling back to blocking grab", exc)
            self._video_mode = False

    def grab(self) -> np.ndarray | None:
        with self._lock:
            self._ensure_started()
            frame = self._cam.grab()
        if frame is None:
            return None
        return normalize_color(np.asarray(frame), self._order)

    def close(self) -> None:
        with self._lock:
            if self._cam is None:
                return
            try:
                self._cam.stop()
                self._cam.release()
            except Exception:
                log.debug("dxcam close raised", exc_info=True)
            self._cam = None
            self._video_mode = False
            self._applied = None


class MssBackend:
    """GDI capture via mss - dependency-light fallback (BGRA output)."""

    name = "mss"

    def __init__(self, cfg: CaptureConfig) -> None:
        self._cfg = cfg
        self._order = cfg.color_order if cfg.color_order != "auto" else BGRA
        self._region: Rect | None = None
        self._sct = None

    def _ensure(self) -> None:
        if self._sct is None:
            import mss

            self._sct = mss.mss()

    def set_region(self, region: Rect | None) -> None:
        self._region = region

    def _target(self) -> dict[str, int]:
        if self._cfg.roi_mode == "window" and self._region is not None and self._region.is_valid:
            r = self._region
            return {"left": r.left, "top": r.top, "width": r.width, "height": r.height}
        self._ensure()
        mon = self._sct.monitors[1]  # type: ignore[index]
        return dict(mon)

    def grab(self) -> np.ndarray | None:
        self._ensure()
        target = self._target()
        if target["width"] <= 0 or target["height"] <= 0:
            return None
        shot = self._sct.grab(target)
        return normalize_color(np.asarray(shot), self._order)

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                log.debug("mss close raised", exc_info=True)
            self._sct = None


class SyntheticBackend:
    """Deterministic moving test pattern; lets every module run without the game."""

    name = "synthetic"

    def __init__(self, cfg: CaptureConfig) -> None:
        self._cfg = cfg
        self._order = cfg.color_order if cfg.color_order != "auto" else BGR
        self._region: Rect | None = None
        self._size = (cfg.synthetic.width, cfg.synthetic.height)
        self._rng = np.random.default_rng(cfg.synthetic.seed)
        self._t0 = time.monotonic()
        self._last = 0.0

    def set_region(self, region: Rect | None) -> None:
        self._region = region
        if region is not None and region.is_valid:
            self._size = (region.width, region.height)

    def grab(self) -> np.ndarray | None:
        period = 1.0 / self._cfg.target_fps
        now = time.monotonic()
        if now - self._last < period * 0.9:
            return None
        self._last = now
        w, h = self._size
        t = now - self._t0
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(30, 90, h, dtype=np.uint8)[:, None]
        frame[:, :, 1] = np.linspace(40, 110, w, dtype=np.uint8)[None, :]
        cx = int((0.5 + 0.35 * np.sin(t * 0.9)) * w) % max(w - 40, 1)
        cy = int((0.5 + 0.30 * np.cos(t * 0.7)) * h) % max(h - 40, 1)
        frame[cy : cy + 40, cx : cx + 40] = (40, 190, 240)
        rows = max((h - 1) // 8 + 1, 1)
        cols = max((w - 1) // 8 + 1, 1)
        noise = self._rng.integers(0, 8, size=(rows, cols, 3), dtype=np.uint8)
        frame[::8, ::8] = noise
        return normalize_color(frame, self._order)

    def close(self) -> None:
        return


def create_backend(cfg: CaptureConfig) -> CaptureBackend:
    """Instantiate the configured backend; ``auto`` prefers dxcam then mss."""
    choice = cfg.backend
    if choice == "synthetic":
        return SyntheticBackend(cfg)
    errors: list[str] = []
    order = ("dxcam", "mss") if choice == "auto" else (choice,)
    for name in order:
        try:
            if name == "dxcam":
                backend: CaptureBackend = DxcamBackend(cfg)
            else:
                backend = MssBackend(cfg)
            log.info("capture backend selected: %s", backend.name)
            return backend
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            log.warning("capture backend %s unavailable: %s", name, exc)
    raise CaptureBackendError("no capture backend available (" + "; ".join(errors) + ")")
