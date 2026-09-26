"""Capture backends, color handling and the capture thread."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from src.capture.backends import (
    CaptureBackendError,
    SyntheticBackend,
    normalize_color,
)
from src.capture.screen_capture import FrameBuffer, FramePacket, ScreenCapture
from src.capture.window_manager import Rect
from src.config import CaptureConfig


def test_normalize_color_rgb_swaps_to_bgr() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    out = normalize_color(frame, "rgb")
    assert out[:, :, 2].max() == 255
    assert out[:, :, 0].max() == 0
    assert out.flags["C_CONTIGUOUS"]


def test_normalize_color_bgr_unchanged() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    out = normalize_color(frame, "bgr")
    assert out[:, :, 0].max() == 255
    assert out[:, :, 2].max() == 0


def test_normalize_color_bgra_drops_alpha() -> None:
    frame = np.full((4, 4, 4), 128, dtype=np.uint8)
    out = normalize_color(frame, "bgra")
    assert out.shape == (4, 4, 3)
    assert out[0, 0, 0] == 128


def test_normalize_color_rejects_bad_shape() -> None:
    with pytest.raises(CaptureBackendError):
        normalize_color(np.zeros((4, 4), dtype=np.uint8), "bgr")


def test_synthetic_backend_respects_region_and_fps() -> None:
    cfg = CaptureConfig(backend="synthetic", target_fps=60)
    backend = SyntheticBackend(cfg)
    backend.set_region(Rect(0, 0, 320, 200))
    frame = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        frame = backend.grab()
        if frame is not None:
            break
        time.sleep(0.01)
    assert frame is not None
    assert frame.shape == (200, 320, 3)
    backend.close()


def test_synthetic_backend_handles_odd_region_size() -> None:
    cfg = CaptureConfig(backend="synthetic", target_fps=60)
    backend = SyntheticBackend(cfg)
    backend.set_region(Rect(0, 0, 302, 273))
    frame = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        frame = backend.grab()
        if frame is not None:
            break
        time.sleep(0.01)
    assert frame is not None
    assert frame.shape == (273, 302, 3)
    backend.close()


def test_frame_buffer_latest_and_ring() -> None:
    buf = FrameBuffer(ring_size=2)
    assert buf.latest() is None
    for i in range(4):
        buf.put(
            FramePacket(
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                frame_id=i,
                wall_time=time.time(),
                timestamp=time.monotonic(),
                source="test",
                capture_latency_ms=1.0,
            )
        )
    assert buf.latest().frame_id == 3  # type: ignore[union-attr]
    assert len(buf.recent(10)) == 2
    buf.clear()
    assert buf.latest() is None


def test_screen_capture_synthetic_produces_frames(app_config) -> None:
    cfg = replace(app_config.capture, backend="synthetic", target_fps=60)
    capture = ScreenCapture(cfg, region_provider=lambda: None)
    capture.start()
    deadline = time.monotonic() + 3.0
    packet = None
    while time.monotonic() < deadline:
        packet = capture.buffer.latest()
        if packet is not None and packet.frame_id >= 5:
            break
        time.sleep(0.02)
    capture.stop()
    assert packet is not None
    assert packet.frame_id >= 5
    assert packet.capture_latency_ms >= 0.0
    assert packet.age_ms >= 0.0
    assert capture.stats.fps > 0.0
    assert capture.stats.backend == "synthetic"
    assert capture.stats.frames >= 5
    assert capture.running is False


def test_screen_capture_stops_cleanly(app_config) -> None:
    cfg = replace(app_config.capture, backend="synthetic", target_fps=30)
    capture = ScreenCapture(cfg, region_provider=lambda: None)
    capture.start()
    time.sleep(0.3)
    capture.stop(timeout_s=3.0)
    assert capture.thread is None


def test_screen_capture_waiting_for_window(app_config) -> None:
    cfg = replace(app_config.capture, backend="mss", target_fps=30, roi_mode="window")
    capture = ScreenCapture(cfg, region_provider=lambda: None)
    capture.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if capture.stats.waiting_for_window:
            break
        time.sleep(0.05)
    capture.stop()
    assert capture.stats.waiting_for_window is True
    assert capture.buffer.latest() is None
