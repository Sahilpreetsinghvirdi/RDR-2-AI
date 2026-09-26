"""Measure capture performance and suggest settings for this machine.

Usage:
    python tools/calibrate_capture.py
    python tools/calibrate_capture.py --backend mss --seconds 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capture.screen_capture import ScreenCapture  # noqa: E402
from src.capture.window_manager import (  # noqa: E402
    GameWindowManager,
    Rect,
    enable_dpi_awareness,
)
from src.config import load_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture calibration.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", choices=["auto", "dxcam", "mss", "synthetic"],
                        default=None)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--roi", choices=["window", "full"], default=None)
    parser.add_argument("--fps", type=int, default=None)
    return parser.parse_args(argv)


def measure(cfg, provider, seconds: float) -> dict[str, float]:
    capture = ScreenCapture(cfg, provider)
    capture.start()
    time.sleep(1.0)
    ids: list[int] = []
    latencies: list[float] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        packet = capture.buffer.latest()
        if packet is not None and (not ids or packet.frame_id > ids[-1]):
            ids.append(packet.frame_id)
            latencies.append(packet.capture_latency_ms)
        time.sleep(0.01)
    stats = capture.stats
    capture.stop()
    return {
        "fps": stats.fps,
        "avg_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "frames": float(len(ids)),
        "failures": float(stats.failures),
        "width": float(stats.width),
        "height": float(stats.height),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.backend:
        cfg.capture.backend = args.backend
    if args.roi:
        cfg.capture.roi_mode = args.roi
    if args.fps:
        cfg.capture.target_fps = args.fps
    cfg.validate()
    enable_dpi_awareness()

    wm = GameWindowManager(cfg.window)
    info = wm.find()
    if info is not None:
        print(f"window: found {info.title!r} {info.client.as_tuple()}")
    else:
        print("window: not found")
    print(f"backend: {cfg.capture.backend}  roi: {cfg.capture.roi_mode}  "
          f"target_fps: {cfg.capture.target_fps}")
    print("-" * 64)

    def provider():
        if info is not None:
            return info.client
        if cfg.capture.roi_mode == "full" or cfg.capture.backend == "synthetic":
            return Rect(0, 0, 2, 2)
        return None

    results = measure(cfg.capture, provider, args.seconds)

    print(f"measured fps      : {results['fps']:.1f}")
    print(f"avg capture latency: {results['avg_latency_ms']:.1f} ms")
    print(f"max capture latency: {results['max_latency_ms']:.1f} ms")
    print(f"frames collected   : {int(results['frames'])}")
    print(f"failures           : {int(results['failures'])}")
    if results["width"]:
        print(f"frame size         : {int(results['width'])}x{int(results['height'])}")

    measured = results["fps"]
    if measured <= 0:
        print("\nsuggestion: capture failed - check backend/monitor_index/window")
        return 1
    suggested = max(15, min(60, int(measured * 0.9)))
    print("\nsuggested config:")
    print(f"  capture.target_fps: {suggested}")
    if results["avg_latency_ms"] > 30:
        print("  - high latency: prefer backend dxcam, roi_mode: window, "
              "or lower the resolution in game settings")
    if info is None:
        print("  - game window not found: launch RDR2 Story Mode for window ROI tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
