"""Grab frames from the game window (or screen) and save them for inspection.

Usage:
    python tools/inspect_frames.py --frames 5
    python tools/inspect_frames.py --backend mss --full --frames 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src.capture.screen_capture import ScreenCapture  # noqa: E402
from src.capture.window_manager import (  # noqa: E402
    GameWindowManager,
    Rect,
    enable_dpi_awareness,
)
from src.config import load_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect capture output.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", choices=["auto", "dxcam", "mss", "synthetic"],
                        default=None)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--full", action="store_true",
                        help="capture the whole output instead of the game window")
    parser.add_argument("--allow-nofocus", action="store_true",
                        help="capture the screen even if the game window is missing")
    parser.add_argument("--out", default="recordings/inspect")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.backend:
        cfg.capture.backend = args.backend
    if args.fps:
        cfg.capture.target_fps = args.fps
    if args.full:
        cfg.capture.roi_mode = "full"
    cfg.validate()
    enable_dpi_awareness()

    wm = GameWindowManager(cfg.window)
    info = wm.find()
    if info is None:
        print("game window not found", end="")
        if args.allow_nofocus or args.full or cfg.capture.backend == "synthetic":
            print(" -> capturing the whole output")
            cfg.capture.roi_mode = "full"
        else:
            print(" (use --allow-nofocus or --full to capture the screen anyway)")
            return 1
    else:
        print(f"window: '{info.title}' client={info.client.as_tuple()}")

    def provider():
        current = wm.current()
        if current is not None:
            return current.client
        return Rect(0, 0, 2, 2) if cfg.capture.roi_mode == "full" else None

    capture = ScreenCapture(cfg.capture, provider)
    capture.start()
    print(f"backend={cfg.capture.backend} roi={cfg.capture.roi_mode} "
          f"target_fps={cfg.capture.target_fps}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    deadline = time.monotonic() + 15.0
    while saved < args.frames and time.monotonic() < deadline:
        packet = capture.buffer.latest()
        if packet is None or packet.frame_id <= saved:
            time.sleep(0.05)
            continue
        path = out_dir / f"frame_{packet.frame_id:06d}.jpg"
        cv2.imwrite(str(path), packet.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"saved {path} ({packet.image.shape[1]}x{packet.image.shape[0]}) "
              f"latency={packet.capture_latency_ms:.1f}ms src={packet.source}")
        saved += 1
        time.sleep(0.2)

    stats = capture.stats
    capture.stop()
    print(f"stats: fps={stats.fps:.1f} avg_latency={stats.latency_ms_avg:.1f}ms "
          f"max_latency={stats.latency_ms_max:.1f}ms frames={stats.frames} "
          f"failures={stats.failures}")
    if stats.last_error:
        print(f"last_error: {stats.last_error}")
    print("NOTE: colours should look natural in the saved JPEGs; if they look "
          "swapped, adjust capture.color_order in config.yaml")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
