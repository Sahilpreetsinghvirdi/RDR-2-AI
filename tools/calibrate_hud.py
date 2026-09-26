"""Run the HUD detector on live frames and save annotated previews.

Prints the pixel boxes and readings so you can tune vision.hud.regions in
config.yaml against the real game.

Usage:
    python tools/calibrate_hud.py
    python tools/calibrate_hud.py --frames 5 --backend synthetic
    python tools/calibrate_hud.py --region health
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
from src.vision.hud import RegionReading, scale_regions  # noqa: E402
from src.vision.perception import Perception  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate HUD regions.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", choices=["auto", "dxcam", "mss", "synthetic"],
                        default=None)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--region", default=None,
                        help="only analyze this region (health|stamina|dead_eye|"
                             "minimap|prompt|wanted)")
    parser.add_argument("--full", action="store_true",
                        help="capture the whole output instead of the game window")
    parser.add_argument("--allow-nofocus", action="store_true",
                        help="capture the screen even if the game window is missing")
    parser.add_argument("--out", default="recordings/hud")
    return parser.parse_args(argv)


def _annotated(image, detection) -> object:
    canvas = image.copy()
    hud_color = (0, 255, 255)
    for box, label in zip(detection.boxes, detection.labels, strict=False):
        x, y, w, h = box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), hud_color, 2)
        cv2.putText(canvas, label, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 1, cv2.LINE_AA)
    return canvas


def _reading_line(name: str, reading: RegionReading | None) -> str:
    if reading is None:
        return f"  {name:<9} missing from config"
    if name in ("minimap", "prompt", "wanted"):
        state = "present" if reading.present else "absent"
        return (f"  {name:<9} {state:<8} conf={reading.confidence:.2f}  "
                f"box={tuple(reading.bbox)}")
    value = "--" if reading.value is None else f"{reading.value * 100:.0f}%"
    return (f"  {name:<9} {value:<8} conf={reading.confidence:.2f}  "
            f"box={tuple(reading.bbox)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.backend:
        cfg.capture.backend = args.backend
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    perception = Perception(cfg.vision)
    saved = 0
    last_id = -1
    deadline = time.monotonic() + 15.0
    while saved < args.frames and time.monotonic() < deadline:
        packet = capture.buffer.latest()
        if packet is None or packet.frame_id <= last_id:
            time.sleep(0.05)
            continue
        last_id = packet.frame_id
        pres = perception.process(packet.image, packet.frame_id)
        detection = pres.hud
        if args.region:
            print(f"frame #{packet.frame_id} region={args.region}")
            reading = detection.gauges.get(args.region)
            if reading is None:
                reading = getattr(detection, args.region, None)
            print(_reading_line(args.region, reading))
        else:
            fh, fw = packet.image.shape[:2]
            boxes = scale_regions(cfg.vision.hud.regions, fw, fh)
            print(f"frame #{packet.frame_id} ({fw}x{fh}) latency="
                  f"{detection.latency_ms:.1f}ms ocr={pres.ocr_engine}")
            for name in ("health", "stamina", "dead_eye", "minimap", "prompt", "wanted"):
                print(_reading_line(name, detection.gauges.get(name)
                                    if name in detection.gauges
                                    else getattr(detection, name, None)))
            print(f"  scaled boxes: {boxes}")
        path = out_dir / f"hud_{packet.frame_id:06d}.jpg"
        cv2.imwrite(str(path), _annotated(packet.image, detection),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  saved {path}")
        saved += 1
        time.sleep(0.2)

    capture.stop()
    if saved == 0:
        print("no frames captured")
        return 1
    print("tune vision.hud.regions in config.yaml until boxes hug the HUD "
          "elements, then re-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
