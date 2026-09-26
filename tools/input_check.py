"""Send single test inputs and verify they were injected.

Examples:
    python tools/input_check.py --key w --hold 0.5
    python tools/input_check.py --mouse 150 0
    python tools/input_check.py --click right --allow-unfocused

By default the RDR2 window must exist and be focused; --allow-unfocused lifts
that restriction (useful when testing against Notepad or the desktop).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capture.window_manager import (  # noqa: E402
    GameWindowManager,
    enable_dpi_awareness,
    make_focus_check,
)
from src.config import load_config  # noqa: E402
from src.input.input_controller import InputController  # noqa: E402
from src.input.keys import NAMED_KEYS, normalize_key  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Input injection check.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--key", default=None, help="key name, e.g. w, shift, f12")
    parser.add_argument("--hold", type=float, default=0.15, help="hold duration seconds")
    parser.add_argument("--mouse", nargs=2, type=int, metavar=("DX", "DY"), default=None)
    parser.add_argument("--click", default=None, help="mouse button: left/right/middle")
    parser.add_argument("--allow-unfocused", action="store_true")
    parser.add_argument("--list-keys", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_keys:
        print("known key names:", ", ".join(sorted(NAMED_KEYS)))
        print("also: single letters a-z and digits 0-9")
        return 0
    if args.key is None and args.mouse is None and args.click is None:
        print("nothing to do: pass --key, --mouse or --click (see --help)")
        return 2

    cfg = load_config(args.config)
    enable_dpi_awareness()
    wm = GameWindowManager(cfg.window)
    focus_check = make_focus_check(
        wm, focus_before_input=not args.allow_unfocused,
        focus_timeout_s=cfg.window.focus_timeout_s,
    )

    def allow_any() -> bool:
        return True

    allow = allow_any if args.allow_unfocused else focus_check

    controller = InputController(cfg.input, allow_input=allow)
    failures = 0

    if args.key is not None:
        key = normalize_key(args.key)
        print(f"holding key '{key.name}' for {args.hold:.2f}s ...")
        ok = controller.hold(key, args.hold)
        print(f"  -> {'OK' if ok else 'FAILED'} (latency "
              f"{controller.stats.last_latency_ms:.1f}ms)")
        failures += 0 if ok else 1

    if args.mouse is not None:
        dx, dy = args.mouse
        print(f"moving mouse by ({dx}, {dy}) ...")
        ok = controller.mouse_move(dx, dy)
        print(f"  -> {'OK' if ok else 'FAILED'}")
        failures += 0 if ok else 1

    if args.click is not None:
        print(f"clicking '{args.click}' ...")
        ok = controller.mouse_click(args.click)
        print(f"  -> {'OK' if ok else 'FAILED'}")
        failures += 0 if ok else 1

    time.sleep(0.1)
    controller.release_all()
    print(f"summary: sent={controller.stats.sent} refused={controller.stats.refused} "
          f"failed={controller.stats.failed}")
    if controller.stats.refused and not args.allow_unfocused:
        print("hint: input was refused because the game window is missing or unfocused")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
