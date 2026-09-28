"""Guided live control verification (Phase 11).

Walks through every input primitive the agent can send - movement, camera
turns, taps, survival macro keys and the combat mouse buttons - shows what
you should see in game (cross-referenced against ``docs/CONTROLS.md``),
sends it through the guarded ``InputController`` and records your y/n
verdict per step.

    .\\run.ps1 -ControlTest                # interactive, game focused
    python -m src.control_test --dry-run  # print the plan, send nothing

Input is only sent while the game window is focused (unless
``--allow-unfocused``). Ctrl+C aborts and releases everything.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.capture.window_manager import (
    GameWindowManager,
    enable_dpi_awareness,
    make_focus_check,
)
from src.config import AppConfig, ConfigError, load_config
from src.input.input_controller import InputController

REFERENCE: dict[str, str] = {
    "w": "Forward movement (on foot, on horse, on wagon) = W",
    "s": "Backward movement = S",
    "a": "Left movement = A",
    "d": "Right movement = D",
    "shift": "Sprint / Run / Increase speed = Shift",
    "space": "Jump = Space-bar / Horse jump = Space-bar",
    "e": "Interact with object = E / Use = E",
    "i": "Quick use item = I",
    "h": "Whistle for horse = H",
    "r": "Reload = R / Pickup item = R",
    "q": "Enter cover = Q",
    "f": "Melee - punch = F",
    "b": "Open satchel = B",
    "g": "Interact with animal = G",
    "tab": "Open weapon wheel = Tab / Holster = Tab (tap)",
    "ctrl": "Crouch / Stealth = Ctrl / Brake = Ctrl",
    "x": "Expand radar = X / Call animal = X",
    "c": "Regular radar = C / Look behind (3rd person) = C",
    "v": "Cycle camera views = V",
    "z": "Compass radar = Z",
    "left": "Attack / Fire weapon = Left mouse button",
    "right": "Aim = Right mouse button",
}
REFERENCE_FALLBACK = "see docs/CONTROLS.md (PC controls reference)"
MOVE_SWEEP_DEGREES = 20.0

KINDS = ("hold", "move", "click")


@dataclass(frozen=True)
class ControlStep:
    name: str
    kind: str        # hold = key hold | move = mouse pan | click = mouse button
    target: str      # key name, direction (left/right) or button (left/right)
    duration_s: float
    expect: str      # what the player should observe in game
    reference: str   # matching entry from docs/CONTROLS.md
    caution: bool = False  # fires a weapon; needs an explicit "y"

    @property
    def action_label(self) -> str:
        if self.kind == "hold":
            return f"HOLD '{self.target}' for {self.duration_s:.2f}s"
        if self.kind == "move":
            return f"mouse move {self.target} (camera pan)"
        return f"CLICK {self.target} mouse button for {self.duration_s:.2f}s"


def _ref(key: str) -> str:
    return REFERENCE.get(key.lower(), REFERENCE_FALLBACK)


def build_steps(cfg: AppConfig) -> list[ControlStep]:
    """Every input primitive the agent can send, in a safe test order."""
    keys = cfg.control.keys
    steps = [
        ControlStep("walk forward", "hold", keys["forward"], 0.6,
                    "Character walks forward.",
                    _ref(keys["forward"])),
        ControlStep("walk backward", "hold", keys["back"], 0.6,
                    "Character steps backward.",
                    _ref(keys["back"])),
        ControlStep("strafe left", "hold", keys["strafe_left"], 0.5,
                    "Character sidesteps left (camera unchanged).",
                    _ref(keys["strafe_left"])),
        ControlStep("strafe right", "hold", keys["strafe_right"], 0.5,
                    "Character sidesteps right (camera unchanged).",
                    _ref(keys["strafe_right"])),
        ControlStep("sprint", "hold", keys["sprint"], 1.0,
                    "Character sprints while held, walks again on release.",
                    _ref(keys["sprint"])),
        ControlStep("camera pan left", "move", "left", 0.0,
                    "Camera pans left (mouse look).",
                    "Look = mouse movement (implicit in the controls list)"),
        ControlStep("camera pan right", "move", "right", 0.0,
                    "Camera pans right (mouse look).",
                    "Look = mouse movement (implicit in the controls list)"),
        ControlStep("jump", "hold", keys["jump"], 0.15,
                    "Character jumps (on a horse: jumps an obstacle).",
                    _ref(keys["jump"])),
        ControlStep("interact tap", "hold", keys["interact"], 0.15,
                    "Near a prompt: the context action starts; otherwise nothing.",
                    _ref(keys["interact"])),
        ControlStep("horse whistle", "hold", keys["whistle"], 0.3,
                    "A nearby horse starts coming to you; otherwise nothing.",
                    _ref(keys["whistle"])),
    ]
    steps.extend(_remedy_steps(cfg))
    steps.append(ControlStep(
        "aim (right mouse)", "click", "right", 0.5,
        "Over-the-shoulder aim while held, normal view after release.",
        _ref("right"),
    ))
    steps.append(ControlStep(
        "fire (left mouse)", "click", "left", 0.2,
        "Weapon FIRES - keep the camera aimed away from people and horses.",
        _ref("left"), caution=True,
    ))
    return steps


def _remedy_steps(cfg: AppConfig) -> list[ControlStep]:
    movement = {value.lower() for value in cfg.control.keys.values()}
    seen: set[str] = set(movement)
    steps: list[ControlStep] = []
    for remedy in cfg.survival.remedies.values():
        raw_keys = remedy.keys if hasattr(remedy, "keys") else []
        if isinstance(remedy, dict):
            raw_keys = remedy.get("keys", [])
        for key in raw_keys:
            name = str(key).lower()
            if name in seen:
                continue
            seen.add(name)
            steps.append(ControlStep(
                f"quick-use '{name}'", "hold", name, 0.15,
                f"Whatever '{name}' is bound to in game fires (e.g. quick use).",
                _ref(name),
            ))
    return steps


def execute_step(step: ControlStep, controller: InputController,
                 move_px: int) -> bool:
    if step.kind == "hold":
        return controller.hold(step.target, step.duration_s)
    if step.kind == "move":
        dx = -abs(move_px) if step.target == "left" else abs(move_px)
        return controller.mouse_move(dx, 0)
    if step.kind == "click":
        return controller.mouse_click(step.target, int(step.duration_s * 1000))
    raise ValueError(f"unknown step kind: {step.kind!r}")


def print_plan(steps: Sequence[ControlStep], out: Callable[[str], None]) -> None:
    out(f"control test plan ({len(steps)} steps):")
    for index, step in enumerate(steps, 1):
        marker = " [CAUTION: fires]" if step.caution else ""
        out(f"  {index:2}. {step.name}{marker}")
        out(f"      agent sends: {step.action_label}")
        out(f"      expect:      {step.expect}")
        out(f"      reference:   {step.reference}")


def run_steps(
    steps: Sequence[ControlStep],
    execute: Callable[[ControlStep], bool],
    input_fn: Callable[[str], str],
    out: Callable[[str], None],
    countdown_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, int, int, list[str]]:
    """Interactive loop; returns (passed, failed, skipped, failed_names)."""
    passed = failed = skipped = 0
    failures: list[str] = []
    for index, step in enumerate(steps, 1):
        out(f"[{index}/{len(steps)}] {step.name}")
        out(f"  agent will:  {step.action_label}")
        out(f"  expect:      {step.expect}")
        out(f"  reference:   {step.reference}")
        prompt = "  send it? [enter]=yes  n=skip  q=quit > "
        if step.caution:
            out("  CAUTION: this fires a weapon!")
            prompt = "  send it? [enter]=skip  y=fire  q=quit > "
        choice = input_fn(prompt).strip().lower()
        if choice == "q":
            remaining = len(steps) - index + 1  # current + untried steps
            skipped += remaining
            out(f"  aborted at step {index}/{len(steps)} "
                f"({remaining} marked skipped)")
            break
        if choice in {"n", "s", "no"}:
            skipped += 1
            out("  skipped")
            continue
        if step.caution and choice not in {"y", "yes"}:
            skipped += 1
            out("  skipped (caution: type y to fire)")
            continue
        out(f"  sending in {countdown_s:.0f}s ...")
        sleep(countdown_s)
        if not execute(step):
            failed += 1
            failures.append(step.name)
            out("  FAILED (input refused or injection failed - focused?)")
            continue
        verdict = input_fn("  did it work? [enter]=yes  n=no > ").strip().lower()
        if verdict in {"n", "no"}:
            failed += 1
            failures.append(step.name)
            out("  marked FAILED")
        else:
            passed += 1
            out("  ok")
    return passed, failed, skipped, failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.control_test",
        description="Guided live verification of every input the agent can send.",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the step plan and exit without sending input")
    parser.add_argument("--allow-unfocused", action="store_true",
                        help="send input even if the game window is not focused")
    parser.add_argument("--countdown-s", type=float, default=1.0,
                        help="pause before each input so you can watch")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    steps = build_steps(cfg)

    if args.dry_run:
        print_plan(steps, print)
        print("dry run - nothing was sent; rerun without --dry-run to execute")
        return 0

    enable_dpi_awareness()
    wm = GameWindowManager(cfg.window)
    focus_check = make_focus_check(
        wm, not args.allow_unfocused, cfg.window.focus_timeout_s
    )
    controller = InputController(cfg.input, allow_input=focus_check)
    move_px = max(1, int(cfg.control.mouse_px_per_degree * MOVE_SWEEP_DEGREES))
    print(
        f"control test: {len(steps)} steps | focus the game window now | "
        "Ctrl+C aborts and releases everything"
    )
    passed = failed = skipped = 0
    failures: list[str] = []
    try:
        passed, failed, skipped, failures = run_steps(
            steps,
            execute=lambda step: execute_step(step, controller, move_px),
            input_fn=input,
            out=print,
            countdown_s=max(0.0, args.countdown_s),
        )
    except KeyboardInterrupt:
        print("\naborted by user")
    finally:
        controller.release_all()

    print(
        f"summary: passed={passed} failed={failed} skipped={skipped} "
        f"sent={controller.stats.sent} refused={controller.stats.refused}"
    )
    if failures:
        print("failed steps: " + ", ".join(failures))
    if controller.stats.refused and not args.allow_unfocused:
        print("hint: refused input means the game window was missing or unfocused")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
