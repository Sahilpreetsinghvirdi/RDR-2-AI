# RDR2 AI - Autonomous Red Dead Redemption 2 Story Mode Agent

A local, modular AI agent that plays Red Dead Redemption 2 **Story Mode** on Windows.
It perceives the game through screen capture and acts through normal keyboard and
mouse input - exactly like a human player. It does not modify game files, inject
code into the game, or interact with Red Dead Online.

> **Story Mode only.** This project must never be used with Red Dead Online.
> All input is sent through the standard Windows input pipeline (SendInput), so
> the game sees ordinary keystrokes and mouse movement.

## What it does (Phase 1)

- Detects the RDR2 window by title and captures only its client area.
- Captures frames at 30 fps via `dxcam` (DXGI Desktop Duplication) with an
  automatic fallback to `mss`, plus a `synthetic` backend for development
  without the game.
- Sends keyboard/mouse input through `SendInput` (scancode mode, which is
  raw-input friendly; `virtual_key` fallback available).
- Runs an independent, high-priority hotkey thread that works even if the
  decision loop freezes:
  - **F12** - emergency stop (release all inputs, halt everything)
  - **F11** - pause / resume
  - **F10** - human takeover / return control to the AI
- Enforces safety in depth: state machine, input guards (never sends input
  unless the game window is focused), a watchdog that releases inputs if any
  critical loop stalls, and a configurable startup grace period.
- Streams structured telemetry: console logs, rotating log files,
  `logs/events.jsonl` (metrics, state transitions, shutdown summary), and a
  debug dashboard window showing the captured frame, overlay, and live status.
- Optionally records frames and state to `recordings/` for dataset building.

## Architecture

```
src/
  main.py              entry point: thread wiring, main decision loop
  config.py            typed config dataclasses loaded from config.yaml
  capture/
    window_manager.py  find/track/focus the game window (Win32)
    backends.py        dxcam / mss / synthetic frame grabbers + color order
    screen_capture.py  capture thread, ring buffer, frame stats, fallback
  input/
    win32_input.py     SendInput structures and helpers
    keys.py            key names -> virtual-key mapping
    keyboard.py        keyboard backend (scancode / virtual_key)
    mouse.py           mouse backend (move / click / wheel / drag)
    input_controller.py guarded public API (hold, press, click, release_all)
  safety/
    state_machine.py   INIT/RUNNING/PAUSED/TAKEOVER/STOPPED/ERROR
    emergency_stop.py  global hotkey listener thread
    watchdog.py        heartbeats + stall faults
  state/
    game_state.py      GameState / AgentStatus data models
  vision/
    fast_pass.py       cheap grayscale downscale stats (motion/brightness)
  telemetry/
    logger.py          console + rotating file logs, event log
    recorder.py        frame/state recording with fps rate limiting
  ui/
    dashboard.py       debug window (frame + overlay + status panel)
    debug_overlay.py   HUD-style annotations drawn on the frame
  demo.py              Phase 1 input demonstration
  planning/ control/ ai/ ai/rl/ mission/   reserved for later phases
tools/
  calibrate_capture.py measure capture fps/latency
  inspect_frames.py    save sample frames for verification
  input_check.py       test keyboard/mouse injection
tests/                 123 tests (pytest)
```

Control flow: `capture thread -> frame buffer -> main loop (state snapshot,
fast-pass vision, planner stub, input controller) -> telemetry/overlay`.
Safety paths (`emergency_stop`, `watchdog`, `input guards`) sit alongside and
can cut input at any moment regardless of loop state.

## Installation

Requirements: Windows 10/11, Python 3.12+ (64-bit), one monitor at 1920x1080
or similar.

```powershell
# from the project root
.\setup.ps1
```

This creates `.venv`, installs runtime + dev dependencies
(`requirements.txt` / `requirements-dev.txt`), and verifies imports.

## RDR2 setup

1. Launch RDR2 **Story Mode** (not Red Dead Online) in windowed or borderless
   windowed mode at 1920x1080.
2. In game settings, keep the HUD enabled - Phase 1 does not require it, but
   later phases read HUD elements.
3. Note the game window title; the default patterns are
   `Red Dead Redemption 2` and `RDR2` (see `config.yaml`).
4. Start the agent and press F10 (human takeover) if you want to play
   yourself while it runs.

## Calibration

Verify what the capture layer sees before running the agent:

```powershell
.\calibrate.ps1                      # measure capture fps + latency (5 s)
.\calibrate.ps1 -Seconds 10 -Backend mss
.\calibrate.ps1 -Inspect             # save 5 sample frames to recordings/
.\calibrate.ps1 -Inspect -Full       # whole screen instead of window ROI
```

Sample frames should show correct colors (not red/blue swapped). If colors are
wrong, set `capture.color_order` explicitly to `bgr`, `rgb`, or `bgra`.

## Running

```powershell
.\run.ps1                     # normal run (debug window on)
.\run.ps1 -Demo               # Phase 1 input demonstration
.\run.ps1 -Headless -Record   # no debug window, record frames
.\run.ps1 -Backend synthetic  # development without the game
.\run.ps1 -ExtraArgs @("--duration","30")

# or directly:
.\.venv\Scripts\python.exe -m src.main --help
.\.venv\Scripts\python.exe -m src.main --set capture.target_fps=60
```

CLI flags: `--config`, `--set KEY=VALUE` (repeatable), `--demo`, `--headless`,
`--record`, `--backend {auto,dxcam,mss,synthetic}`, `--duration N`,
`--version`.

## Emergency stop

**Press F12 at any time.** The hotkey thread polls `GetAsyncKeyState` at 250 Hz
in its own high-priority thread, so it works even if the planner, model, or
main loop is stuck. F12 refuses new decisions, releases every held key/button,
and halts the agent into `STOPPED`.

- **F11** pause/resume (releases inputs on pause).
- **F10** human takeover: the agent stops sending input; press again to resume.
- The watchdog independently releases inputs if any critical loop stops
  beating for `safety.heartbeat_stall_s` (default 2 s).
- Input is only ever sent while the game window is focused
  (`window.focus_before_input`).

## Debugging

- `logs/rdr2ai_*.log` - full application log (rotating, 10 MB x 3).
- `logs/events.jsonl` - machine-readable events: state transitions, per-second
  metrics (fps, latency, input counters), shutdown summary.
- `logs/effective_config.yaml` - the exact config used for this run.
- Debug window - live frame with overlay and status panel (disable with
  `--headless` or `debug.gui: false`).
- `.\calibrate.ps1 -Inspect` - save frames to `recordings/inspect/`.
- `tools\input_check.py` - verify keyboard/mouse injection.
- Run tests: `.\.venv\Scripts\python.exe -m pytest -q`
- Lint: `.\.venv\Scripts\python.exe -m ruff check .`

## Configuration

All tunables live in `config.yaml`; source code must not hardcode them.

| Section     | Highlights |
|-------------|------------|
| `agent`     | loop rate (`loop_hz`), phase label |
| `window`    | `title_patterns`, `exclude_patterns`, focus rules, poll intervals |
| `capture`   | backend, `target_fps`, `color_order`, `roi_mode`, synthetic size |
| `input`     | `keyboard_mode` (scancode/virtual_key), hold/verify timing, clamps |
| `safety`    | F12/F11/F10 bindings, watchdog stalls, startup grace, release rules |
| `vision`    | fast-pass work image size and cadence |
| `debug`     | dashboard window name, render fps, overlay, panel width |
| `recording` | fps, format, max frames, state jsonl |
| `telemetry` | console/file logs, `events.jsonl`, metrics interval, log level |

Overrides can be applied per run with `--set capture.target_fps=60`.
Unknown keys are reported and ignored, so typos do not silently do nothing.

## Performance notes

Target machine: Intel iGPU / CPU laptop, 16 GB RAM, single 1920x1080 display.
Defaults are tuned for that:

- `capture.target_fps: 30` - comfortable for iGPU; raise to 60 only after
  measuring with `.\calibrate.ps1`.
- `dxcam` is the preferred backend (GPU-side duplication, ~3-5 ms frames);
  `mss` fallback costs more CPU.
- Vision Phase 1 uses a 160x90 grayscale pass, so the full frame is only
  decoded for display/recording.
- Recording writes JPEG at a capped rate; `recording.enabled` stays off by
  default to keep disk quiet.

Measured on this machine: dxcam ~23 fps at 1920x1080, ~26 ms capture latency;
synthetic backend ~24 fps, ~3 ms latency.

## Roadmap (phases)

1. **Phase 1 (done)** - window detection, capture backends, input controller,
   state machine, emergency stop, watchdog, telemetry, debug dashboard.
2. **Phase 2** - HUD/OCR reading, richer debug overlay, perception pipeline.
3. **Phase 3** - world state estimation (player, horse, weapons, minimap).
4. **Phase 4** - navigation and locomotion control (mouse look, movement).
5. **Phase 5** - mission/task framework and goal management.
6. **Phase 6** - dialogue, encounters, and camp interactions.
7. **Phase 7** - combat and self-defense behaviors.
8. **Phase 8** - survival systems (food, camp, crafting, economy).
9. **Phase 9** - learned/optimized policies (RL) on top of scripted skills.
10. **Phase 10** - robustness, long-run autonomy, packaging and docs.

Each phase ships with tests and stays revertible: the safety layer and manual
hotkeys are never bypassed by later phases.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `capture backend selected: mss` + slow fps | dxcam unavailable; reinstall `dxcam`, run as the same user as the desktop session. |
| Frames are 0, log says `waiting_for_window` | RDR2 Story Mode is not running, or the title does not match `window.title_patterns`. Check `.\calibrate.ps1 -Inspect`. |
| Colors look swapped in saved frames | Set `capture.color_order` explicitly (`bgr`/`rgb`); dxcam delivers RGB, mss delivers BGRA. |
| Input has no effect in game | Click the game window first; input is refused unless the game is focused. Try `input.keyboard_mode: virtual_key` if scancodes are ignored. |
| `hotkeys not armed` / F12 ignored | The agent process must be running; check `logs/rdr2ai_*.log` for the `hotkeys armed` line. |
| Watchdog faults at startup | Lower `agent.loop_hz` or raise `safety.heartbeat_stall_s`; check `logs/events.jsonl` for the stalled component. |
| Debug window eats CPU | Run `.\run.ps1 -Headless` or set `debug.gui: false`. |
| Wrong monitor captured | Set `capture.monitor_index` and verify with `.\calibrate.ps1 -Inspect -Full`. |
| Pytest import errors | Run from the project root: `.\.venv\Scripts\python.exe -m pytest -q` (uses `pythonpath = ["."]`). |

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest -q      # 123 tests
.\.venv\Scripts\python.exe -m ruff check .   # lint
```

Conventions: type hints everywhere, no hardcoded tunables (config only),
docstrings for public APIs, and the safety layer must remain reachable from
any state.

## Disclaimer

This is an unofficial, educational project, unaffiliated with Rockstar Games.
It targets single-player Story Mode only; using automation in Red Dead Online
may violate the game's terms of service. Use at your own risk, and keep F12
under your finger while the agent is running.
