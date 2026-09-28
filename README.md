# RDR2 AI - Autonomous Red Dead Redemption 2 Story Mode Agent

A local, modular AI agent that plays Red Dead Redemption 2 **Story Mode** on Windows.
It perceives the game through screen capture and acts through normal keyboard and
mouse input - exactly like a human player. It does not modify game files, inject
code into the game, or interact with Red Dead Online.

> **Story Mode only.** This project must never be used with Red Dead Online.
> All input is sent through the standard Windows input pipeline (SendInput), so
> the game sees ordinary keystrokes and mouse movement.

## What it does

### Phase 1 - foundation
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

### Phase 2 - perception (HUD reading)
- Region-based HUD detectors (`vision.hud.regions`, fraction-of-frame boxes):
  - **Core gauges** (health / stamina / dead-eye): polar ring-fill read with
    contrast-based confidence. Below the confidence threshold the value stays
    *unknown* instead of guessed, and fully depleted cores read as unknown
    (their track is indistinguishable from "hidden").
  - **Minimap**: presence via colour-histogram distance to its surroundings.
  - **Interaction prompts**: near-white text with enough edge density.
  - **Wanted stars**: bright-blob counting (feeds `ThreatState.wanted_level`).
- Optional **OCR** for prompt/objective text via Tesseract
  (`vision.ocr.engine: auto`). If Tesseract is not installed the agent runs
  exactly the same, just without text - nothing fails.
- Detections project onto the shared `GameState` (player health/stamina/
  dead-eye, threat, mission prompt) with per-field confidence; untouched
  fields stay `None`.
- The debug dashboard shows a `HUD : HP .. ST .. DE .. map .. ocr=..` row,
  an optional `PROMPT :` line, and yellow boxes around detected regions.
- `logs/events.jsonl` metrics now include a `hud` block; prompt text changes
  are emitted as `prompt` events.
- `.\calibrate.ps1 -Hud` runs the detectors on live frames and saves
  annotated previews so regions can be tuned to your screen layout.

### Phase 3 - world state estimation
- New `src/vision/world.py` estimators, all region-based with confidence:
  - **Minimap**: player-marker blob (position offset), colour fractions
    classified as water / road / vegetation -> `EnvironmentState` hints.
  - **Sky**: top-strip luminance and warmth -> `time_of_day`
    (day/dusk/night), uniformity -> `weather` (clear/cloudy).
  - **Weapon**: ammo-counter visibility; the number is read via OCR
    (throttled independently from prompt OCR) -> `player.weapon_ammo`.
  - **Horse cores**: health/stamina rings that appear above yours when the
    horse is nearby -> `HorseState` (`detected` only; `mounted` stays
    unknown until a later phase proves it).
- Detections fill `GameState.environment`, `GameState.horse` and
  `GameState.player.weapon_ammo`; `confidence.navigation/horse` move with
  them. Nothing is inferred beyond what is on screen.
- Debug window: `WORLD : dusk clear ammo 24 horse Y` row, world boxes in
  the overlay, and a `world` block in `logs/events.jsonl` metrics.
- Config: new `vision.world` section (regions + thresholds).

### Phase 4 - locomotion control (in progress)
- New `src/control/locomotion.py`: non-blocking `LocomotionController` -
  `move()` holds movement keys for a clamped duration (releases on `tick()`),
  `turn()` converts degrees to clamped mouse deltas, `tap()` for jump/
  interact, `stop()` releases everything. Commands never sleep the loop.
- New `src/planning/scripted.py`: `ScriptedPlanner` walks a fixed
  look-around / walk / sprint / jump cycle so the whole control path is
  exercised before a real decision layer replaces it.
- Autopilot is **off by default** (`control.enabled: false` or `--autopilot`).
  While active it still only runs in `RUNNING`, and input stops when the game
  window is missing or unfocused, on pause/takeover/F12, and on shutdown.
- Route navigation (`control.mode: route`): `src/planning/route.py` follows
  `[bearing_deg, meters]` legs from `control.nav.legs` - turns to each bearing
  with clamped mouse steps, walks in short bursts while dead-reckoning
  distance, auto-sprints beyond `sprint_after_m`, and marks a leg **blocked**
  when the minimap stops moving (optical flow under `stall_flow_eps` for
  `stall_timeout_s`) instead of walking in place. Completion shows as
  `ROUTE n/N` / `ROUTE ARRIVED` in the status panel.
- Window matching hardened: `window.match: exact` requires the whole title to
  equal a pattern (browser tabs mentioning the game no longer match), and
  console/terminal/browser window classes are always excluded.

### Phase 5 - mission/task framework
- New `src/mission/tasks.py`: config-declared tasks - `turn` (relative
  degrees), `walk` (meters, optional bearing), `wait` (seconds),
  `wait_for` (dotted GameState field equals a value, optional timeout),
  `interact` (tap E), `log` (marker). Strict parsing at startup: a bad task
  is a config error naming `mission.tasks[i]`, never a runtime surprise.
- New `src/mission/runner.py`: `MissionRunner` executes the list one control
  tick at a time, duck-typed with the other planners. Walking reuses the
  single-leg route planner (same clamps and stall detection); turning uses
  the same clamped mouse steps; heading carries across tasks.
- Failure policy: a refused key, blocked walk, or `wait_for` timeout parks
  the mission as `MISSION FAILED` with the reason, releases all input, and
  never auto-retries. Success shows `MISSION n/N` then `MISSION DONE`
  (`mission.loop: true` restarts the list instead).
- Enabled via `mission.enabled: true` or `--mission`; takes precedence over
  `control.mode`. Off by default.

### Phase 6 - dialogue, encounters, prompt reactions
- New `src/vision/dialogue.py`: subtitle-band presence detection - a
  bright-pixel fraction test over `vision.dialogue.region` (bottom-center
  band) marks subtitles visible or not; `match_keyword()` checks configured
  encounter keywords against on-screen text (case-insensitive, first match).
- Perception OCRs the subtitle band only while it is visible (every
  `dialogue.every_n_frames` frames) and matches encounter keywords against
  text that is **on screen right now**: the visible prompt or the visible
  subtitle - a stale prompt never re-fires after it disappears.
- New `DialogueState` on `GameState` (`active`/`text`/`encounter`), plus a
  `DIALOG` overlay line and a `dialogue` telemetry block.
- New `src/mission/respond.py`: `PromptResponder` taps the key mapped in
  `vision.dialogue.respond_to` (e.g. `greet -> interact`) when an encounter
  keyword is detected, rate-limited by `respond_cooldown_s`; a refused tap
  does not consume the cooldown. Off while the Phase 1 demo runs.
- Arm it by filling `vision.dialogue.respond_to` (empty by default - the
  agent observes encounters without reacting until you choose mappings).

### Phase 7 - combat and self-defense
- Minimap threat detection (Phase 3 extension): red enemy blips are counted
  in the minimap region (`vision.world.enemy_*` thresholds) and the nearest
  one yields a direction offset on `ThreatState`.
- New `src/ai/threat.py`: `ThreatAssessor` (stateful, input-free) watches the
  health gauge for a drop inside `combat.health_drop_window_s` to flag
  `incoming_fire`, and maintains the level ladder
  `none < enemy < wanted < under_fire` from detected signals only.
- New `src/planning/combat.py`: `CombatPlanner` engages on wanted stars,
  minimap blips, or incoming fire; aims at the nearest blip's minimap offset
  (clamped corrections, dead zone), fires config-clamped bursts
  (`burst_s` / `burst_interval_s`, never with 0 ammo), retreats below
  `retreat_health_frac`, and stands down after the threat clears for
  `disengage_clear_s`. While engaged it replaces the navigation planner.
- Safety unchanged: `combat.enabled: false` by default, every button event
  still passes the focus/state guard, and pause/takeover/lost focus/window
  loss all release the fire button. Off by default.

### Phase 8 - survival systems (provisioning)
- New `src/planning/survival.py`: `SurvivalPlanner` watches the HUD-derived
  cores and runs a remedy's key sequence when any tracked core drops below
  its threshold (`needs: {health: 0.4, ...}` = any-of). Sequences are raw
  key names (`keys: [i, ...]`) tuned to *your* provisioning binds - nothing
  about the in-game menus is assumed or hardcoded.
- Discipline: each remedy has a `cooldown_s`, must re-arm only after every
  known core recovers past threshold + `recover_margin` (hysteresis), and
  with `require_clear: true` nothing starts or continues while a threat is
  active (mid-sequence threats abort the run). A refused key press retries
  the step instead of skipping it.
- State: `GameState.survival` (`active_remedy`, `low_cores`) with a
  `SURV` overlay line and `survival` telemetry block; while a remedy runs it
  is the primary planner (navigation pauses for those ticks).
- Honest gaps: `GameState.economy.cash` stays `None` - cash and crafting
  live behind pause menus that the gameplay HUD never shows; they are not
  guessed. Off by default (`survival.enabled: false`).

### Phase 9 - learned policies (RL, best-effort research scaffold)
- New `src/ai/rl/` package: record -> train offline -> act. The policy is a
  tiny numpy MLP (one tanh hidden layer) over a fixed, versioned observation
  vector built from `GameState` (`features.py`) - unknown fields are `0.0`
  with a presence flag of `0.0`, never guessed. The action space is eight
  *scripted skills* (`noop`, `forward`, `back`, `strafe_*`, `turn_*`,
  `sprint`) executed through the Phase 4 locomotion planner, so the policy
  never touches raw keys and keeps every focus/state guard.
- Two modes (`rl.mode`): `policy` acts from a checkpoint - **inactive and
  silent until `learning/policy.npz` loads** (missing/corrupt/mismatched
  checkpoints log a reason instead of acting); `random` explores the skill
  space to bootstrap `learning/buffer.npz`. Transitions record
  (observation, action, reward) while acting, with episode boundaries at
  `rl.boundary_gap_s` so Monte-Carlo returns do not bleed across sessions.
- Training is offline and numpy-only: `python -m src.ai.rl.train --data
  learning/buffer.npz --out learning/policy.npz` (REINFORCE with a mean
  baseline by default, or `--method bc` for behavior cloning).
- Rewards come from known deltas only (`rl.reward`: core changes minus a
  threat-level penalty); an unknown core contributes nothing.
- Wiring: the policy planner is the primary driver when `rl.enabled` (it
  conflicts with `mission.enabled`/`control.enabled` by config validation),
  yields to combat and survival, and shows an `RL` overlay line plus an
  `rl` telemetry block. Buffer saves on shutdown (`rl_buffer` event).
- Honest scope: this is a research scaffold, not a game-winning agent -
  REINFORCE here is high-variance and there is no reward shaping for
  objectives/missions yet. Off by default (`rl.enabled: false`).

### Phase 10 - robustness, long-run autonomy, packaging and docs
- `ComponentGuard` (`src/safety/resilience.py`): any stepping component
  (planner, mission runner, responder, assessor, combat, survival, demo,
  RL policy) that throws is **disabled for the session** - inputs are
  released immediately, a `component_fault` event is emitted, the status
  line shows the error, and the loop keeps running observation-only
  instead of crashing. Disabled components appear under `guard` in
  metrics. Perception failures still propagate (never act on broken
  vision).
- `.\run.ps1 -Doctor` / `python -m src.doctor`: environment self-check
  with PASS/WARN/FAIL rows - interpreter, imports, Tesseract, config
  validity (warnings included), log writability, game-window presence,
  key map sanity. Exits 1 only on FAIL; run it before any hands-on test.
- Long-run memory bounds: `rl.buffer_max` caps recorded transitions
  (oldest row dropped, episode boundary promoted), so multi-hour sessions
  cannot grow the RL buffer without limit.
- Packaging polish: `pytesseract` declared in `pyproject.toml`,
  version bumped to 0.2.0, `-Doctor` switch in `run.ps1`.
- Hands-on verification checklist (performed after all phases):
  1. `.\run.ps1 -Doctor` - all rows PASS or WARN.
  2. `.\run.ps1 -ControlTest` - guided live check of every input the
     agent can send (Phase 11); or quick single-key checks with
     `.\.venv\Scripts\python.exe tools\input_check.py`.
  3. `.\run.ps1 -Demo` with the game focused - observe the short input
     demo, F12 must kill it instantly.
  4. `.\run.ps1 -Autopilot -ExtraArgs @("--duration","30")` - scripted
     look/walk cycle with the HUD visible.
  5. Only then try mission/responder/combat/survival flags one at a time.

### Phase 11 - control verification harness
- `src/control_test.py` (`.\run.ps1 -ControlTest`): a guided, step-by-step
  live test of **every input primitive the agent can send** - movement
  (`control.keys`), camera pans, jump/interact taps, survival quick-use
  keys, aim, and fire. Each step prints what you should see in game and
  the matching command from [`docs/CONTROLS.md`](docs/CONTROLS.md), sends
  it through the guarded `InputController` (game-window focus required),
  and records your y/n verdict. The fire step is gated behind an explicit
  `y`. `--dry-run` prints the plan without sending anything.
- Abort any time with Ctrl+C - inputs are released in the shutdown path;
  focus loss mid-step refuses/aborts the input automatically.
- Summary line reports passed/failed/skipped plus sent/refused counters;
  exit code 1 if any step failed (so the run is scriptable).
- This is the "simple control verification" step of the agreed sequence:
  run it in Story Mode before any autonomous behavior, then the demo and
  autopilot runs.

### Story mode (Mode 1) - scripted story director

`src/story/runner.py` (NOTES Mode 1) plays a configured mission list
end-to-end behind the disabled-by-default `story.enabled` flag (or
`.\run.ps1 -Story`). Per mission the director runs up to four stages, then
advances:

- **mount** (optional, `mount: true`) - taps the horse whistle (`control.keys`
  `whistle`, H by default), waits for the `mount` prompt and boards; route
  legs afterwards are tracked at `story.mounted_speed_mps`. If the horse
  never comes, the mission continues on foot.
- **travel** - follows `legs: [[bearing, meters], ...]` route legs through the
  Phase 4 `RoutePlanner` (minimap stall detection included). With
  `seek_marker: true` the heading is continuously corrected toward a
  persistent minimap marker (the gold mission/stranger blip), so travel homes
  in like a human following the map instead of pure dead reckoning.
- **objective** - runs the profile's `tasks` through the Phase 5
  `MissionRunner` (`turn`/`walk`/`wait`/`wait_for`/`interact`/`key`/`log`;
  `key` taps any `control.keys` role, e.g. `{key: {name: whistle}}`).
- **complete** - watches the OCR'd objective text, prompt, subtitles and a
  center-screen banner region for `story.completion_keywords`
  ("mission passed", ...), then moves to the next profile.

Failures and stage timeouts (`mount_timeout_s`, `travel_timeout_s`,
`objective_timeout_s`, `completion_timeout_s`) skip forward instead of
parking the session, and the combat/survival gates pause the director without
burning stage time (pause gaps longer than `pause_grace_s` are rebased out of
the timers).

Side missions: profiles with `kind: stranger` are stranger/side content; with
`story.opportunistic` (default on) the director detours to one when its
marker is visible on the minimap, then resumes the story mission it left.

Honor: `story.auto_greet` (default on) arms the prompt responder to greet
every opportunity - the honor value itself is not readable from the HUD, so
the strategy is to greet whenever a prompt appears.

New perception: the `objective` HUD region (top-left) is edge/brightness
detected and OCR'd into `GameState.mission.objective_text`; tune the box with
`.\calibrate.ps1 -Hud` if objectives never appear. The minimap marker offset
feeds `GameState.mission.objective_location_estimate`, which drives seeking
and opportunistic detours.

`story` is a primary driver - it conflicts with `mission`, `control` and `rl`
(pick one). Example `config.yaml`:

```yaml
story:
  enabled: true
  auto_greet: true
  completion_keywords: [mission passed, mission complete]
  missions:
    - name: first mission
      mount: true
      seek_marker: true
      legs: [[0.0, 120.0], [45.0, 60.0]]    # bearing/meters to the start
      tasks:
        - {kind: interact}
        - {kind: wait, seconds: 2}
    - name: stranger on the road
      kind: stranger
      legs: [[10.0, 40.0]]
```

Route legs are dead-reckoned and real mission profiles need authoring +
playtesting against the live game (see NOTES.md); dismounting at the
destination is not implemented yet.

## Architecture

```
src/
  main.py              entry point: thread wiring, main decision loop
  doctor.py            self-check CLI (run with -Doctor before testing)
  control_test.py      Phase 11 guided live input verification
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
    resilience.py      ComponentGuard: fault isolation for long runs
    watchdog.py        heartbeats + stall faults
  state/
    game_state.py      GameState / AgentStatus data models
  vision/
    fast_pass.py       cheap grayscale downscale stats (motion/brightness)
    hud.py             region detectors: gauges, minimap, prompts, stars
    world.py           world estimators: minimap, sky, weapon, horse
    dialogue.py        subtitle band presence + encounter keywords (Phase 6)
    ocr.py             pluggable OCR (tesseract) with graceful fallback
    perception.py      per-frame orchestration -> GameState
  telemetry/
    logger.py          console + rotating file logs, event log
    recorder.py        frame/state recording with fps rate limiting
  ui/
    dashboard.py       debug window (frame + overlay + status panel)
    debug_overlay.py   HUD-style annotations drawn on the frame
  demo.py              Phase 1 input demonstration
  control/
    locomotion.py      Phase 4 tick-based movement holds and camera turns
  planning/
    scripted.py        Phase 4 scripted autopilot (look/walk cycle)
    route.py           Phase 4 route navigation (bearing/distance legs)
    combat.py          Phase 7 self-defense planner (aim/fire/retreat)
    survival.py        Phase 8 core-triggered provisioning macros
  mission/
    tasks.py           Phase 5 task model + config parser
    runner.py          Phase 5 mission executor (MISSION n/N status)
    respond.py         Phase 6 prompt responder (encounter -> key tap)
  story/
    runner.py          story director: travel -> objective -> complete -> next
  ai/
    threat.py          Phase 7 threat assessor (damage window, level ladder)
    rl/
      features.py      Phase 9 fixed observation featurizer (presence flags)
      actions.py       Phase 9 discrete action space (scripted skills)
      policy.py        Phase 9 numpy MLP policy (act, save/load)
      buffer.py        Phase 9 transition buffer (.npz, episode boundaries)
      reward.py        Phase 9 reward from known state deltas only
      train.py         Phase 9 offline trainer (REINFORCE / behavior cloning)
      planner.py       Phase 9 policy driver (record while acting)
tools/
  calibrate_capture.py measure capture fps/latency
  calibrate_hud.py     live HUD detection preview for region tuning
  inspect_frames.py    save sample frames for verification
  input_check.py       test keyboard/mouse injection
docs/
  CONTROLS.md          full PC controls reference + agent key cross-map
tests/                 579 tests (pytest)
```

Control flow: `capture thread -> frame buffer -> main loop (state snapshot,
fast-pass vision, HUD perception + OCR, planner/locomotion tick, input
controller) -> telemetry/overlay`.
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
   `Red Dead Redemption 2` and `RDR2`, matched **exactly** by default
   (`window.match: exact`) so look-alike windows are never used as the
   input target (see `config.yaml`).
4. Start the agent and press F10 (human takeover) if you want to play
   yourself while it runs.
5. Full PC keyboard/mouse command reference (all default game bindings,
   collected by the project owner): [`docs/CONTROLS.md`](docs/CONTROLS.md).
   It also maps every key the agent sends to the matching in-game command.

## Calibration

Verify what the capture layer sees before running the agent:

```powershell
.\calibrate.ps1                      # measure capture fps + latency (5 s)
.\calibrate.ps1 -Seconds 10 -Backend mss
.\calibrate.ps1 -Inspect             # save 5 sample frames to recordings/
.\calibrate.ps1 -Inspect -Full       # whole screen instead of window ROI
.\calibrate.ps1 -Hud                 # HUD detection preview (recordings/hud/)
.\calibrate.ps1 -Hud -Frames 5
```

Sample frames should show correct colors (not red/blue swapped). If colors are
wrong, set `capture.color_order` explicitly to `bgr`, `rgb`, or `bgra`.

The HUD preview prints every detector's pixel box and reading and saves
annotated images. Run it with Story Mode visible, then adjust
`vision.hud.regions` in `config.yaml` until the yellow boxes hug the actual
HUD elements (leave a small margin around each core).

## Running

```powershell
.\run.ps1 -Doctor                # environment self-check first
.\run.ps1 -ControlTest           # guided live input verification (Phase 11)
.\run.ps1 -ControlTest -ExtraArgs @("--dry-run")   # print the plan only
.\run.ps1                     # normal run (debug window on)
.\run.ps1 -Demo               # Phase 1 input demonstration
.\run.ps1 -Headless -Record   # no debug window, record frames
.\run.ps1 -Backend synthetic  # development without the game
.\run.ps1 -ExtraArgs @("--duration","30")

# or directly:
.\.venv\Scripts\python.exe -m src.main --help
.\.venv\Scripts\python.exe -m src.main --set capture.target_fps=60
```

CLI flags: `--config`, `--set KEY=VALUE` (repeatable), `--demo`, `--autopilot`,
`--mission`, `--headless`, `--record`,
`--backend {auto,dxcam,mss,synthetic}`, `--duration N`, `--version`.

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
| `window`    | `title_patterns`, `exclude_patterns`, `match` (exact/substring), focus rules, poll intervals |
| `capture`   | backend, `target_fps`, `color_order`, `roi_mode`, synthetic size |
| `input`     | `keyboard_mode` (scancode/virtual_key), hold/verify timing, clamps |
| `safety`    | F12/F11/F10 bindings, watchdog stalls, startup grace, release rules |
| `control`   | autopilot on/off, `mode` (script/route), movement keys, look sensitivity, `nav` legs/gains/stalls |
| `mission`   | task list (`turn`/`walk`/`wait`/`wait_for`/`interact`/`key`/`log`), `loop` restart |
| `story`     | story director: mission profiles (`kind` story/stranger, legs + tasks, `mount`, `seek_marker`), completion keywords/region, stage timeouts, auto-greet, opportunistic detours (Mode 1, off by default) |
| `combat`    | self-defense on/off, engage thresholds, aim/fire/retreat clamps |
| `survival`  | core-triggered remedies (key sequences, needs, cooldowns, hysteresis) |
| `rl`        | learned-policy driver (mode, checkpoint/buffer paths, reward weights) |
| `vision`    | `fast` pass, `hud` regions/thresholds, `world` estimators + minimap enemy blips, `dialogue` band/keywords/reactions, `ocr` engine/throttling |
| `debug`     | dashboard window, `show_overlay`/`show_hud` boxes, panel width |
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
2. **Phase 2 (done)** - HUD/OCR perception: core gauges, minimap, prompts,
   wanted stars, pluggable OCR, richer overlay and telemetry.
3. **Phase 3 (done)** - world state estimation: minimap markers/terrain
   hints, sky time-of-day and weather, ammo via OCR, horse cores.
4. **Phase 4 (done)** - navigation and locomotion control (mouse look,
   movement): locomotion controller, scripted autopilot, and route
   navigation with minimap stall detection.
5. **Phase 5 (done)** - mission/task framework and goal management: config
   tasks executed in order with blocking waits, state conditions and a
   fail-stop policy.
6. **Phase 6 (done)** - dialogue, encounters, and camp interactions:
   subtitle-band detection, encounter keyword matching against visible
   prompt/subtitle text, and configurable prompt reactions.
7. **Phase 7 (done)** - combat and self-defense behaviors: minimap enemy
   blips, damage-window threat assessment, and an engage/aim/fire/retreat
   planner that is off by default.
8. **Phase 8 (done)** - survival systems: core-triggered provisioning
   macros (cooldown, hysteresis, threat gating); cash/crafting stay
   unknown - they are not visible on the gameplay HUD.
9. **Phase 9 (done)** - learned/optimized policies (RL) on top of scripted
   skills: record -> train offline (numpy REINFORCE/BC) -> act from a
   checkpoint; a best-effort research scaffold, off by default.
10. **Phase 10 (done)** - robustness, long-run autonomy, packaging and
    docs: `ComponentGuard` fault isolation (a crashing component is
    disabled, inputs released, loop continues), `src.doctor` self-check,
    `rl.buffer_max` memory cap, packaging/version polish.
11. **Phase 11 (done)** - control verification: full PC controls
    reference (`docs/CONTROLS.md`) plus the `.\run.ps1 -ControlTest`
    guided harness that sends every agent input primitive live and
    records your verdict against the reference.
12. **Story mode (done)** - NOTES Mode 1 director behind
    `story.enabled` (or `.\run.ps1 -Story`): mount/travel/objective/
    completion stages over whistle + RoutePlanner + MissionRunner,
    objective-region OCR, completion keywords with banner OCR, stage
    timeouts, marker homing (`seek_marker`), stranger detours
    (`kind: stranger` + opportunistic), auto-greet honor strategy.
    Mission profiles still need authoring against the live game.

Each phase ships with tests and stays revertible: the safety layer and manual
hotkeys are never bypassed by later phases.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Anything unexpected at startup | Run `.\run.ps1 -Doctor` first - it reports interpreter, dependencies, Tesseract, config and window status as PASS/WARN/FAIL. |
| `capture backend selected: mss` + slow fps | dxcam unavailable; reinstall `dxcam`, run as the same user as the desktop session. |
| Frames are 0, log says `waiting_for_window` | RDR2 Story Mode is not running, or the window title is not exactly a `window.title_patterns` entry (`window.match: exact` by default). Check `.\calibrate.ps1 -Inspect`. |
| Colors look swapped in saved frames | Set `capture.color_order` explicitly (`bgr`/`rgb`); dxcam delivers RGB, mss delivers BGRA. |
| Cores always show `--` on the HUD row | Run `.\calibrate.ps1 -Hud`: either the HUD is hidden/the window is missing, or the boxes do not frame the cores - tighten `vision.hud.regions`. Regions dominated by bright scenery read *unknown* on purpose rather than a wrong value. |
| `WORLD` row shows `--` time / `horse N` | Normal indoors or with the HUD hidden; tune `vision.world.regions` (sky strip, horse cores, weapon box) with `.\calibrate.ps1 -Hud`. |
| Debug window looks wrong or mis-sized | Run `.\calibrate.ps1 -Gui`; tune `debug.panel_width` / `debug.render_fps`. `debug.show_hud: false` hides the detection boxes. |
| No prompt text (`ocr=off` in the HUD row) | Tesseract is not installed. `winget install Mannuel.Tesseract-OCR`, reopen the shell; the engine switches to `tesseract` automatically (`vision.ocr.engine: auto`). |
| Prompt line never appears | Raise `vision.hud.prompt_min_edge_density`/`prompt_min_bright_frac` down, check the `prompt` region with `.\calibrate.ps1 -Hud`. |
| Input has no effect in game | Click the game window first; input is refused unless the game is focused. Try `input.keyboard_mode: virtual_key` if scancodes are ignored. |
| `hotkeys not armed` / F12 ignored | The agent process must be running; check `logs/rdr2ai_*.log` for the `hotkeys armed` line. |
| Watchdog faults at startup | Lower `agent.loop_hz` or raise `safety.heartbeat_stall_s`; check `logs/events.jsonl` for the stalled component. |
| Debug window eats CPU | Run `.\run.ps1 -Headless` or set `debug.gui: false`. |
| Wrong monitor captured | Set `capture.monitor_index` and verify with `.\calibrate.ps1 -Inspect -Full`. |
| Pytest import errors | Run from the project root: `.\.venv\Scripts\python.exe -m pytest -q` (uses `pythonpath = ["."]`). |

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest -q      # 579 tests
.\.venv\Scripts\python.exe -m ruff check .   # lint
.\run.ps1 -Doctor                            # environment self-check
.\run.ps1 -ControlTest                       # guided input verification
```

Conventions: type hints everywhere, no hardcoded tunables (config only),
docstrings for public APIs, and the safety layer must remain reachable from
any state.

## Safety: the RDR2 installation is never touched

The agent reads pixels and sends input - nothing else. Concretely, it only:

1. reads screen frames from the game window (capture backends),
2. sends synthetic keyboard/mouse events through the Windows `SendInput`
   API (never via shell commands),
3. enumerates and focuses the game window (Win32 window messages),
4. writes logs, recordings and checkpoints **inside this project folder**
   (`logs/`, `recordings/`, `learning/` - every configured output path is
   resolved relative to `config.yaml`).

It never opens, reads, writes, moves or deletes any file in the RDR2
installation folder or the Rockstar settings folders
(`Documents\Rockstar Games\...`, `%LOCALAPPDATA%\Rockstar Games\...`).
There is no registry editing, no DLL injection, no process modification,
no memory reading and no subprocess/shell usage anywhere in the codebase.
The one library that *does* spawn a process - `pytesseract` invoking the
standalone `tesseract` OCR binary - runs only for prompt/objective text
reading and touches nothing but the screen frames it is handed.

Enforced, not just promised:

- **Config validation refuses game paths**: `telemetry.dir`,
  `recording.dir`, `rl.checkpoint` and `rl.buffer` raise a `ConfigError`
  if their resolved path contains `Rockstar Games` or
  `Red Dead Redemption 2` (the game folder and its settings stay
  untouched even if misconfigured by hand).
- **`tests/test_game_safety.py` scans the source** (`src/`, `tools/`) for
  registry access (`winreg`), process spawning (`subprocess`),
  shell execution (`os.system`/`os.popen`), file/tree deletion
  (`os.remove`/`shutil.rmtree`), game-folder path literals and absolute
  drive paths - so this guarantee cannot silently regress.
- Note: RDR2's own autosave may run while you play (that is the game
  process writing its own save data, exactly as when playing manually).
  This software never opens those files.

## Disclaimer

This is an unofficial, educational project, unaffiliated with Rockstar Games.
It targets single-player Story Mode only; using automation in Red Dead Online
may violate the game's terms of service. Use at your own risk, and keep F12
under your finger while the agent is running.
