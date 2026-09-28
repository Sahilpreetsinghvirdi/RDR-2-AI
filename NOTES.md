# Notes: Two Modes

Status: **Mode 1 implemented** - `StoryRunner` behind
`story.enabled = false` (mount/travel/dismount/objective/roam/identify/
complete stages, objective-region OCR, completion keywords + banner OCR,
stage timeouts, marker homing, `nearest` chasing with title-card
identification, stranger/camp profiles, auto-greet) plus Mode 2
(`.\Outlaw.ps1 -Interactive`: pick a mission, free roam, or observe) and
`agent.mode`. Still needs live-game work: authored real-mission profiles
with calibrated legs and title keywords (starters are commented in
`config.yaml`). Experimental: the local vision brain (`src/ai/brain`,
`.\Outlaw.ps1 -Brain`) replaces the scripted directors with a seeing model -
slow on CPU, but visual. See the README "Story mode" section.

## Mode 1 - Full Autonomous Story Mode
The AI plays the story end-to-end on its own.

- **Between missions**: travels to the next mission start point by
  horse - whistle, wait for mount prompt, tap interact, ride the route
  legs, dismount at destination.
- **During missions**: completes each mission on its own using scripted
  objective behaviors + OCR of objective text / prompts, with
  completion keywords ("mission passed") advancing to the next step.
- **Honor**: raises low honor by greeting NPCs whenever a greet prompt
  appears. Constraint: the honor value itself is NOT readable (not on
  the gameplay HUD, no memory reading), so the strategy is to take
  every greeting opportunity rather than read a number.
- **Survival** (Phase 8) keeps cores fed automatically during the run.

## Mode 2 - Interactive / Ask-User Mode
The agent asks the user what to do each session:

- **Pick a mission**: agent presents the list of configured/known
  missions, user chooses one, agent runs that mission's script.
- **Free roam**: user picks free roam - wander, greet (honor),
  survival upkeep, wander-and-explore behaviors, no mission objective.

Question prompt would appear in the console/dashboard at startup or
between tasks; user answers (mission name / "free roam").

## Sketch for Later
- Config: `agent.mode: assist | autonomous | interactive`
  (`assist` = current behavior, untouched). *not built*
- `StoryRunner` episode loop: stages `mount -> travel -> objective
  -> complete -> next step`; rides on top of existing RoutePlanner,
  mission tasks, prompt responder, survival, combat.
  *implemented as `src/story/runner.py` (mount/travel/dismount/objective/
  roam/complete stages).*
- `story:` config section: named waypoints (legs), per-step scripts,
  mount/dismount keys, completion keywords, auto-greet toggle.
  *implemented (legs, tasks incl. a generic `key` tap for raw keys,
  keywords, banner region, timeouts, auto-greet, `kind: stranger|camp` +
  opportunistic detours, `seek_marker` homing, `roam` wandering;
  starter profiles commented in `config.yaml`).*
- Objective-region OCR into existing `GameState.mission.objective_text`
  (field exists, currently unused). *implemented (`vision.hud.regions.objective`).*
- `nav.mounted_speed_mps` for route dead-reckoning while on horseback.
  *implemented as `story.mounted_speed_mps` (mounted control copy).*
- Interactive mode: console picker over a mission-profile library; a
  `free_roam` profile with wander/greet/survive behaviors.
  *implemented (`.\Outlaw.ps1 -Interactive`, `agent.mode`).*
- Mission profiles: config-declared scripts per story mission (no
  memory reading; vision/OCR only) - real missions need authoring.

## Why Deferred
1. User wants to verify basic control first: after all phases, run the
   software and confirm it can control simple inputs (movement, camera,
   taps) reliably before layering story autonomy on top.
2. Mission scripts need real playtesting against the live game to be
   trusted; premature mode work would build on untested foundations.

## Sequencing (agreed)
1. Finish remaining roadmap phases (9-10) as currently scoped.
2. Hands-on test run: simple control verification - guided by
   `.\Outlaw.ps1 -ControlTest` (Phase 11 harness, steps cross-referenced to
   docs/CONTROLS.md), then input_check / demo / locomotion against the
   live game.
3. THEN revisit this file and implement the two modes, starting with
   the StoryRunner skeleton behind a disabled-by-default flag.
