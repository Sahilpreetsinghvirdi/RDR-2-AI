# Notes: Two Planned Modes (DESIGN ONLY - NOT IMPLEMENTED)

Status: **notes only** (per user decision). Do not implement until all
phases (1-11) are complete AND basic control test runs (simple
keyboard/mouse input verification) have been done first.

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

## Sketch for Later (do not build now)
- Config: `agent.mode: assist | autonomous | interactive`
  (`assist` = current behavior, untouched).
- New `StoryRunner` episode loop: stages `mount -> travel -> objective
  -> complete -> next step`; rides on top of existing RoutePlanner,
  mission tasks, prompt responder, survival, combat.
- `story:` config section: named waypoints (legs), per-step scripts,
  mount/dismount keys, completion keywords, auto-greet toggle.
- Objective-region OCR into existing `GameState.mission.objective_text`
  (field exists, currently unused).
- `nav.mounted_speed_mps` for route dead-reckoning while on horseback.
- Interactive mode: console picker over a mission-profile library; a
  `free_roam` profile with wander/greet/survive behaviors.
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
   `.\run.ps1 -ControlTest` (Phase 11 harness, steps cross-referenced to
   docs/CONTROLS.md), then input_check / demo / locomotion against the
   live game.
3. THEN revisit this file and implement the two modes, starting with
   the StoryRunner skeleton behind a disabled-by-default flag.
