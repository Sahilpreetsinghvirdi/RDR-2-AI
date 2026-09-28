# Red Dead Redemption 2 - PC Keyboard/Mouse Controls Reference

This is the complete PC controls cheat-sheet the project owner collected
from a third-party website, kept here so the agent's inputs can be checked
against the game's real bindings (see "How this agent uses these keys"
below). If something does not match the game, the game is right - verify
with Options > Controls in RDR2.

Scope notes from the original list:

- Organized alphabetically by command. First line = command name, second
  line = the PC keyboard key(s) and/or mouse button(s), third line (where
  present) = notes.
- Some commands with the same effect are listed more than once so that
  different search terms find the same command.
- Does NOT include commands for the in-game games blackjack, dominoes,
  five finger fillet, or poker; the portable camera / special photo mode;
  or online play.

## How this agent uses these keys

| Agent action | Config key (`control.keys` / `input`) | Default | Reference command |
|---|---|---|---|
| Walk forward | `control.keys.forward` | `w` | Forward movement = W |
| Walk backward | `control.keys.back` | `s` | Backward movement = S |
| Strafe left | `control.keys.strafe_left` | `a` | Left movement = A |
| Strafe right | `control.keys.strafe_right` | `d` | Right movement = D |
| Sprint | `control.keys.sprint` | `shift` | Sprint / Run / Increase speed = Shift |
| Jump | `control.keys.jump` | `space` | Jump = Space-bar |
| Interact / Use | `control.keys.interact` | `e` | Interact with object / Use = E |
| Horse whistle | `control.keys.whistle` | `h` | Whistle for horse = H |
| Camera turn | mouse move (no key) | dx px | Look = mouse (implicit in list) |
| Aim | mouse right button | RMB | Aim = Right mouse button |
| Fire | mouse left button | LMB | Attack / Fire weapon = Left mouse button |
| Quick-use / survival macro | raw keys in `survival.remedies[].keys` | `i` | Quick use item = I |
| Emergency stop | `safety.emergency_key` | F12 | (agent-only, not a game bind) |
| Pause / resume | `safety.pause_key` | F11 | (agent-only, not a game bind) |
| Human takeover | `safety.takeover_key` | F10 | (agent-only, not a game bind) |

The guided live check of exactly these inputs is `.\run.ps1 -ControlTest`
(Phase 11 harness).

## Commands A - C

- **Accept (in game menus)** — Enter (either on keyboard or number pad)
- **Additional down (in game menus)** — Number-pad 5
- **Additional left (in game menus)** — Number-pad 4
- **Additional right (in game menus)** — Number-pad 6
- **Additional up (in game menus)** — Number-pad 8
- **Aim** — Right mouse button
- **Aim into the air** — U
- **Attack** — Left mouse button
- **Auto-ride** — Tap X then hold V (the cinematic camera) until the small frame fills, then release V, then tap W (forward) to start the auto-ride
  - Note 1: Eliminates the need to keep your hands on the keyboard/mouse when traveling to a selected way point.
  - Note 2: Once underway the speed can be increased by pressing the shift key.
  - Note 3: Does not work during missions, cut scenes, or when riding with another person.
  - Note 4: Not the same as fast travel - auto-ride does not save any in-game time; you experience the entire trip at normal time speed.
  - Note 5: During auto-ride you can change point of view by pressing C. Do not press V - this knocks the player out of auto-ride.
  - Note 6: Auto-ride does not mean safe-ride; it is still possible to be attacked by NPCs while auto-riding.
- **Backward movement (in boat, on foot, on wagon, but not on horse)** — S
- **Backward movement on horse** — Double tap left control AND hold it on the second tap
  - Note 1: Can only be performed after reaching highest level in horse handling.
- **Bait wheel** — Tab then use Q or E to select from available baits
- **Binoculars** — F4 then select binoculars on the wheel
- **Binoculars, zoom in or zoom out** — Roll mouse wheel
- **Brake (in boat, on foot, on horse, on wagon)** — Ctrl
- **Break down an object in the satchel** — Highlight the object in the satchel, then Enter
  - Note 1: The method for breaking down some types of fish into "gritty fish meat" (mix with berries to make strong predator bait).
  - Note 2: Not all types of fish can be broken down.
- **Brush horse** — With horse command list open (right mouse button) press B
- **Buy item from shop** — E
- **Call animal** — X
  - Note 1: Only available when aiming at an animal with a weapon; makes the animal momentarily raise its head for a clearer shot.
- **Call horse** — H
- **Camp symbols** — The three symbols in the upper right corner of camp are first aid, ammunition, and food: red = low, white = OK, yellow = well stocked.
- **Cancel (in game menus)** — Backspace
- **Carry** — R
- **Change camera position** — V
  - Note 1: Useful when using auto-ride to provide different views during the ride.
- **Change camera shoulder view** — X
- **Change seat in vehicle (boat, train, wagon)** — Z
- **Change shop currency type** — Shift (hold)
  - Note 1: Red Dead Redemption 2 Online only; switches between Dollars ($) and Gold at shopkeepers.
- **Change shooting shoulder when using Dead Eye** — V
  - Note 1: Switches the view from right shoulder (default) to left shoulder.
- **Cinematic camera startup** — Hold V until the small frame fills, then release V
- **Close menu** — Esc
- **Compass radar** — Z
  - Note 1: May open GeForce Experience instead if that software is installed.
- **Context action** — Space-bar
- **Content action 2** — F
- **Crouch** — Ctrl
- **Cycle camera views** — V

## Commands D - G

- **Dead Eye** — Middle mouse button OR Caps Lock
  - Note 1: Only works while aiming a weapon; duration limited by the dead eye core value.
  - Note 2: There are 5 tiers in Dead Eye. Tiers 2-5 are locked to completed game chapters: tier 2 = "Pouring Forth Oil IV" (chapter 2), tier 3 = "Urban Pleasures" (chapter 4), tier 4 = "Fleeting Joy" (chapter 5), tier 5 = "Goodbye Dear Friend" (chapter 6).
  - Note 3: Increasing one's Dead Eye level extends Dead Eye duration.
- **Dead Eye tag enemies** — Q
  - Note 1: Not possible to mark multiple targets when using a sniper rifle.
  - Note 2: See "Mark targets using Dead Eye" for additional information.
- **Disable radar** — V
- **Down (in game menus)** — Down arrow
- **Drift on horse** — Hold space-bar and press A/D/W (depending on the direction you want to go)
- **Dual wield sidearms** — 2
- **Eagle eye** — Middle mouse button OR Caps Lock
  - Note 1: Used for spotting animals, plants, people, animal tracks, or pickable items.
  - Note 2: Can only be activated when the player is not aiming a weapon.
  - Note 3: Does not use any player resources.
- **Eat (at campfire, from bowl, or from a plate)** — E
- **Enter cover (hide behind a wall or boxes)** — Q
- **Enter wagon** — E
- **Equip melee weapon** — 5
- **Equip left sidearm** — 1
- **Equip right sidearm** — 3
- **Equip sidearm in both hands (dual wield)** — 2
- **Equip thrown weapon (rope, throwing knives, tomahawk)** — 7
- **Equip unarmed** — 4
- **Equip weapon being carried on back** — 6
- **Equip weapon being carried on shoulder** — 8
- **Exit boat or canoe** — E
- **Exit cover (leave protection provided by a wall or boxes)** — Q
- **Expand radar** — X
- **Extra option (in game menus)** — Space-bar
- **Extra option 2 (in game menus)** — F
- **Extra option 3 (in game menus)** — Ctrl
- **Extra option 4 (in game menus)** — Tab

## Commands F - L

- **Fast travel** — Initially only by stage coach and by train (both cost money): buy a ticket at a stage coach boarding sign or from any staffed train station clerk.
- **Fast travel using a campfire** — Space-bar
  - Note 1: Player must have progressed to the story point where Arthur can fast travel and the campfire must be in place (see "Setup campfire").
- **Fast travel using the map near Arthur's bed** — Available after improving the gang's encampment; required for campfire fast travel and independent of train/wagon travel.
- **Feed horse** — R
- **Feed horse a specific food** — Stand at the front of the horse, hold Tab, press R for the "horse" wheel, select the block in the nine o'clock position, press Q or E to select the specific food, release Tab to feed.
  - Note 1: Give your horse some variety in their diet.
- **Feed horse from horseback** — Hold F4 then R for the horse wheel, select the "9 o'clock" food block, use Q/E to view food types, select the desired food, release F4.
  - Note 1: Lets you remain mounted while feeding; nicely animated.
- **Feed menu - interact with** — F1
- **Find a dropped/lost personal weapon, gear, supplies, or animal pelts** — Look for the white marker on the map (pistol-shaped for weapons, flattened pelt for pelts) and return to that location.
  - Note 1: If dropped in water, slowly approach and be ready to press the retrieve key "R" as soon as it appears.
  - Note 2: The marker indicates the location of the last equipped weapon only.
  - Note 3: A lost "legendary" pelt can be found at any trapper location - he already has it in inventory ready to craft.
- **Fire weapon** — Left mouse button
- **Fishing rod** — Hold Tab then tap R, then select fishing rod from the wheel
- **Fishing techniques**
  - Attach lure = E
  - Bait hook = E
  - Cut line = E
  - Fishing reel speed up = R
  - Fishing reel speed down = F
  - Fishing reel line in = Space-bar
  - Fishing reel line out = Shift
  - Stop line loss of reeled in line = Left-Ctrl
  - Note 1: Holding Left-Ctrl prevents losing reeling progress and greatly reduces broken lines.
- **Flee** — With horse command list open (right mouse button) press F
  - Note 1: Causes the dismounted player's horse to quickly leave the area; it usually stays within call-horse range.
- **Focus camera** — V
- **Forward movement (on foot, on horse, on wagon)** — W
- **Give command to a horse** — While standing next to a horse, hold the right mouse button down and select from the options: brush = B, feed = R, flee = F, information = Q, lead = E, pat = G, remove saddle = H
- **Ground tie your horse** — Hold Q and press E
  - Note 1: Not routinely reliable; included in case you want to try it.
  - Note 2: Some sources say it only works while riding on grass.
- **Headlight (on wagon)** — O (letter O, not zero)
- **Holster or un-holster weapon** — Tab (tap only, do not hold)
  - Note 1: Double tapping when holstering a single sidearm imparts a little spin to the weapon as it is holstered.
- **Horse information** — With horse command list open (right mouse button) press Q
- **Backup horse** — Double tap left control AND hold it on the second tap
- **Horse jump** — Space-bar
- **Horse melee mode** — F
- **Horse melee attack - left** — Left mouse button
- **Horse melee attack - right** — Right mouse button
- **Horse sprint** — Space-bar
- **Horse slow/stop** — Ctrl
- **Horse whistle** — H
  - Note 1: Calls horse to you or at least near to you.

## Commands I - M

- **Ignite explosives** — E
  - Note 1: Explosives are a throwing weapon, accessed by pressing 7.
  - Note 2: To place explosives without igniting them use the right mouse button.
- **Increase speed (in boat, on foot, on horse, on wagon)** — Shift
- **Increase horse speed** — Tap repeatedly or hold shift
  - Note: Holding shift lets you go faster more quickly.
- **Interact lock-on** — Right mouse button
- **Inspect item** — Middle mouse button OR F
- **Interact with animal** — G
- **Interact with object** — E
- **Jump** — Space-bar
- **Jump obstacle riding your horse** — Space-bar
- **Lead horse** — With horse command list open (right mouse button) press E
- **Left (in game menus)** — Left arrow
- **Left movement (in boat, on foot, on horse, on wagon)** — A
- **Look behind (in 3rd person only)** — C
- **Lock-on options 1-3**
  - Note 1: The 3 commands below (G, H, Space-bar) appear to give flexibility in how the Dead Eye viewing function works (narrow, normal, wide view) - console commands with no known PC use.
- **Lock-on option 1** — G
- **Lock-on option 2** — H
- **Lock-on option 3** — Space-bar
- **Lull target animal using "call"** — X
  - Note 1: Only available when aiming at an animal with a weapon; makes it momentarily raise its head for a shot.
- **Map zoom** — Middle mouse wheel rolled
- **Mark location on map** — With map open, place cursor on location and press Enter to mark
- **Mark target (manually)** — Target marking works differently at one point in the game: it is automatic until the player completes the entire mission "Pouring Forth Oil" in chapter 2, after which auto-tagging multiple targets is lost. Manual steps:
  1. With weapon in hand, enter Dead Eye (caps lock or middle mouse button) - screen gets brighter and everything slows down.
  2. Place the gun's sight cursor on a target; the cursor turns from white to red.
  3. With the red cursor on the target press X to mark it. Pressing X multiple times on the same location causes several shots to be fired at that location.
  - Note 1: Player can only mark as many targets as there are rounds in the current gun.
  4. After marking all targets, press the left mouse button - one shot fires at each marked target in order.
  - Note 2: During Dead Eye, if no targets are marked but the left mouse button is pressed, a single round fires.
- **Melee** — Use only the keyboard when fist fighting: hold the block key (R) when your opponent swings, tap the punch key (F) several times to strike, tap the grab key (E) to grab (and if health is low enough, tap F for extra hits). Stay off the mouse.
- **Melee - attack** — F
- **Melee - block** — R
- **Melee - grapple** — E
- **Melee - interact** — E
- **Melee - punch** — F
- **Melee - restrain** — E
- **Melee - rob** — E
- **Melee - strike** — F
- **Melee - take down** — F
- **Menu controls** — H
- **Mount horse** — E
- **Mount wagon** — E
- **Move backward** — S
- **Move forward** — W
- **Move horse in a circle** — Hold left control and press W repeatedly to circle right, OR hold left control and press D repeatedly to circle left
- **Move left** — A
- **Move right** — D
- **Move to the left or right in menus** — Q to move left, E to move right
  - Note 1: Used when looking at the camp ledger to purchase camp improvements.
- **Move up or down in menus** — Up or down arrow
  - Note 1: Used when looking at the camp ledger to purchase camp improvements.

## Commands N - R

- **Next document page (in game menus)** — Right arrow
- **Next item** — X
- **Next prompt page** — Q
- **Next secondary tab (in game menus)** — X
- **Next tab (in game menus)** — E OR page down
  - Note 1: Frequently used to select items/objects/choices after opening a selection wheel, e.g. the satchel wheel.
- **Next weapon** — Mouse wheel rolled
- **Open item wheel** — F4
- **Open journal** — J
- **Open log** — L
- **Open map** — M
- **Open player menu** — L
- **Open satchel** — B
- **Open weapon wheel** — Tab
- **Pat horse** — With horse command list open (right mouse button) press G
- **Pat horse while riding** — G
- **Pause game** — Esc
- **Pause menu** — P
  - Note 1: Returns the player to a screen with choices: view map, get help, view progress, read player story, go online, participate in social club, or quit.
- **Pay bounty in shop or train station** — B
- **Photo mode activation** — F6
- **Piaffe horse** — Hold space-bar
  - Note 1: Piaffe is a horse training term for a cadenced trot in place or nearly in place.
- **Pickup item** — R
- **Place marker on map** — With map open, place cursor on desired location and press Enter to mark
  - Note 1: Multiple markers can be placed at the same time.
- **Previous document page (in game menus)** — Left arrow
- **Previous item** — Z
- **Previous secondary tab (in game menus)** — Z
- **Previous tab (in game menus)** — Q or page up
  - Note 1: Frequently used to select items/objects/choices after opening a selection wheel.
- **Previous weapon** — Mouse wheel rolled
- **Previous wheel menu** — Q
- **Put on mask** — Hold F4, select bandanna mask from the wheel, then release F4
- **Quick use item** — I
- **Radar (the circular mini-map, default lower left corner)** — press Alt once to access these four options:
  - compass radar = Z (may open GeForce Experience instead)
  - disable radar = V
  - expanded radar = X
  - regular radar = C
- **Rear horse** — Left control + space-bar
  - Note 1: Causes your horse to rear back on its back legs.
- **Regular radar** — C
- **Reload** — R
- **Remove a marker from map** — With map open press Enter to remove marker
- **Remove saddle from horse** — With horse command list open (right mouse button) press H
- **Remove mask** — Hold F4, select bandanna mask from the wheel, then release F4
- **Rest** — E
  - Note 1: Player kneels and is offered options to setup camp (E), craft (R), and leave (F).
- **Right movement (in boat, on foot, on horse, on wagon)** — D
- **Right (in game menus)** — Right arrow
- **Run** — Shift

## Commands S - V

- **Select side on which to dismount from horse** — Look in the direction you wish to dismount and press E
  - Note 1: Works well in first person mode; no known way to select a specific dismount direction in third person.
- **Sell item to shop** — R
- **Setup campfire** — Hold F4 to open item wheel, select "crafting camp", release F4; the campfire is placed at the player's location.
- **Setup scout campfire** — Crafted at the gang's camp as part of the camp upgrade sequence (look for a separate fire near where the horses are kept).
- **Shoot into the air** — U then right mouse button
- **Show radar information** — Alt
- **Sliding stop while on a horse** — While galloping press left control
- **Special shop function** — F
- **Sprint** — Shift
- **Stand up from a crouch position** — Shift
- **Stand up in a boat or canoe** — E
- **Stealth** — Ctrl
- **Stop horse immediately after jumping** — Tap left control while in the jump
  - Note 1: Useful for reducing injuries from slamming into a wall or other object after jumping with horse.
- **Swap weapon in hand with one laying on ground** — Tab (tap it - do not hold it)
  - Note 1: Lets the player exchange a rifle or pistol in hand with one on the ground.
  - Note 2: Useful for picking up unique or unusual weapons dropped by characters.
  - Note 3: Also useful when the current weapon is degraded, out of gun oil, and a better one is on the ground.
- **Switch to first or third person view** — Tap V repeatedly to rotate between the available views
- **Switch to left shoulder view when using Dead Eye** — X
- **Switch weapon firing mode** — B
  - Note 1: Only available when using the LeMat revolver; toggles firing mode between revolver and shotgun.
- **Switch zoom magnification or levels on binoculars or weapons** — Middle mouse wheel rolled
- **Tear down your personal camp** — F
  - Note 1: Removes your tent and campfire.
- **Toggle weapon sight** — Middle mouse button
- **Train whistle** — G
  - Note 1: Causes train to sound its whistle; can only be done from the train locomotive's cab.

## Commands U - Z

- **Un-arm yourself (no weapon in either hand)** — 4
- **Up (in game menus)** — Up arrow
- **Use** — E
- **Use binoculars** — F4 then select binoculars on the wheel
- **Wait until...** — E
  - Note 1: Offers the ability to fast-forward time until either morning or night, depending on the situation.
- **Walk - run** — Ctrl
- **Weapon zoom in** — ]
  - Note 1: Works in first person mode only; does not work with scoped weapons.
- **Weapon zoom out** — [
  - Note 1: Works in first person mode only; does not work with scoped weapons.
- **Whistle for horse** — H
- **Whistle on train** — G
  - Note 1: Causes train to sound its whistle; can only be done from the train locomotive's cab.
- **Zoom (in game menus)** — Right mouse button
- **Zoom binoculars or rifle scopes** — Middle mouse wheel rolled
- **Zoom map** — Middle mouse wheel rolled
