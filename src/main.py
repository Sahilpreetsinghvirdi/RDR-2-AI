"""RDR2 AI - Phase 3 agent entry point.

Closed loop: capture -> fast vision + HUD/world perception (OCR when available)
-> guarded input -> observe. Emergency stop F12, pause F11, human takeover F10.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections.abc import Callable

from src import __version__
from src.ai.brain import BrainPlanner
from src.ai.rl.planner import PolicyPlanner, rl_summary
from src.ai.threat import ThreatAssessor
from src.capture.screen_capture import FramePacket, ScreenCapture
from src.capture.window_manager import (
    GameWindowManager,
    Rect,
    WindowInfo,
    enable_dpi_awareness,
    make_focus_check,
)
from src.config import AppConfig, ConfigError, load_config, save_effective_config
from src.control.locomotion import LocomotionController
from src.demo import Phase1Demo
from src.input.input_controller import InputController
from src.mission.respond import PromptResponder
from src.mission.runner import MissionRunner
from src.planning.combat import CombatPlanner
from src.planning.route import RoutePlanner
from src.planning.scripted import ScriptedPlanner
from src.planning.survival import SurvivalPlanner, survival_summary
from src.safety.emergency_stop import EmergencyStop
from src.safety.resilience import ComponentGuard
from src.safety.state_machine import AgentState, StateMachine
from src.safety.watchdog import HeartbeatRegistry, Watchdog
from src.state.agent_status import StatusTracker
from src.state.game_state import GameState
from src.story import StoryRunner
from src.telemetry.logger import EventLog, setup_logging
from src.telemetry.recorder import SessionRecorder
from src.ui.dashboard import Dashboard
from src.vision.fast_pass import FastPass
from src.vision.ocr import create_ocr
from src.vision.perception import (
    Perception,
    PerceptionResult,
    dialogue_summary,
    hud_summary,
    threat_summary,
    world_summary,
)

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rdr2-ai",
        description="Autonomous AI player foundation for RDR2 Story Mode (Phase 3).",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value (repeatable), e.g. --set capture.target_fps=60",
    )
    parser.add_argument("--demo", action="store_true", help="run the Phase 1 input demo")
    parser.add_argument(
        "--autopilot", action="store_true",
        help="enable the Phase 4 scripted locomotion autopilot",
    )
    parser.add_argument(
        "--mission", action="store_true",
        help="run the configured mission task list (Phase 5)",
    )
    parser.add_argument(
        "--story", action="store_true",
        help="run the story mode director over story.missions (Mode 1)",
    )
    parser.add_argument(
        "--brain", action="store_true",
        help="let the local vision model drive (needs ollama + model)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="ask what to do each session: pick a mission, free roam, observe (Mode 2)",
    )
    parser.add_argument("--headless", action="store_true", help="disable the debug window")
    parser.add_argument("--record", action="store_true", help="record frames to disk")
    parser.add_argument(
        "--backend", choices=["auto", "dxcam", "mss", "synthetic"], default=None,
        help="force a capture backend",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="stop automatically after N seconds (0 = run until hotkey)",
    )
    parser.add_argument("--version", action="version", version=f"rdr2-ai {__version__}")
    return parser.parse_args(argv)


def _apply_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> None:
    if args.headless:
        cfg.debug.gui = False
    if args.record:
        cfg.recording.enabled = True
    if args.backend:
        cfg.capture.backend = args.backend
    if args.autopilot:
        cfg.control.enabled = True
        cfg.validate()
    if args.mission:
        cfg.mission.enabled = True
        cfg.validate()
    if args.story:
        cfg.story.enabled = True
        cfg.validate()
    if args.brain:
        cfg.brain.enabled = True
        cfg.validate()
    if not args.interactive and (
        cfg.mission.enabled or cfg.control.enabled
        or cfg.story.enabled or cfg.rl.enabled or cfg.brain.enabled
    ):
        cfg.agent.mode = "autonomous"


def make_region_provider(wm: GameWindowManager) -> Callable[[], Rect | None]:
    def provider() -> Rect | None:
        info = wm.current()
        if info is None or info.minimized or not info.visible:
            return None
        return info.client

    return provider


def run_agent(cfg: AppConfig, args: argparse.Namespace) -> int:
    machine = StateMachine()
    heartbeats = HeartbeatRegistry()
    wm = GameWindowManager(cfg.window)
    focus_check = make_focus_check(
        wm, cfg.window.focus_before_input, cfg.window.focus_timeout_s
    )
    status = StatusTracker(cfg.agent.phase)

    def allow_input() -> bool:
        return machine.state is AgentState.RUNNING and focus_check()

    input_ctrl = InputController(
        cfg.input,
        allow_input=allow_input,
        stop_event=machine.stop_event,
    )
    locomotion = LocomotionController(cfg.control, input_ctrl)
    ocr_engine = create_ocr(cfg.vision.ocr)
    planner: ScriptedPlanner | RoutePlanner | MissionRunner | StoryRunner | None = None
    if cfg.story.enabled:
        planner = StoryRunner(
            cfg.story,
            cfg.control,
            locomotion,
            cfg.vision.hud.regions.get("minimap"),
            ocr=ocr_engine,
            ocr_min_confidence=cfg.vision.ocr.min_confidence,
        )
        log.info(
            "story mode - %d missions%s%s",
            len(cfg.story.missions),
            ", looping" if cfg.story.loop else "",
            ", auto-greet on" if cfg.story.auto_greet else "",
        )
    elif cfg.mission.enabled:
        planner = MissionRunner(
            cfg.mission, cfg.control, locomotion,
            cfg.vision.hud.regions.get("minimap"),
        )
        log.info("mission enabled - %d tasks%s", len(cfg.mission.tasks),
                 ", looping" if cfg.mission.loop else "")
    elif cfg.control.enabled:
        if cfg.control.mode == "route":
            planner = RoutePlanner(
                cfg.control, locomotion, cfg.vision.hud.regions.get("minimap")
            )
            log.info(
                "autopilot enabled - following %d route legs",
                len(cfg.control.nav.legs),
            )
        else:
            planner = ScriptedPlanner(cfg.control, locomotion)
            log.info("autopilot enabled - scripted locomotion active while RUNNING")

    if cfg.story.enabled and cfg.story.auto_greet:
        cfg.vision.dialogue.respond_to.setdefault("greet", "interact")

    responder: PromptResponder | None = None
    if cfg.vision.dialogue.respond_to:
        responder = PromptResponder(cfg.vision.dialogue, locomotion)
        log.info(
            "prompt responder armed: %s",
            ", ".join(f"{k}->{v}" for k, v in cfg.vision.dialogue.respond_to.items()),
        )

    assessor = ThreatAssessor(cfg.combat)
    combat: CombatPlanner | None = None
    if cfg.combat.enabled:
        combat = CombatPlanner(cfg.combat, locomotion, input_ctrl)
        log.info("combat enabled - self-defense armed (engage on wanted/enemies/fire)")
    survival: SurvivalPlanner | None = None
    if cfg.survival.enabled:
        survival = SurvivalPlanner(cfg.survival, input_ctrl, locomotion)
        log.info(
            "survival enabled - %d remedies armed",
            len(cfg.survival.remedies),
        )
    policy_planner: PolicyPlanner | None = None
    if cfg.rl.enabled:
        policy_planner = PolicyPlanner(
            cfg.rl,
            locomotion,
            checkpoint_path=cfg.resolve(cfg.rl.checkpoint),
            buffer_path=cfg.resolve(cfg.rl.buffer),
        )
        if policy_planner.active:
            log.info(
                "rl enabled - mode=%s explore=%.2f record=%s",
                cfg.rl.mode, cfg.rl.explore, cfg.rl.record,
            )
        else:
            log.warning("rl enabled but inactive: %s", policy_planner.reason)
    brain: BrainPlanner | None = None
    if cfg.brain.enabled:
        brain = BrainPlanner(cfg.brain, locomotion)
        log.info(
            "brain enabled - model=%s interval=%.1fs",
            cfg.brain.model, cfg.brain.interval_s,
        )

    events_path = cfg.resolve(cfg.telemetry.dir) / cfg.telemetry.events_file
    events = EventLog(events_path if cfg.telemetry.file else None)
    guard = ComponentGuard(
        on_fault=_make_fault_handler(events, status, input_ctrl)
    )

    emergency = EmergencyStop(
        cfg.safety,
        machine,
        input_ctrl.release_all,
        on_pause_change=lambda paused: _on_mode_change(events, status, "pause", paused),
        on_takeover_change=lambda taken: _on_mode_change(events, status, "takeover", taken),
    )
    emergency.start()

    watchdog = Watchdog(cfg.safety, machine, heartbeats, input_ctrl.release_all)

    capture = ScreenCapture(
        cfg.capture, make_region_provider(wm), on_beat=heartbeats.beat  # type: ignore[arg-type]
    )
    recorder: SessionRecorder | None = None
    if cfg.recording.enabled:
        recorder = SessionRecorder(cfg.recording, cfg.resolve(cfg.recording.dir))
    dashboard = Dashboard(
        cfg.debug,
        emergency=cfg.safety.emergency_key,
        pause=cfg.safety.pause_key,
        takeover=cfg.safety.takeover_key,
    )
    fast = FastPass(cfg.vision.fast)
    perception = Perception(cfg.vision, ocr=ocr_engine)
    game_state = GameState()
    if not cfg.vision.hud.enabled:
        log.info("hud perception disabled (vision.hud.enabled=false)")
    if perception.ocr_engine == "off":
        log.info("ocr disabled - prompt/objective text will not be read")
    demo: Phase1Demo | None = None
    if args.demo:
        demo = Phase1Demo(
            input_ctrl,
            ensure_focus=focus_check,
            on_result=lambda label, ok, ms: events.emit(
                "demo", step=label, ok=ok, elapsed_ms=round(ms, 2)
            ),
        )

    rc = 0
    render_thread: threading.Thread | None = None
    started = time.monotonic()
    try:
        if cfg.recording.enabled and recorder is not None:
            recorder.start(
                {
                    "backend": cfg.capture.backend,
                    "target_fps": cfg.capture.target_fps,
                    "loop_hz": cfg.agent.loop_hz,
                    "phase": cfg.agent.phase,
                    "demo": bool(args.demo),
                    "autopilot": bool(
                        cfg.control.enabled or cfg.mission.enabled
                        or cfg.story.enabled or cfg.rl.enabled
                    ),
                }
            )
        if cfg.debug.gui:
            render_thread = threading.Thread(
                target=_render_loop, name="render", daemon=True,
                args=(cfg, capture, dashboard, status, heartbeats, machine),
            )
            render_thread.start()
        capture.start()
        watchdog.register_thread("capture", capture.thread)
        watchdog.register_thread("render", render_thread)
        watchdog.start()
        machine.request(AgentState.RUNNING, "startup")
        events.emit("startup", name=cfg.agent.name, phase=cfg.agent.phase,
                      backend=cfg.capture.backend)
        log.info(
            "agent running: loop=%.1fHz capture=%d fps backend=%s gui=%s demo=%s",
            cfg.agent.loop_hz, cfg.capture.target_fps, cfg.capture.backend,
            cfg.debug.gui, bool(args.demo),
        )
        rc = _agent_loop(
            cfg, args, machine, heartbeats, wm, capture, fast, perception,
            game_state, status, input_ctrl, recorder, events, demo, started,
            locomotion, planner, responder, assessor, combat, survival,
            policy_planner, brain, guard,
        )
    except KeyboardInterrupt:
        log.warning("interrupted by user (Ctrl+C)")
        machine.request(AgentState.STOPPED, "keyboard-interrupt")
        rc = 130
    except Exception as exc:
        log.exception("agent crashed")
        status.update(state=machine.state.value, error=f"{type(exc).__name__}: {exc}")
        events.emit("crash", error=f"{type(exc).__name__}: {exc}")
        try:
            input_ctrl.release_all()
        except Exception:
            log.exception("release_all failed during crash handling")
        machine.request(AgentState.ERROR, f"crash: {type(exc).__name__}")
        rc = 1
    finally:
        machine.request(AgentState.STOPPED, "shutdown")
        input_ctrl.release_all()
        emergency.stop()
        capture.stop()
        watchdog.stop()
        if render_thread is not None:
            render_thread.join(2.0)
        if recorder is not None:
            recorder.close()
        if policy_planner is not None:
            saved = policy_planner.close()
            if saved is not None:
                events.emit("rl_buffer", path=str(saved))
        if brain is not None:
            brain.close()
        events.emit(
            "shutdown",
            state=machine.state.value,
            uptime_s=round(time.monotonic() - started, 2),
            capture=capture.stats.to_dict(),
            input=input_ctrl.snapshot(),
            hotkey_triggers=emergency.trigger_count,
        )
        events.close()
        log.info(
            "shutdown complete: state=%s frames=%d input_sent=%d refused=%d "
            "released=%d events=%d",
            machine.state.value, capture.stats.frames, input_ctrl.stats.sent,
            input_ctrl.stats.refused, input_ctrl.stats.released, events.count,
        )
    return rc


def _make_fault_handler(
    events: EventLog, status: StatusTracker, input_ctrl: InputController
) -> Callable[[str, BaseException], None]:
    def handler(name: str, exc: BaseException) -> None:
        input_ctrl.release_all()
        events.emit("component_fault", component=name,
                    error=f"{type(exc).__name__}: {exc}")
        status.update(
            error=f"{name}: {type(exc).__name__}",
            message=f"{name} failed - observing without it",
        )

    return handler


def _halt_drivers(
    locomotion: LocomotionController,
    combat: CombatPlanner | None,
    survival: SurvivalPlanner | None,
    policy_planner: PolicyPlanner | None,
    game_state: GameState,
) -> None:
    """Release movement/fire and reset mid-sequence driver state.

    Used whenever the loop must stop acting (pause, takeover, missing or
    stale frames, focus lost): locomotion releases its held keys, combat
    releases the fire button, an in-flight survival remedy is aborted and
    the RL planner drops its pending half-recorded transition.
    """
    locomotion.stop()
    if combat is not None:
        combat.stop()
    if survival is not None:
        survival.stop(game_state)
    if policy_planner is not None:
        policy_planner.stop()


def _on_mode_change(events: EventLog, status: StatusTracker, mode: str, active: bool) -> None:
    events.emit("mode", mode=mode, active=active)
    if mode == "pause":
        status.update(message="paused (F11 to resume)" if active else "resumed")
    else:
        status.update(
            message="human takeover active (F10 to return control)" if active
            else "control returned to AI"
        )


def _render_loop(
    cfg: AppConfig,
    capture: ScreenCapture,
    dashboard: Dashboard,
    status: StatusTracker,
    heartbeats: HeartbeatRegistry,
    machine: StateMachine,
) -> None:
    interval = 1.0 / max(1, cfg.debug.render_fps)
    while not machine.stop_event.is_set():
        heartbeats.beat("render")
        try:
            dashboard.show(capture.buffer.latest(), status.snapshot())
        except Exception:
            log.exception("render iteration failed")
        machine.stop_event.wait(interval)
    dashboard.close()


def _agent_loop(
    cfg: AppConfig,
    args: argparse.Namespace,
    machine: StateMachine,
    heartbeats: HeartbeatRegistry,
    wm: GameWindowManager,
    capture: ScreenCapture,
    fast: FastPass,
    perception: Perception,
    game_state: GameState,
    status: StatusTracker,
    input_ctrl: InputController,
    recorder: SessionRecorder | None,
    events: EventLog,
    demo: Phase1Demo | None,
    started: float,
    locomotion: LocomotionController,
    planner: ScriptedPlanner | RoutePlanner | MissionRunner | StoryRunner | None,
    responder: PromptResponder | None = None,
    assessor: ThreatAssessor | None = None,
    combat: CombatPlanner | None = None,
    survival: SurvivalPlanner | None = None,
    policy_planner: PolicyPlanner | None = None,
    brain: BrainPlanner | None = None,
    guard: ComponentGuard | None = None,
) -> int:
    if guard is None:
        guard = ComponentGuard()
    period = 1.0 / max(1.0, cfg.agent.loop_hz)
    hz = 0.0
    prev_start: float | None = None
    window_seen = False
    last_pres: PerceptionResult | None = None
    next_metrics = time.monotonic() + cfg.telemetry.metrics_interval_s
    deadline = started + args.duration if args.duration > 0 else None

    while not machine.stop_event.is_set():
        heartbeats.beat("main")
        loop_start = time.perf_counter()
        now = time.monotonic()

        if deadline is not None and now >= deadline:
            log.info("duration limit reached (%.0fs)", args.duration)
            machine.request(AgentState.STOPPED, "duration")
            break

        info = wm.current()
        packet = capture.buffer.latest()
        state = machine.state

        if info is None:
            if window_seen:
                log.warning("game window lost; waiting for it to return")
                window_seen = False
            status.update(
                window_title="", window_bounds=None, window_focused=False,
                message="waiting for RDR2 window (launch Story Mode)",
            )
        elif not window_seen:
            window_seen = True
            log.info(
                "game window: '%s' client=%dx%d @ %d,%d",
                info.title, info.client.width, info.client.height,
                info.client.left, info.client.top,
            )
            events.emit("window_found", **info.to_dict())
        else:
            status.update(
                window_title=info.title,
                window_bounds=list(info.client.as_tuple()),
                window_focused=info.focused,
            )

        if state is AgentState.RUNNING:
            last_pres = _run_active_work(
                cfg, capture, fast, perception, game_state, packet, info, status,
                input_ctrl, recorder, demo, events, locomotion, planner, responder,
                assessor, combat, survival, policy_planner, brain, guard,
            )
        else:
            _halt_drivers(locomotion, combat, survival, policy_planner, game_state)
            status.update(
                state=state.value,
                action="none",
                held_keys=input_ctrl.held_keys,
                held_buttons=input_ctrl.held_buttons,
                message={
                    AgentState.PAUSED: "paused - F11 resumes",
                    AgentState.TAKEOVER: "human takeover - F10 returns control",
                    AgentState.ERROR: "error state - restart required",
                    AgentState.STOPPED: "stopped",
                    AgentState.INIT: "initializing",
                }.get(state, state.value),
            )

        elapsed = time.perf_counter() - loop_start
        tick = time.perf_counter()
        if prev_start is not None and tick > prev_start:
            rate = 1.0 / (tick - prev_start)
            hz = rate if hz == 0.0 else 0.9 * hz + 0.1 * rate
        prev_start = tick
        status.update(
            state=state.value,
            loop_hz=round(hz, 2),
            capture_backend=capture.stats.backend,
            capture_fps=round(capture.stats.fps, 1),
            capture_latency_ms=round(capture.stats.latency_ms_avg, 2),
            input_latency_ms=round(input_ctrl.stats.last_latency_ms, 2),
            held_keys=input_ctrl.held_keys,
            held_buttons=input_ctrl.held_buttons,
            last_action=input_ctrl.last_action,
        )

        if now >= next_metrics:
            next_metrics = now + cfg.telemetry.metrics_interval_s
            events.emit(
                "metrics",
                state=state.value,
                loop_hz=round(hz, 1),
                capture=capture.stats.to_dict(),
                input=input_ctrl.snapshot(),
                window="found" if info is not None else "missing",
                hud=hud_summary(last_pres) if last_pres is not None else None,
                world=world_summary(last_pres) if last_pres is not None else None,
                dialogue=dialogue_summary(last_pres) if last_pres is not None else None,
                threat=threat_summary(game_state),
                survival=survival_summary(game_state),
                rl=rl_summary(policy_planner),
                guard=guard.summary(),
            )
            log.debug(
                "metrics: %.1f Hz, capture %.1f fps (%.1f ms), vision %.1f ms, input %s",
                hz, capture.stats.fps, capture.stats.latency_ms_avg,
                status.snapshot().vision_ms, input_ctrl.snapshot()["last_action"],
            )

        if elapsed < period:
            time.sleep(period - elapsed)
    return 0


def _run_active_work(
    cfg: AppConfig,
    capture: ScreenCapture,
    fast: FastPass,
    perception: Perception,
    game_state: GameState,
    packet: FramePacket | None,
    info: WindowInfo | None,
    status: StatusTracker,
    input_ctrl: InputController,
    recorder: SessionRecorder | None,
    demo: Phase1Demo | None,
    events: EventLog,
    locomotion: LocomotionController,
    planner: ScriptedPlanner | RoutePlanner | MissionRunner | StoryRunner | None,
    responder: PromptResponder | None = None,
    assessor: ThreatAssessor | None = None,
    combat: CombatPlanner | None = None,
    survival: SurvivalPlanner | None = None,
    policy_planner: PolicyPlanner | None = None,
    brain: BrainPlanner | None = None,
    guard: ComponentGuard | None = None,
) -> PerceptionResult | None:
    if guard is None:
        guard = ComponentGuard()
    if info is None:
        _halt_drivers(locomotion, combat, survival, policy_planner, game_state)
        status.update(action="none", message="waiting for RDR2 window")
        return None
    if packet is None:
        _halt_drivers(locomotion, combat, survival, policy_planner, game_state)
        status.update(
            action="none",
            frame_id=0,
            frame_age_ms=0.0,
            message=f"waiting for frames ({capture.stats.backend})",
        )
        return None
    if packet.age_ms > cfg.capture.max_frame_age_ms:
        _halt_drivers(locomotion, combat, survival, policy_planner, game_state)
        status.update(
            action="none", frame_age_ms=round(packet.age_ms, 1),
            message=f"stale frame ({packet.age_ms:.0f}ms > "
                    f"{cfg.capture.max_frame_age_ms}ms)",
        )
        return None
    if not info.focused:
        _halt_drivers(locomotion, combat, survival, policy_planner, game_state)
        status.update(
            action="none",
            message="waiting for the game window to be focused",
        )
        return None

    result = fast.process(packet.image)
    pres = perception.process(packet.image, packet.frame_id)
    perception.apply_to_state(game_state, pres)
    if assessor is not None:
        guard.call("threat", assessor.assess, time.monotonic(), game_state)
    boxes, labels = (
        perception.overlay_boxes(pres) if cfg.debug.show_hud else ([], [])
    )
    gauges = pres.hud.gauges
    prompt_visible = bool(pres.hud.prompt is not None and pres.hud.prompt.present)
    minimap = pres.hud.minimap
    world = pres.world
    sky = None if world.skipped else world.sky
    horse = None if world.skipped else world.horse
    status.update(
        frame_id=packet.frame_id,
        frame_age_ms=round(packet.age_ms, 1),
        frame_size=[packet.image.shape[1], packet.image.shape[0]],
        vision_ms=round(result.latency_ms, 2),
        brightness=round(result.brightness, 1),
        motion=round(result.motion, 2),
        hud_health=gauges["health"].value if "health" in gauges else None,
        hud_stamina=gauges["stamina"].value if "stamina" in gauges else None,
        hud_dead_eye=gauges["dead_eye"].value if "dead_eye" in gauges else None,
        hud_minimap=minimap.present if minimap is not None else None,
        hud_prompt_visible=prompt_visible,
        hud_prompt=pres.prompt_text or "",
        confidence=round(game_state.confidence.overall, 3),
        hud_boxes=boxes,
        hud_labels=labels,
        hud_ms=round(pres.latency_ms, 2),
        ocr_engine=pres.ocr_engine,
        world_time=sky.time_of_day if sky is not None else None,
        world_weather=sky.weather if sky is not None else None,
        world_ammo=pres.ammo,
        world_horse_detected=bool(horse is not None and horse.detected),
        world_ms=round(world.latency_ms, 2),
        dialogue_active=(
            None if pres.dialogue.skipped else pres.dialogue.present
        ),
        dialogue_text=pres.dialogue.text or "",
        dialogue_encounter=pres.encounter or "",
        dialogue_ms=round(pres.dialogue.latency_ms, 2),
        threat_level=game_state.threat.level,
        threat_enemies=game_state.threat.enemies_detected,
        threat_wanted=game_state.threat.wanted_level,
        threat_fire=game_state.threat.incoming_fire,
        survival_active=game_state.survival.active_remedy or "",
        survival_low=",".join(game_state.survival.low_cores),
        goal=(
            f"IDLE (phase {cfg.agent.phase})"
            if demo is None or not demo.active else "VERIFY_INPUT"
        ),
        message="observing" if demo is None or not demo.active
        else status.snapshot().message,
    )

    if pres.prompt_changed and pres.prompt_text:
        log.info("prompt: %s", pres.prompt_text)
        events.emit("prompt", text=pres.prompt_text)

    engaged = False
    survival_busy = False
    if demo is not None and demo.active:
        guard.call("demo", demo.step, status)
        if guard.is_disabled("demo"):
            demo.active = False
            status.update(
                goal=f"IDLE (phase {cfg.agent.phase})",
                action="none",
                demo_step="failed - demo disabled",
            )
    else:
        if combat is not None:
            guard.call("combat", combat.step, time.monotonic(), status,
                       packet.image, game_state)
            engaged = combat.engaged if not guard.is_disabled("combat") else False
        if survival is not None:
            survival_busy = bool(
                guard.call("survival", survival.step, time.monotonic(), status,
                           packet.image, game_state)
            )
        if not engaged and not survival_busy:
            if planner is not None:
                guard.call("planner", planner.step, time.monotonic(), status,
                           packet.image, game_state)
            elif policy_planner is not None:
                guard.call("policy", policy_planner.step, time.monotonic(),
                           status, packet.image, game_state)
            elif brain is not None:
                guard.call("brain", brain.step, time.monotonic(),
                           status, packet.image, game_state)
            elif demo is not None:
                status.update(
                    goal=f"IDLE (phase {cfg.agent.phase})", action="none"
                )
        if responder is not None:
            guard.call("responder", responder.step, time.monotonic(), status,
                       game_state)

    if recorder is not None:
        recorder.record(packet, status.snapshot().to_dict())
    return pres


def _disable_primary_drivers(cfg: AppConfig) -> None:
    cfg.mission.enabled = False
    cfg.control.enabled = False
    cfg.story.enabled = False
    cfg.rl.enabled = False
    cfg.brain.enabled = False


def _apply_interactive_choice(cfg: AppConfig) -> None:
    """NOTES Mode 2: console picker over missions, free roam, or observe."""
    missions = [p for p in cfg.story.missions if isinstance(p, dict)]
    if not sys.stdin.isatty():
        log.warning("interactive mode without a console - observing only")
        _disable_primary_drivers(cfg)
        cfg.agent.mode = "interactive"
        cfg.validate()
        return
    print("RDR2 AI - what should Arthur do this session?")
    for i, profile in enumerate(missions, start=1):
        name = str(profile.get("name") or f"mission {i}")
        kind = str(profile.get("kind", "story"))
        print(f"  {i}. {name} [{kind}]")
    print("  R. free roam (wander + greet + survival upkeep)")
    print("  O. observe only")
    try:
        choice = input(
            f"choice [1..{len(missions)}/R/O, default O]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "o"
    if choice == "r":
        _disable_primary_drivers(cfg)
        cfg.story.missions = [{"name": "free roam", "kind": "story", "roam": True}]
        cfg.story.enabled = True
        cfg.story.loop = True
    elif choice.isdigit() and 1 <= int(choice) <= len(missions):
        cfg.story.missions = [missions[int(choice) - 1]]
        _disable_primary_drivers(cfg)
        cfg.story.enabled = True
    else:
        _disable_primary_drivers(cfg)
    cfg.agent.mode = "interactive"
    cfg.validate()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides)
        _apply_cli_overrides(cfg, args)
        if args.interactive:
            _apply_interactive_choice(cfg)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    enable_dpi_awareness()
    log_path = setup_logging(cfg.telemetry)
    for warning in cfg.warnings:
        log.warning("config: %s", warning)
    if log_path is not None:
        try:
            save_effective_config(cfg, cfg.resolve(cfg.telemetry.dir) / "effective_config.yaml")
        except Exception:
            log.debug("could not save effective config", exc_info=True)
    log.info(
        "rdr2-ai %s | phase %d | loop %.1f Hz | capture %d fps (%s) | keys %s",
        __version__, cfg.agent.phase, cfg.agent.loop_hz, cfg.capture.target_fps,
        cfg.capture.backend, cfg.input.keyboard_mode,
    )
    log.info(
        "hotkeys: %s=EMERGENCY STOP  %s=pause/resume  %s=human takeover",
        cfg.safety.emergency_key, cfg.safety.pause_key, cfg.safety.takeover_key,
    )
    try:
        return run_agent(cfg, args)
    except ConfigError as exc:
        log.error("config error: %s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted before startup completed")
        return 130


if __name__ == "__main__":
    sys.exit(main())
