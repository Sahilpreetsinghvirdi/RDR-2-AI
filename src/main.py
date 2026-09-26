"""RDR2 AI - Phase 1 agent entry point.

Closed loop: capture -> fast vision -> status -> (Phase 1: verification only)
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
from src.capture.screen_capture import FramePacket, ScreenCapture
from src.capture.window_manager import (
    GameWindowManager,
    Rect,
    WindowInfo,
    enable_dpi_awareness,
    make_focus_check,
)
from src.config import AppConfig, ConfigError, load_config, save_effective_config
from src.demo import Phase1Demo
from src.input.input_controller import InputController
from src.safety.emergency_stop import EmergencyStop
from src.safety.state_machine import AgentState, StateMachine
from src.safety.watchdog import HeartbeatRegistry, Watchdog
from src.state.agent_status import StatusTracker
from src.telemetry.logger import EventLog, setup_logging
from src.telemetry.recorder import SessionRecorder
from src.ui.dashboard import Dashboard
from src.vision.fast_pass import FastPass

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rdr2-ai",
        description="Autonomous AI player foundation for RDR2 Story Mode (Phase 1).",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value (repeatable), e.g. --set capture.target_fps=60",
    )
    parser.add_argument("--demo", action="store_true", help="run the Phase 1 input demo")
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
        cfg.validate()


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

    events_path = cfg.resolve(cfg.telemetry.dir) / cfg.telemetry.events_file
    events = EventLog(events_path if cfg.telemetry.file else None)

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
    dashboard = Dashboard(cfg.debug)
    fast = FastPass(cfg.vision.fast)
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
        events.emit("startup", phase=cfg.agent.phase, backend=cfg.capture.backend)
        log.info(
            "agent running: loop=%.1fHz capture=%d fps backend=%s gui=%s demo=%s",
            cfg.agent.loop_hz, cfg.capture.target_fps, cfg.capture.backend,
            cfg.debug.gui, bool(args.demo),
        )
        rc = _agent_loop(
            cfg, args, machine, heartbeats, wm, capture, fast, status,
            input_ctrl, recorder, events, demo, started,
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
    status: StatusTracker,
    input_ctrl: InputController,
    recorder: SessionRecorder | None,
    events: EventLog,
    demo: Phase1Demo | None,
    started: float,
) -> int:
    period = 1.0 / max(1.0, cfg.agent.loop_hz)
    hz = 0.0
    window_seen = False
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
            _run_active_work(
                cfg, capture, fast, packet, info, status, input_ctrl,
                recorder, demo, events,
            )
        else:
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
        rate = 1.0 / max(elapsed, 1e-6)
        hz = rate if hz == 0.0 else 0.9 * hz + 0.1 * rate
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
    packet: FramePacket | None,
    info: WindowInfo | None,
    status: StatusTracker,
    input_ctrl: InputController,
    recorder: SessionRecorder | None,
    demo: Phase1Demo | None,
    events: EventLog,
) -> None:
    if info is None:
        status.update(action="none", message="waiting for RDR2 window")
        return
    if packet is None:
        status.update(
            action="none",
            frame_id=0,
            frame_age_ms=0.0,
            message=f"waiting for frames ({capture.stats.backend})",
        )
        return
    if packet.age_ms > cfg.capture.max_frame_age_ms:
        status.update(
            action="none", frame_age_ms=round(packet.age_ms, 1),
            message=f"stale frame ({packet.age_ms:.0f}ms > "
                    f"{cfg.capture.max_frame_age_ms}ms)",
        )
        return

    result = fast.process(packet.image)
    status.update(
        frame_id=packet.frame_id,
        frame_age_ms=round(packet.age_ms, 1),
        frame_size=[packet.image.shape[1], packet.image.shape[0]],
        vision_ms=round(result.latency_ms, 2),
        brightness=round(result.brightness, 1),
        motion=round(result.motion, 2),
        goal="IDLE (phase 1)" if demo is None or not demo.active else "VERIFY_INPUT",
        message="observing" if demo is None or not demo.active else status.snapshot().message,
    )

    if demo is not None and demo.active:
        demo.step(status)
    elif demo is not None:
        status.update(goal="IDLE (phase 1)", action="none")

    if recorder is not None:
        recorder.record(packet, status.to_dict())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides)
        _apply_cli_overrides(cfg, args)
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
