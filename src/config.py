"""Typed configuration loading for RDR2 AI.

The YAML file is merged over built-in defaults, dotted ``--set a.b.c=value``
overrides are applied, unknown keys produce warnings and invalid values raise
:class:`ConfigError` before the agent starts.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME = "config.yaml"


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or is invalid."""


@dataclass
class AgentConfig:
    name: str = "rdr2-ai"
    phase: int = 1
    loop_hz: float = 10.0


@dataclass
class WindowConfig:
    title_patterns: list[str] = field(
        default_factory=lambda: ["Red Dead Redemption 2", "RDR2"]
    )
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["RDR2 AI - Debug"]
    )
    match: str = "exact"    # exact = whole title must equal a pattern (safe);
                            # substring = pattern anywhere in the title
    focus_before_input: bool = True
    poll_interval_s: float = 0.5
    focus_timeout_s: float = 1.5


@dataclass
class SyntheticConfig:
    width: int = 1280
    height: int = 720
    seed: int = 7


@dataclass
class CaptureConfig:
    backend: str = "auto"
    target_fps: int = 30
    color_order: str = "auto"
    roi_mode: str = "window"
    monitor_index: int = 0
    max_frame_age_ms: int = 500
    ring_size: int = 4
    dxcam_max_buffer: int = 64
    backend_failures_before_fallback: int = 30
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


@dataclass
class ControllerConfig:
    enabled: bool = False


@dataclass
class InputConfig:
    keyboard_mode: str = "scancode"
    verify_injection: bool = True
    verify_timeout_ms: int = 50
    default_hold_ms: int = 60
    hold_check_interval_s: float = 0.025
    mouse_max_delta: int = 3000
    click_hold_ms: int = 30
    controller: ControllerConfig = field(default_factory=ControllerConfig)


@dataclass
class SafetyConfig:
    emergency_key: str = "F12"
    pause_key: str = "F11"
    takeover_key: str = "F10"
    hotkey_poll_hz: int = 250
    watchdog_interval_s: float = 0.1
    heartbeat_stall_s: float = 2.0
    startup_grace_s: float = 10.0
    release_on_pause: bool = True


DEFAULT_CONTROL_KEYS: dict[str, str] = {
    "forward": "w",
    "back": "s",
    "strafe_left": "a",
    "strafe_right": "d",
    "sprint": "shift",
    "jump": "space",
    "interact": "e",
}


@dataclass
class ControlConfig:
    """Locomotion control (Phase 4): movement keys, look sensitivity, clamps."""

    enabled: bool = False          # scripted autopilot; off = agent only observes
    keys: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONTROL_KEYS))
    mouse_px_per_degree: float = 12.0   # camera look sensitivity (game mouse dpi varies)
    max_turn_deg_per_tick: float = 30.0  # camera step clamp per control tick
    min_step_s: float = 0.1
    max_step_s: float = 4.0             # clamp for a single move command


@dataclass
class FastPassConfig:
    enabled: bool = True
    width: int = 160
    height: int = 90
    every_n_frames: int = 1


DEFAULT_HUD_REGIONS: dict[str, list[float]] = {
    # x, y, w, h as fractions of the frame (reference layout 1920x1080).
    # Tune against the real game with tools/calibrate_hud.py.
    "health": [0.185, 0.905, 0.055, 0.085],
    "stamina": [0.245, 0.905, 0.055, 0.085],
    "dead_eye": [0.305, 0.905, 0.055, 0.085],
    "minimap": [0.004, 0.810, 0.100, 0.185],
    "prompt": [0.350, 0.720, 0.300, 0.120],
    "wanted": [0.870, 0.030, 0.110, 0.080],
}


@dataclass
class HudConfig:
    enabled: bool = True
    regions: dict[str, list[float]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_HUD_REGIONS)
    )
    gauge_contrast_min: float = 22.0
    gauge_min_confidence: float = 0.35
    minimap_hist_threshold: float = 0.35
    prompt_min_edge_density: float = 0.035
    prompt_min_bright_frac: float = 0.01
    wanted_min_blob_px: int = 12
    wanted_max_blob_px: int = 900


@dataclass
class OcrConfig:
    enabled: bool = True
    engine: str = "auto"          # auto | tesseract | off
    every_n_frames: int = 5
    languages: str = "eng"
    min_confidence: float = 40.0  # 0..100, engine confidence scale


DEFAULT_WORLD_REGIONS: dict[str, list[float]] = {
    # x, y, w, h as fractions of the frame (reference layout 1920x1080).
    "horse_health": [0.185, 0.830, 0.055, 0.070],
    "horse_stamina": [0.245, 0.830, 0.055, 0.070],
    "weapon": [0.860, 0.100, 0.120, 0.080],
    "sky": [0.0, 0.0, 1.0, 0.120],
}


@dataclass
class WorldConfig:
    enabled: bool = True
    regions: dict[str, list[float]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_WORLD_REGIONS)
    )
    marker_bright: int = 200
    marker_min_blob_px: int = 8
    marker_max_blob_px: int = 600
    road_luma: float = 150.0
    water_min_frac: float = 0.04
    road_min_frac: float = 0.03
    green_min_frac: float = 0.45
    night_luma: float = 45.0
    dusk_warmth: float = 15.0
    sky_max_std: float = 35.0
    ammo_min_bright_frac: float = 0.004
    ammo_min_luma: float = 170.0


@dataclass
class VisionConfig:
    fast: FastPassConfig = field(default_factory=FastPassConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    world: WorldConfig = field(default_factory=WorldConfig)


@dataclass
class DebugConfig:
    gui: bool = True
    window_name: str = "RDR2 AI - Debug"
    render_fps: int = 20
    show_overlay: bool = True
    show_hud: bool = True
    panel_width: int = 360


@dataclass
class RecordingConfig:
    enabled: bool = False
    fps: int = 5
    dir: str = "recordings"
    image_format: str = "jpg"
    jpeg_quality: int = 85
    max_frames: int = 2000
    save_state_jsonl: bool = True


@dataclass
class TelemetryConfig:
    console: bool = True
    file: bool = True
    dir: str = "logs"
    events_file: str = "events.jsonl"
    metrics_interval_s: float = 1.0
    log_level: str = "INFO"
    log_max_bytes: int = 10_000_000
    log_backups: int = 3


@dataclass
class AppConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    input: InputConfig = field(default_factory=InputConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    source_path: Path | None = field(default=None, init=False, repr=False, compare=False)
    warnings: list[str] = field(default_factory=list, init=False, repr=False, compare=False)

    @property
    def project_root(self) -> Path:
        if self.source_path is not None:
            return self.source_path.parent
        return Path.cwd()

    def resolve(self, path: str | Path) -> Path:
        """Resolve a possibly-relative configured path against the config file location."""
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        data.pop("warnings", None)
        return data

    def validate(self) -> None:
        problems: list[str] = []

        def _positive(name: str, value: Any, numeric: tuple[type, ...] = (int, float)) -> None:
            if not isinstance(value, numeric) or isinstance(value, bool):
                problems.append(f"{name} must be a number, got {value!r}")
            elif value <= 0:
                problems.append(f"{name} must be > 0, got {value!r}")

        _positive("agent.loop_hz", self.agent.loop_hz)
        _positive("capture.target_fps", self.capture.target_fps, (int,))
        _positive("debug.render_fps", self.debug.render_fps, (int,))
        _positive("recording.fps", self.recording.fps, (int,))
        _positive("safety.hotkey_poll_hz", self.safety.hotkey_poll_hz, (int,))
        _positive("safety.heartbeat_stall_s", self.safety.heartbeat_stall_s)
        _positive("telemetry.metrics_interval_s", self.telemetry.metrics_interval_s)
        _positive("capture.max_frame_age_ms", self.capture.max_frame_age_ms, (int,))

        if self.capture.backend not in {"auto", "dxcam", "mss", "synthetic"}:
            problems.append(
                f"capture.backend must be auto|dxcam|mss|synthetic, got {self.capture.backend!r}"
            )
        if self.capture.color_order not in {"auto", "bgr", "rgb", "bgra"}:
            problems.append(
                f"capture.color_order must be auto|bgr|rgb|bgra, got {self.capture.color_order!r}"
            )
        if self.capture.roi_mode not in {"window", "full"}:
            problems.append(f"capture.roi_mode must be window|full, got {self.capture.roi_mode!r}")
        if self.input.keyboard_mode not in {"scancode", "virtual_key"}:
            problems.append(
                "input.keyboard_mode must be scancode|virtual_key, "
                f"got {self.input.keyboard_mode!r}"
            )
        if self.recording.image_format not in {"jpg", "png"}:
            problems.append(
                f"recording.image_format must be jpg|png, got {self.recording.image_format!r}"
            )
        if not isinstance(self.window.title_patterns, list) or not self.window.title_patterns:
            problems.append("window.title_patterns must be a non-empty list")
        if self.window.match not in {"exact", "substring"}:
            problems.append(
                f"window.match must be exact|substring, got {self.window.match!r}"
            )
        if self.vision.ocr.engine not in {"auto", "tesseract", "off"}:
            problems.append(
                f"vision.ocr.engine must be auto|tesseract|off, got {self.vision.ocr.engine!r}"
            )
        _positive("vision.ocr.every_n_frames", self.vision.ocr.every_n_frames, (int,))
        if not 0 <= self.vision.ocr.min_confidence <= 100:
            problems.append("vision.ocr.min_confidence must be within 0..100")
        _check_regions(self.vision.hud.regions, "vision.hud.regions", problems)
        if not 0.0 <= self.vision.hud.gauge_min_confidence <= 1.0:
            problems.append("vision.hud.gauge_min_confidence must be within 0..1")
        _check_regions(self.vision.world.regions, "vision.world.regions", problems)
        for name in ("water_min_frac", "road_min_frac", "green_min_frac",
                     "ammo_min_bright_frac"):
            value = getattr(self.vision.world, name)
            if not 0.0 <= value <= 1.0:
                problems.append(f"vision.world.{name} must be within 0..1")
        if not isinstance(self.agent.phase, int) or self.agent.phase < 1:
            problems.append(f"agent.phase must be an int >= 1, got {self.agent.phase!r}")
        ctrl = self.control
        missing = set(DEFAULT_CONTROL_KEYS) - set(ctrl.keys)
        extra = set(ctrl.keys) - set(DEFAULT_CONTROL_KEYS)
        if missing:
            problems.append(f"control.keys missing {sorted(missing)}")
        if extra:
            problems.append(f"control.keys unknown {sorted(extra)}")
        if any(not isinstance(v, str) or not v.strip() for v in ctrl.keys.values()):
            problems.append("control.keys values must be non-empty strings")
        if ctrl.mouse_px_per_degree <= 0:
            problems.append("control.mouse_px_per_degree must be > 0")
        if ctrl.max_turn_deg_per_tick <= 0:
            problems.append("control.max_turn_deg_per_tick must be > 0")
        if ctrl.min_step_s <= 0 or ctrl.max_step_s < ctrl.min_step_s:
            problems.append("control step clamps require 0 < min_step_s <= max_step_s")
        if not isinstance(self.telemetry.log_level, str):
            problems.append("telemetry.log_level must be a string")

        if problems:
            raise ConfigError("invalid configuration: " + "; ".join(problems))


def _plain(data: Any) -> Any:
    if is_dataclass(data):
        return {f.name: _plain(getattr(data, f.name)) for f in fields(data) if f.init}
    if isinstance(data, dict):
        return {k: _plain(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_plain(v) for v in data]
    return data


def _merge(defaults: dict[str, Any], override: dict[str, Any], path: str,
           warnings: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(defaults)
    for key, value in override.items():
        dotted = f"{path}.{key}" if path else key
        if key not in out:
            warnings.append(f"unknown config key ignored: {dotted}")
            continue
        if isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value, dotted, warnings)
        else:
            out[key] = value
    return out


def _build(cls: type, data: dict[str, Any], warnings: list[str]) -> Any:
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls) if f.init}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        sub_cls: type | None = None
        if f.default is not MISSING and is_dataclass(f.default):
            sub_cls = type(f.default)
        elif f.default_factory is not MISSING and f.default_factory is not None:
            sample = f.default_factory()  # type: ignore[call-arg]
            if is_dataclass(sample):
                sub_cls = type(sample)
        if sub_cls is not None and isinstance(value, dict):
            kwargs[name] = _build(sub_cls, value, warnings)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _check_regions(regions: Any, where: str, problems: list[str]) -> None:
    if not isinstance(regions, dict):
        problems.append(f"{where} must be a mapping of name -> [x,y,w,h]")
        return
    for name, frac in regions.items():
        if (
            not isinstance(frac, (list, tuple)) or len(frac) != 4
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       for v in frac)
        ):
            problems.append(f"{where}.{name} must be [x, y, w, h] numbers")
            continue
        x, y, w, h = (float(v) for v in frac)
        if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0 and 0.0 < w <= 1.0
                and 0.0 < h <= 1.0 and x + w <= 1.0001 and y + h <= 1.0001):
            problems.append(f"{where}.{name} out of range (fractions of the frame)")


def _apply_override(data: dict[str, Any], expr: str, warnings: list[str]) -> None:
    if "=" not in expr:
        raise ConfigError(f"--set expects key=value, got {expr!r}")
    dotted, raw = expr.split("=", 1)
    parts = [p for p in dotted.strip().split(".") if p]
    if not parts:
        raise ConfigError(f"--set expects key=value, got {expr!r}")
    node: dict[str, Any] = data
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ConfigError(f"--set path not found: {dotted!r} (at {part!r})")
        node = child
    leaf = parts[-1]
    if leaf not in node:
        warnings.append(f"--set targets unknown config key: {dotted}")
    node[leaf] = yaml.safe_load(raw)


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> AppConfig:
    """Load ``config.yaml`` (or *path*), apply overrides, validate and return it."""
    explicit = path is not None
    cfg_path = Path(path).expanduser() if explicit else Path(DEFAULT_CONFIG_NAME)
    data: dict[str, Any] = {}
    if cfg_path.exists():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"config root must be a mapping, got {type(loaded).__name__}")
        data = loaded
    elif explicit:
        raise ConfigError(f"config file not found: {cfg_path}")

    warnings: list[str] = []
    merged = _merge(_plain(AppConfig()), data, "", warnings)
    for expr in overrides or []:
        _apply_override(merged, expr, warnings)

    cfg = _build(AppConfig, merged, warnings)
    cfg.source_path = cfg_path.resolve() if cfg_path.exists() else None
    cfg.warnings = warnings
    cfg.validate()
    return cfg


def save_effective_config(cfg: AppConfig, path: str | Path) -> Path:
    """Write the fully-merged configuration for debugging/inspection."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out
