"""Self-check / readiness report (Phase 10).

Run before hands-on testing::

    .\\.venv\\Scripts\\python.exe -m src.doctor
    .\\run.ps1 -Doctor

Checks the interpreter, dependencies, OCR, configuration, log writability
and - informationally - whether the game window is present. Exits 1 when
any check FAILs (WARN does not fail the run).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from src.config import AppConfig, ConfigError, load_config

MIN_PYTHON = (3, 11)
REQUIRED_MODULES = ("numpy", "cv2", "yaml", "mss")
OPTIONAL_MODULES = ("dxcam", "pytesseract")
MARKS = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


@dataclass
class CheckResult:
    name: str
    status: str  # pass | warn | fail
    detail: str


def check_python() -> CheckResult:
    ok = (sys.version_info.major, sys.version_info.minor) >= MIN_PYTHON
    version = sys.version.split()[0]
    detail = f"{version} on {sys.platform}"
    if not ok:
        detail += f" (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})"
    return CheckResult("python", "pass" if ok else "fail", detail)


def check_imports() -> CheckResult:
    missing = [mod for mod in REQUIRED_MODULES if not _importable(mod)]
    if missing:
        return CheckResult("imports", "fail", f"missing: {', '.join(missing)}")
    notes = [f"{mod} unavailable" for mod in OPTIONAL_MODULES if not _importable(mod)]
    detail = f"{', '.join(REQUIRED_MODULES)} ok"
    if notes:
        detail += f" ({'; '.join(notes)})"
        return CheckResult("imports", "warn", detail)
    return CheckResult("imports", "pass", detail)


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def check_tesseract() -> CheckResult:
    try:
        import pytesseract

        from src.vision.ocr import ensure_tesseract

        ensure_tesseract(pytesseract)
        version = str(pytesseract.get_tesseract_version())
        return CheckResult("tesseract", "pass", version)
    except Exception as exc:
        return CheckResult(
            "tesseract", "warn",
            f"not installed - OCR disabled ({type(exc).__name__})",
        )


def check_window(cfg: AppConfig) -> CheckResult:
    try:
        from src.capture.window_manager import GameWindowManager

        info = GameWindowManager(cfg.window).current()
    except Exception as exc:
        return CheckResult("window", "warn", f"enumeration failed: {type(exc).__name__}")
    if info is None:
        return CheckResult("window", "warn", "RDR2 window not found (start Story Mode)")
    return CheckResult(
        "window", "pass",
        f"{info.title!r} {info.client.width}x{info.client.height}",
    )


def check_log_dir(cfg: AppConfig) -> CheckResult:
    log_dir = cfg.resolve(cfg.telemetry.dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        probe = log_dir / ".doctor_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckResult("logs", "fail", f"not writable: {log_dir} ({exc})")
    return CheckResult("logs", "pass", str(log_dir))


def check_input() -> CheckResult:
    try:
        from src.input.keys import normalize_key

        normalize_key("w")
        normalize_key("f12")
    except Exception as exc:
        return CheckResult("input", "fail", f"key map broken: {exc}")
    return CheckResult(
        "input", "pass", "key map ok - live test: tools/input_check.py, then --demo",
    )


def check_output_dirs(cfg: AppConfig) -> CheckResult:
    """WARN when a configured write path resolves outside the project folder."""
    root = Path(cfg.project_root).resolve()
    names = ("telemetry.dir", "recording.dir", "rl.checkpoint", "rl.buffer")
    values = (
        cfg.telemetry.dir, cfg.recording.dir, cfg.rl.checkpoint, cfg.rl.buffer,
    )
    outside: list[str] = []
    for name, value in zip(names, values, strict=True):
        try:
            cfg.resolve(value).resolve().relative_to(root)
        except (ValueError, OSError):
            outside.append(name)
    if outside:
        return CheckResult(
            "outputs", "warn",
            f"outside project folder: {', '.join(outside)} "
            f"(project: {root})",
        )
    return CheckResult("outputs", "pass", "all output paths inside project")


def run_checks(config_path: str | None = None) -> list[CheckResult]:
    """Run every check; never raises (a broken config becomes a FAIL row)."""
    results = [check_python(), check_imports(), check_tesseract()]
    cfg: AppConfig | None = None
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        results.append(CheckResult("config", "fail", str(exc)[:240]))
        return results
    results.append(
        CheckResult(
            "config", "pass",
            f"phase {cfg.agent.phase}, loop {cfg.agent.loop_hz} Hz, "
            f"{len(cfg.warnings)} warning(s)",
        )
    )
    for warning in cfg.warnings[:3]:
        results.append(CheckResult("config", "warn", warning))
    results.append(check_window(cfg))
    results.append(check_log_dir(cfg))
    results.append(check_output_dirs(cfg))
    results.append(check_input())
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.doctor",
        description="Environment/configuration self-check for RDR2 AI.",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_checks(args.config)
    failed = 0
    for result in results:
        print(f"{MARKS.get(result.status, '????')}  {result.name}: {result.detail}")
        if result.status == "fail":
            failed += 1
    print(f"{len(results)} checks, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
