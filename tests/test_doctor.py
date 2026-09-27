"""Tests for the Phase 10 doctor self-check."""

from __future__ import annotations

from pathlib import Path

from src.doctor import CheckResult, main, run_checks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = str(PROJECT_ROOT / "config.yaml")


def test_run_checks_covers_essentials() -> None:
    results = run_checks(CONFIG_PATH)
    assert isinstance(results, list) and results
    assert all(isinstance(r, CheckResult) for r in results)
    assert all(r.status in {"pass", "warn", "fail"} for r in results)
    names = {r.name for r in results}
    assert {"python", "imports", "tesseract", "config"} <= names
    python = next(r for r in results if r.name == "python")
    assert python.status == "pass"
    config = next(r for r in results if r.name == "config")
    assert config.status == "pass"
    assert "fail" not in {r.status for r in results if r.name == "config"}


def test_doctor_main_exit_zero_on_default_config(capsys) -> None:
    rc = main(["--config", CONFIG_PATH])
    out = capsys.readouterr().out
    assert rc == 0
    assert "FAIL" not in out
    assert "checks," in out


def test_missing_config_fails(tmp_path: Path) -> None:
    results = run_checks(str(tmp_path / "missing.yaml"))
    fails = [r for r in results if r.status == "fail"]
    assert fails and fails[0].name == "config"
    rc = main(["--config", str(tmp_path / "missing.yaml")])
    assert rc == 1


def test_invalid_config_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("capture:\n  target_fps: 0\n", encoding="utf-8")
    results = run_checks(str(bad))
    assert any(r.status == "fail" and r.name == "config" for r in results)
    assert main(["--config", str(bad)]) == 1


def test_window_check_is_informational() -> None:
    results = run_checks(CONFIG_PATH)
    window = next(r for r in results if r.name == "window")
    assert window.status in {"pass", "warn"}


def test_log_dir_check_passes(tmp_path: Path) -> None:
    results = run_checks(CONFIG_PATH)
    logs = next(r for r in results if r.name == "logs")
    assert logs.status == "pass"
