"""End-to-end run of the Phase 1 agent without the game (synthetic capture)."""

from __future__ import annotations

import json
from pathlib import Path

from src.main import main

BASE_ARGS = [
    "--backend", "synthetic",
    "--headless",
    "--set", "capture.target_fps=30",
    "--set", "agent.loop_hz=10",
    "--set", "safety.hotkey_poll_hz=100",
    "--set", "safety.startup_grace_s=1",
]


def test_full_run_synthetic_headless(tmp_path: Path) -> None:
    args = BASE_ARGS + [
        "--duration", "3",
        "--set", "telemetry.console=false",
        "--set", f"telemetry.dir={tmp_path.as_posix()}",
        "--set", "telemetry.file=true",
    ]
    rc = main(args)
    assert rc == 0

    events = tmp_path / "events.jsonl"
    assert events.exists()
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    kinds = [row["kind"] for row in rows]
    assert "startup" in kinds
    assert "metrics" in kinds
    assert kinds[-1] == "shutdown"
    shutdown = rows[-1]
    assert shutdown["capture"]["frames"] > 5
    assert shutdown["state"] in {"RUNNING", "STOPPED"}


def test_run_with_demo_flag_completes(tmp_path: Path) -> None:
    args = BASE_ARGS + [
        "--duration", "4",
        "--demo",
        "--set", "telemetry.console=false",
        "--set", f"telemetry.dir={tmp_path.as_posix()}",
    ]
    rc = main(args)
    assert rc == 0
    events = tmp_path / "events.jsonl"
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    demo_rows = [r for r in rows if r["kind"] == "demo"]
    assert demo_rows == [] or all("step" in r for r in demo_rows)


def test_config_error_exit_code(tmp_path: Path) -> None:
    rc = main(["--config", str(tmp_path / "missing.yaml")])
    assert rc == 2


def test_invalid_override_exit_code() -> None:
    rc = main([
        "--headless",
        "--set", "capture.target_fps=0",
        "--set", "telemetry.console=false",
        "--set", "telemetry.file=false",
    ])
    assert rc == 2


def test_version_flag_exits() -> None:
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
