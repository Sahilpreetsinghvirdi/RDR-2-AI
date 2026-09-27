"""End-to-end ComponentGuard wiring: one failing planner faults and recovers.

Runs the real agent loop (synthetic capture, headless) with a fake game
window and a MissionRunner that always raises. The run must stay healthy:
exactly one ``component_fault`` event for ``planner``, no crash event,
telemetry ``guard`` summary non-empty, clean shutdown.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.capture.window_manager import Rect, WindowInfo
from src.main import main


class FakeWindowManager:
    def __init__(self, cfg: object) -> None:
        self._info = WindowInfo(
            hwnd=1,
            title="Red Dead Redemption 2",
            rect=Rect(0, 0, 1280, 720),
            client=Rect(0, 0, 1280, 720),
            focused=True,
            minimized=False,
            visible=True,
        )

    def current(self, force: bool = False) -> WindowInfo | None:
        return self._info

    def find(self) -> WindowInfo | None:
        return self._info

    def focus(self, hwnd: int, timeout_s: float | None = None) -> bool:
        return True

    def is_foreground(self, hwnd: int) -> bool:
        return True

    def is_alive(self, hwnd: int) -> bool:
        return True


class ExplodingMissionRunner:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def step(
        self,
        now: float,
        status: object,
        frame: object | None = None,
        game_state: object | None = None,
    ) -> bool:
        raise ValueError("boom")


def test_planner_fault_is_isolated(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join([
            "mission:",
            "  enabled: true",
            "  tasks:",
            "    - turn: {deg: 10}",
            "telemetry:",
            "  file: true",
            f"  dir: {tmp_path.as_posix()}",
            "  console: false",
            "capture:",
            "  backend: synthetic",
            "  target_fps: 30",
            "agent:",
            "  loop_hz: 10",
            "safety:",
            "  hotkey_poll_hz: 100",
            "  startup_grace_s: 1",
            "",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.main.GameWindowManager", FakeWindowManager)
    monkeypatch.setattr("src.main.MissionRunner", ExplodingMissionRunner)

    rc = main(["--config", str(cfg_path), "--headless", "--duration", "3"])
    assert rc == 0

    events_path = tmp_path / "events.jsonl"
    assert events_path.exists()
    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    kinds = [row["kind"] for row in rows]
    assert "startup" in kinds
    assert "crash" not in kinds
    assert kinds[-1] == "shutdown"

    faults = [row for row in rows if row["kind"] == "component_fault"]
    assert len(faults) == 1
    assert faults[0]["component"] == "planner"
    assert "ValueError" in faults[0]["error"]
    assert "boom" in faults[0]["error"]

    metrics = [row for row in rows if row["kind"] == "metrics"]
    assert metrics
    guarded = [row["guard"] for row in metrics if row.get("guard")]
    assert guarded
    assert any("planner" in guard for guard in guarded)
