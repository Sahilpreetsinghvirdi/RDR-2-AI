"""Story wiring: the director takes the planner slot and auto-greet arms."""

from __future__ import annotations

import json
from pathlib import Path

from src.main import main
from tests.test_main_guard import FakeWindowManager


class RecordingStory:
    instances: list[RecordingStory] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.steps = 0
        self.kwargs = dict(kwargs)
        RecordingStory.instances.append(self)

    def step(
        self,
        now: float,
        status: object,
        frame: object | None = None,
        game_state: object | None = None,
    ) -> None:
        self.steps += 1
        status.update(goal="STORY TEST", action="none")  # type: ignore[attr-defined]


class RecordingResponder:
    captured: dict[str, str] | None = None

    def __init__(self, dialogue_cfg: object, locomotion: object) -> None:
        RecordingResponder.captured = dict(
            dialogue_cfg.respond_to  # type: ignore[arg-type]
        )
        self.taps = 0

    def step(self, now: float, status: object, game_state: object) -> str | None:
        return None


def test_story_takes_planner_slot_and_arms_greet(
    tmp_path: Path, monkeypatch
) -> None:
    RecordingStory.instances = []
    RecordingResponder.captured = None
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join([
            "story:",
            "  enabled: true",
            "  missions:",
            "    - name: smoke",
            "      legs: [[0.0, 5.0]]",
            "      tasks: []",
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
    monkeypatch.setattr("src.main.StoryRunner", RecordingStory)
    monkeypatch.setattr("src.main.PromptResponder", RecordingResponder)

    rc = main(["--config", str(cfg_path), "--headless", "--duration", "2"])
    assert rc == 0

    assert RecordingStory.instances, "StoryRunner was not constructed"
    story = RecordingStory.instances[0]
    assert story.steps > 0, "story director never stepped"
    assert story.kwargs.get("ocr") is not None, "shared OCR engine not passed"
    assert RecordingResponder.captured is not None, "responder not armed"
    assert RecordingResponder.captured.get("greet") == "interact"

    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    kinds = [row["kind"] for row in rows]
    assert "crash" not in kinds
    assert kinds[-1] == "shutdown"


def _write_base_config(tmp_path: Path, story_block: str) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join([
            story_block,
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
    return cfg_path


def test_story_flag_enables_director(tmp_path: Path, monkeypatch) -> None:
    RecordingStory.instances = []
    cfg_path = _write_base_config(
        tmp_path,
        "story:\n  missions:\n    - name: flagged\n      tasks: []\n",
    )
    monkeypatch.setattr("src.main.GameWindowManager", FakeWindowManager)
    monkeypatch.setattr("src.main.StoryRunner", RecordingStory)
    rc = main(["--config", str(cfg_path), "--story", "--headless",
               "--duration", "1"])
    assert rc == 0
    assert RecordingStory.instances, "StoryRunner was not built via --story"
    assert RecordingStory.instances[0].steps > 0


def test_story_flag_without_missions_fails(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_base_config(tmp_path, "")
    monkeypatch.setattr("src.main.GameWindowManager", FakeWindowManager)
    rc = main(["--config", str(cfg_path), "--story", "--headless",
               "--duration", "1"])
    assert rc == 2
