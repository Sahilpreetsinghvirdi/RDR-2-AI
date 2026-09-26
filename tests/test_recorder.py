"""Session recorder file output tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.capture.screen_capture import FramePacket
from src.config import RecordingConfig
from src.telemetry.recorder import SessionRecorder


def make_packet(frame_id: int) -> FramePacket:
    return FramePacket(
        image=np.full((48, 64, 3), 30, dtype=np.uint8),
        frame_id=frame_id,
        wall_time=time.time(),
        timestamp=time.monotonic(),
        source="test",
        capture_latency_ms=1.5,
    )


def test_records_frames_and_states(tmp_path: Path) -> None:
    cfg = RecordingConfig(fps=10_000, max_frames=5, save_state_jsonl=True)
    rec = SessionRecorder(cfg, tmp_path)
    session = rec.start({"backend": "synthetic"})
    assert session.exists()
    saved = 0
    for i in range(20):
        if rec.record(make_packet(i), {"goal": "TEST"}):
            saved += 1
        time.sleep(0.003)
    rec.close()

    assert saved == 5
    frames = sorted((session / "frames").glob("*.jpg"))
    assert len(frames) == 5
    states = (session / "states.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(states) == 5
    row = json.loads(states[0])
    assert row["state"]["goal"] == "TEST"
    meta = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    assert meta["frames_saved"] == 5
    assert "ended_at" in meta


def test_disabled_recorder_does_nothing(tmp_path: Path) -> None:
    rec = SessionRecorder(RecordingConfig(enabled=False), tmp_path)
    assert rec.enabled is False
    assert rec.record(make_packet(1)) is False
    rec.close()


def test_rate_limiting_by_fps(tmp_path: Path) -> None:
    cfg = RecordingConfig(fps=1, max_frames=100)
    rec = SessionRecorder(cfg, tmp_path)
    rec.start()
    first = rec.record(make_packet(1))
    second = rec.record(make_packet(2))
    rec.close()
    assert first is True
    assert second is False


def test_unique_session_dirs(tmp_path: Path) -> None:
    cfg = RecordingConfig(fps=10_000, max_frames=1)
    a = SessionRecorder(cfg, tmp_path).start()
    b = SessionRecorder(cfg, tmp_path).start()
    assert a != b


def test_event_record_is_swallowed_when_no_session(tmp_path: Path) -> None:
    rec = SessionRecorder(RecordingConfig(), tmp_path)
    rec.record_event({"hello": "world"})
    rec.close()
