"""Overlay drawing and status snapshot tests."""

from __future__ import annotations

import numpy as np

from src.state.agent_status import AgentStatus, StatusTracker
from src.ui.debug_overlay import build_lines, draw_overlay, state_color


def _frame() -> np.ndarray:
    return np.full((360, 640, 3), 60, dtype=np.uint8)


def test_build_lines_contains_state_and_goal() -> None:
    status = AgentStatus(state="RUNNING", goal="TRAVEL", action="follow road",
                         confidence=0.91)
    lines = build_lines(status)
    text = "\n".join(lines)
    assert "RUNNING" in text
    assert "TRAVEL" in text
    assert "91%" in text


def test_build_lines_confidence_none_shows_na() -> None:
    lines = build_lines(AgentStatus())
    assert any("n/a" in line for line in lines)


def test_build_lines_includes_hud_row() -> None:
    status = AgentStatus(hud_health=0.92, hud_stamina=0.5, hud_minimap=True,
                         ocr_engine="tesseract")
    text = "\n".join(build_lines(status))
    assert "HP 92%" in text
    assert "ST 50%" in text
    assert "map Y" in text
    assert "ocr=tesseract" in text


def test_build_lines_unknown_gauges_show_dashes() -> None:
    text = "\n".join(build_lines(AgentStatus()))
    assert "HP --" in text
    assert "map -" in text


def test_build_lines_prompt_line() -> None:
    status = AgentStatus(hud_prompt_visible=True, hud_prompt="PRESS E TO MOUNT")
    text = "\n".join(build_lines(status))
    assert "PROMPT : PRESS E TO MOUNT" in text


def test_draw_overlay_draws_hud_boxes() -> None:
    frame = _frame()
    status = AgentStatus(state="RUNNING", hud_boxes=[[100, 200, 50, 40]],
                         hud_labels=["HP 75%"])
    out = draw_overlay(frame, status)
    assert out.shape == frame.shape
    assert out.sum() != frame.sum()


def test_state_colors_are_distinct() -> None:
    colors = {s: state_color(s) for s in ("RUNNING", "PAUSED", "STOPPED", "ERROR")}
    assert len(set(colors.values())) == len(colors)


def test_draw_overlay_returns_new_image() -> None:
    frame = _frame()
    out = draw_overlay(frame, AgentStatus(state="RUNNING"))
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert not np.shares_memory(out, frame)
    assert out.sum() != frame.sum()


def test_draw_overlay_border_changes_with_state() -> None:
    frame = _frame()
    running = draw_overlay(frame, AgentStatus(state="RUNNING"))
    stopped = draw_overlay(frame, AgentStatus(state="STOPPED"))
    assert running[0, 0].tolist() != stopped[0, 0].tolist()


def test_draw_overlay_handles_long_messages() -> None:
    status = AgentStatus(message="x" * 500, error="y" * 500, state="ERROR")
    out = draw_overlay(_frame(), status)
    assert out.shape == (360, 640, 3)


def test_status_tracker_snapshot_isolated() -> None:
    tracker = StatusTracker(phase=3)
    tracker.update(goal="EXPLORE")
    snap = tracker.snapshot()
    tracker.update(goal="FIGHT")
    assert snap.goal == "EXPLORE"
    assert tracker.snapshot().goal == "FIGHT"
    assert snap.phase == 3


def test_status_tracker_rejects_unknown_field() -> None:
    tracker = StatusTracker()
    try:
        tracker.update(nonexistent=1)
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError")


def test_status_to_dict_round_trip_keys() -> None:
    data = AgentStatus().to_dict()
    assert "state" in data and "loop_hz" in data and "held_keys" in data
