"""Debug overlay drawn on top of captured frames."""

from __future__ import annotations

import cv2
import numpy as np

from src.state.agent_status import AgentStatus

STATE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "INIT": (190, 190, 190),
    "RUNNING": (80, 200, 80),
    "PAUSED": (0, 200, 255),
    "TAKEOVER": (0, 170, 255),
    "STOPPED": (60, 60, 230),
    "ERROR": (40, 40, 245),
}


def state_color(state: str) -> tuple[int, int, int]:
    return STATE_COLORS_BGR.get(state, (200, 200, 200))


def _pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.0f}%"


def _world_line(status: AgentStatus) -> str:
    time_part = status.world_time or "--"
    weather = status.world_weather or "--"
    ammo = "--" if status.world_ammo is None else str(status.world_ammo)
    horse = "Y" if status.world_horse_detected else "N"
    return f"{time_part} {weather} ammo {ammo} horse {horse}  {status.world_ms:.1f}ms"


def build_lines(status: AgentStatus) -> list[str]:
    """Human-readable status lines shared by overlay and side panel."""
    bounds = status.window_bounds
    window = status.window_title or "(not found)"
    if len(window) > 34:
        window = window[:31] + "..."
    size = f"{status.frame_size[0]}x{status.frame_size[1]}" if status.frame_size else "-"
    held = ", ".join(status.held_keys + status.held_buttons) or "none"
    conf = f"{status.confidence * 100:.0f}%" if status.confidence is not None else "n/a"
    minimap = "-" if status.hud_minimap is None else ("Y" if status.hud_minimap else "N")
    lines = [
        "RDR2 AI  PHASE " + str(status.phase),
        "STATUS : " + status.state,
        "WINDOW : " + window,
        "CLIENT : " + (f"{bounds[2]}x{bounds[3]} @ {bounds[0]},{bounds[1]}"
                       if bounds else "-") + f"  focus={'Y' if status.window_focused else 'N'}",
        f"CAPTURE: {status.capture_backend} {status.capture_fps:.1f} fps  "
        f"lat {status.capture_latency_ms:.1f}ms",
        f"FRAME  : #{status.frame_id} age {status.frame_age_ms:.0f}ms  {size}",
        f"VISION : {status.vision_ms:.1f}ms  bright {status.brightness:.0f}  "
        f"motion {status.motion:.1f}",
        f"HUD    : HP {_pct(status.hud_health)} ST {_pct(status.hud_stamina)} "
        f"DE {_pct(status.hud_dead_eye)}  map {minimap}  "
        f"{status.hud_ms:.1f}ms  ocr={status.ocr_engine}",
        "WORLD  : " + _world_line(status),
        f"LOOP   : {status.loop_hz:.1f} Hz   input lat {status.input_latency_ms:.1f}ms",
        "HELD   : " + held,
        f"GOAL   : {status.goal}   ACTION: {status.action}",
        "CONF   : " + conf,
    ]
    if status.hud_prompt_visible or status.hud_prompt:
        prompt = status.hud_prompt or "(visible, reading...)"
        if len(prompt) > 62:
            prompt = prompt[:59] + "..."
        lines.append("PROMPT : " + prompt)
    if status.dialogue_active or status.dialogue_text:
        state = "ON " if status.dialogue_active else "OFF"
        text = status.dialogue_text or "(reading...)"
        if len(text) > 54:
            text = text[:51] + "..."
        encounter = f"  enc={status.dialogue_encounter}" if status.dialogue_encounter else ""
        lines.append(f"DIALOG : {state} {text}{encounter}")
    if status.demo_step:
        lines.append("DEMO   : " + status.demo_step)
    if status.message:
        msg = status.message if len(status.message) <= 70 else status.message[:67] + "..."
        lines.append("MSG    : " + msg)
    if status.error:
        err = status.error if len(status.error) <= 70 else status.error[:67] + "..."
        lines.append("ERROR  : " + err)
    return lines


def draw_overlay(
    image: np.ndarray,
    status: AgentStatus,
    *,
    copy: bool = True,
    scale: float = 0.55,
    line_gap: int = 20,
    margin: int = 10,
) -> np.ndarray:
    """Draw the status block and a state-colored border on *image*."""
    canvas = image.copy() if copy else image
    lines = build_lines(status)
    color = state_color(status.state)

    box_w = min(canvas.shape[1] - 2 * margin, 470)
    box_h = line_gap * len(lines) + 2 * margin
    if box_h < canvas.shape[0] and box_w > 40:
        roi = canvas[margin:margin + box_h, margin:margin + box_w]
        if roi.shape[:2] == (box_h, box_w):
            overlay = np.full_like(roi, (18, 18, 18))
            blended = cv2.addWeighted(overlay, 0.62, roi, 0.38, 0)
            canvas[margin:margin + box_h, margin:margin + box_w] = blended

    thickness = 3
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, thickness)

    y = margin + line_gap - 5
    for i, line in enumerate(lines):
        line_color = color if i < 2 else (235, 235, 235)
        if line.startswith("ERROR"):
            line_color = (60, 60, 255)
        cv2.putText(
            canvas, line, (margin + 4, y),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, line, (margin + 4, y),
            cv2.FONT_HERSHEY_SIMPLEX, scale, line_color, 1, cv2.LINE_AA,
        )
        y += line_gap

    hud_color = (0, 255, 255)
    for box, label in zip(status.hud_boxes, status.hud_labels, strict=False):
        if len(box) != 4:
            continue
        bx, by, bw, bh = (int(v) for v in box)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), hud_color, 2)
        if label:
            ty = max(14, by - 6)
            cv2.putText(
                canvas, label, (bx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, label, (bx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA,
            )
    return canvas
