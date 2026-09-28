"""Debug overlay drawn on top of captured frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.state.agent_status import AgentStatus

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

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
    if status.threat_level != "none":
        enemies = "?" if status.threat_enemies is None else str(status.threat_enemies)
        fire = "fire Y" if status.threat_fire else "fire N"
        lines.append(
            f"THREAT : {status.threat_level}  enemies {enemies}  "
            f"wanted {status.threat_wanted}  {fire}"
        )
    if status.survival_active or status.survival_low:
        active = status.survival_active or "-"
        low = status.survival_low or "-"
        lines.append(f"SURV   : remedy {active}  low cores {low}")
    if status.rl_mode:
        lines.append(
            f"RL     : {status.rl_mode} {status.rl_action} p={status.rl_prob:.2f}  "
            f"steps={status.rl_steps}  buf={status.rl_buffer}"
        )
    elif status.rl_note:
        note = status.rl_note if len(status.rl_note) <= 66 else status.rl_note[:63] + "..."
        lines.append("RL     : " + note)
    if status.demo_step:
        lines.append("DEMO   : " + status.demo_step)
    if status.message:
        msg = status.message if len(status.message) <= 70 else status.message[:67] + "..."
        lines.append("MSG    : " + msg)
    if status.error:
        err = status.error if len(status.error) <= 70 else status.error[:67] + "..."
        lines.append("ERROR  : " + err)
    return lines


def card_lines(status: AgentStatus) -> list[tuple[str, str]]:
    """Curated (text, kind) rows for the small status card.

    Kinds: state, doing, msg, cores, threat, prompt, dialog, error,
    warn, stats. Only rows with something to say are included.
    """
    rows = [("STATUS  " + status.state + f"  (phase {status.phase})", "state")]
    doing = status.goal or "IDLE"
    if status.action and status.action != "none":
        doing += "  /  " + status.action
    rows.append(("DOING   " + doing, "doing"))
    if status.message:
        rows.append(("MSG     " + status.message, "msg"))
    rows.append(
        (f"CORES   HP {_pct(status.hud_health)}  "
         f"ST {_pct(status.hud_stamina)}  DE {_pct(status.hud_dead_eye)}",
         "cores")
    )
    if status.threat_level != "none":
        hot = status.threat_level in ("wanted", "under_fire")
        rows.append((
            f"THREAT  {status.threat_level}  wanted {status.threat_wanted}  "
            f"fire {'Y' if status.threat_fire else 'N'}",
            "alert" if hot else "threat",
        ))
    if status.hud_prompt_visible or status.hud_prompt:
        rows.append(("PROMPT  " + (status.hud_prompt or "(reading...)"), "prompt"))
    if status.dialogue_active or status.dialogue_text:
        rows.append((
            "SAYS    " + (status.dialogue_text or "(reading...)")
            + (f"  [{status.dialogue_encounter}]" if status.dialogue_encounter else ""),
            "dialog",
        ))
    if status.error:
        rows.append(("ERROR   " + status.error, "error"))
    if not status.window_focused:
        rows.append(("Click the game window so input is accepted", "warn"))
    rows.append(
        (f"{status.capture_fps:.0f} fps  |  loop {status.loop_hz:.0f} Hz  |  "
         f"ocr {status.ocr_engine}", "stats")
    )
    return rows


_CARD_WIDTH = 400
_CARD_ROW_H = 22
_CARD_MARGIN = 12
_CARD_MAX_ROWS = 13
_TITLE_RED_RGB = (238, 34, 35)
_TITLE_FONT_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "fonts" / "Rye-Regular.ttf"
)
_title_fonts: dict[int, object] = {}

_CARD_STYLE: dict[str, tuple[float, tuple[int, int, int]]] = {    "state": (0.60, (80, 200, 80)),
    "doing": (0.50, (235, 235, 235)),
    "msg": (0.46, (200, 210, 230)),
    "cores": (0.50, (235, 235, 235)),
    "threat": (0.50, (235, 235, 235)),
    "alert": (0.50, (0, 165, 255)),
    "prompt": (0.46, (0, 255, 255)),
    "dialog": (0.46, (200, 210, 230)),
    "error": (0.50, (60, 60, 255)),
    "warn": (0.50, (0, 255, 255)),
    "stats": (0.45, (150, 150, 150)),
}


def _slab_font(size_px: int):
    """Western slab title font, cached per size; raises when unavailable."""
    if ImageFont is None:
        raise ImportError("Pillow is not installed")
    cached = _title_fonts.get(size_px)
    if cached is None:
        cached = ImageFont.truetype(str(_TITLE_FONT_PATH), size_px)
        _title_fonts[size_px] = cached
    return cached


def _draw_title(card: np.ndarray, title: str) -> None:
    """Red slab-serif title; falls back to a Hershey face when needed."""
    try:
        rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)
        size = 40
        font = _slab_font(size)
        limit = _CARD_WIDTH - 2 * _CARD_MARGIN
        while size > 14:
            box = draw.textbbox((0, 0), title, font=font)
            if box[2] - box[0] <= limit:
                break
            size -= 2
            font = _slab_font(size)
        draw.text((_CARD_MARGIN, 8), title, font=font, fill=_TITLE_RED_RGB)
        card[:] = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    except Exception:
        _draw_card_text(card, title, _CARD_MARGIN, 40, 0.9,
                        (35, 34, 238), cv2.FONT_HERSHEY_TRIPLEX)


def _draw_card_text(canvas: np.ndarray, text: str, x: int, y: int,
                    scale: float, color: tuple[int, int, int],
                    font: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
    cv2.putText(canvas, text, (x, y),
                font, scale, color, 1, cv2.LINE_AA)


def build_card(status: AgentStatus, *, title: str = "RDR2 AI",
               emergency: str = "F12",
               pause: str = "F11", takeover: str = "F10",
               waiting: bool = False) -> np.ndarray:
    """Render the small status card (no game frame): title, rows, hotkeys."""
    rows = card_lines(status)
    if waiting:
        rows.insert(1, ("Waiting for game frames...", "warn"))
    wrapped: list[tuple[str, str]] = []
    max_chars = (_CARD_WIDTH - 2 * _CARD_MARGIN) // 8
    for text, kind in rows:
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [""]
        for chunk in chunks:
            wrapped.append((chunk, kind))
            if len(wrapped) >= _CARD_MAX_ROWS:
                break
        if len(wrapped) >= _CARD_MAX_ROWS:
            break
    height = (50 + 14 + len(wrapped) * _CARD_ROW_H + 10 + 24 + 10)
    card = np.full((height, _CARD_WIDTH, 3), 22, dtype=np.uint8)
    color = state_color(status.state)
    cv2.rectangle(card, (0, 0), (_CARD_WIDTH - 1, 6), color, -1)

    _draw_title(card, title or "RDR2 AI")

    y = 50 + 14
    for text, kind in wrapped:
        scale, row_color = _CARD_STYLE.get(kind, (0.5, (235, 235, 235)))
        if kind == "state":
            row_color = color
        _draw_card_text(card, text, _CARD_MARGIN, y, scale, row_color)
        y += _CARD_ROW_H

    footer = f"{emergency} stop  |  {pause} pause  |  {takeover} takeover"
    cv2.line(card, (_CARD_MARGIN, y + 2),
             (_CARD_WIDTH - _CARD_MARGIN, y + 2), (70, 70, 70), 1)
    _draw_card_text(card, footer, _CARD_MARGIN, y + 22, 0.45, (170, 170, 170))
    return card


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
