"""Small status card window (no duplicated game frame unless enabled)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.capture.screen_capture import FramePacket
from src.config import DebugConfig
from src.state.agent_status import AgentStatus
from src.ui.debug_overlay import build_card, build_lines, draw_overlay, state_color

log = logging.getLogger(__name__)


class Dashboard:
    """Renders the status card. Fails safe: GUI errors disable the window only."""

    def __init__(self, cfg: DebugConfig, *, emergency: str = "F12",
                 pause: str = "F11", takeover: str = "F10") -> None:
        self._cfg = cfg
        self._disabled = not cfg.gui
        self._created = False
        self._keys = (emergency, pause, takeover)
        self.shown = 0
        self.errors = 0

    @property
    def enabled(self) -> bool:
        return not self._disabled

    def _ensure_window(self) -> None:
        if self._created:
            return
        flags = cv2.WINDOW_NORMAL if self._cfg.show_video else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(self._cfg.window_name, flags)
        self._created = True

    def _panel(self, status: AgentStatus, height: int) -> np.ndarray:
        width = max(160, self._cfg.panel_width)
        panel = np.full((height, width, 3), 24, dtype=np.uint8)
        color = state_color(status.state)
        cv2.rectangle(panel, (0, 0), (width - 1, 6), color, -1)
        cv2.putText(
            panel, status.state, (10, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
        )
        y = 62
        for line in build_lines(status):
            text = line
            max_chars = max(8, (width - 20) // 8)
            chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [""]
            for chunk in chunks:
                if y > height - 8:
                    return panel
                bright = (235, 235, 235) if not chunk.startswith(("ERROR", "MSG")) else (
                    (60, 60, 255) if chunk.startswith("ERROR") else (200, 210, 230)
                )
                cv2.putText(
                    panel, chunk, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    panel, chunk, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bright, 1, cv2.LINE_AA,
                )
                y += 17
            y += 3
        return panel

    def show(self, packet: FramePacket | None, status: AgentStatus) -> None:
        if self._disabled:
            return
        try:
            self._ensure_window()
            if self._cfg.show_video and packet is not None:
                frame = packet.image
                if self._cfg.show_overlay:
                    frame = draw_overlay(frame, status)
                else:
                    frame = frame.copy()
                panel = self._panel(status, frame.shape[0])
                if panel.shape[0] != frame.shape[0]:
                    panel = cv2.resize(panel, (panel.shape[1], frame.shape[0]))
                view = np.hstack([frame, panel])
            else:
                emergency, pause, takeover = self._keys
                view = build_card(
                    status, emergency=emergency, pause=pause,
                    takeover=takeover, waiting=packet is None,
                )
            cv2.imshow(self._cfg.window_name, view)
            cv2.waitKey(1)
            self.shown += 1
        except Exception as exc:
            self.errors += 1
            if self.errors == 1 or self.errors % 100 == 0:
                log.error("dashboard disabled after rendering error: %s", exc)
            self._disabled = True

    def close(self) -> None:
        if not self._created:
            return
        try:
            cv2.destroyWindow(self._cfg.window_name)
        except Exception:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        self._created = False
