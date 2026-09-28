"""Small status card window (no duplicated game frame unless enabled)."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

import cv2
import numpy as np

from src.capture.screen_capture import FramePacket
from src.config import DebugConfig
from src.state.agent_status import AgentStatus
from src.ui.debug_overlay import build_card, build_lines, draw_overlay, state_color

log = logging.getLogger(__name__)

_WM_SETICON = 0x80
_ICON_BIG = 1
_ICON_SMALL = 0
_IMAGE_ICON = 1
_LR_LOADFROMFILE = 0x10
_LR_DEFAULTSIZE = 0x40
_IDC_ARROW = 32512
_GCL_HCURSOR = -12


def window_icon_path() -> Path | None:
    """Repo-bundled card icon; None when the asset is missing."""
    path = Path(__file__).resolve().parents[2] / "assets" / "Outlaw.ico"
    return path if path.is_file() else None


def apply_window_icon(window_name: str, icon_path: Path) -> bool:
    """Pin the icon on the card window (title bar and taskbar button)."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return False
        hicon = user32.LoadImageW(
            None, str(icon_path), _IMAGE_ICON, 0, 0,
            _LR_LOADFROMFILE | _LR_DEFAULTSIZE,
        )
        if not hicon:
            return False
        user32.SendMessageW(hwnd, _WM_SETICON, _ICON_BIG, hicon)
        user32.SendMessageW(hwnd, _WM_SETICON, _ICON_SMALL, hicon)
        return True
    except Exception:
        return False


def apply_arrow_cursor(window_name: str) -> bool:
    """Replace OpenCV's crosshair with the normal arrow pointer."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return False
        arrow = user32.LoadCursorW(None, _IDC_ARROW)
        if not arrow:
            return False
        set_class_long = getattr(user32, "SetClassLongPtrW", None)
        if set_class_long is None:
            return False
        return bool(set_class_long(hwnd, _GCL_HCURSOR, arrow))
    except Exception:
        return False


class Dashboard:
    """Renders the status card. Fails safe: GUI errors disable the window only."""

    def __init__(self, cfg: DebugConfig, *, emergency: str = "F12",
                 pause: str = "F11", takeover: str = "F10") -> None:
        self._cfg = cfg
        self._disabled = not cfg.gui
        self._created = False
        self._keys = (emergency, pause, takeover)
        self._chrome_applied = False
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

    def _ensure_chrome(self) -> None:
        """Pin the icon and arrow cursor once the native window exists."""
        if self._chrome_applied:
            return
        self._chrome_applied = True
        icon = window_icon_path()
        if icon is not None:
            apply_window_icon(self._cfg.window_name, icon)
        apply_arrow_cursor(self._cfg.window_name)

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
                    status, title=self._cfg.app_title,
                    emergency=emergency, pause=pause,
                    takeover=takeover, waiting=packet is None,
                )
            cv2.imshow(self._cfg.window_name, view)
            cv2.waitKey(1)
            self._ensure_chrome()
            self.shown += 1
        except Exception as exc:
            self.errors += 1
            if self.errors == 1 or self.errors % 100 == 0:
                log.error("dashboard disabled after rendering error: %s", exc)
            self._disabled = True

    def visible(self) -> bool:
        """False once the user closes the card window (errors count as open)."""
        if self._disabled or not self._created:
            return True
        try:
            return bool(cv2.getWindowProperty(
                self._cfg.window_name, cv2.WND_PROP_VISIBLE
            ))
        except Exception:
            return True

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
