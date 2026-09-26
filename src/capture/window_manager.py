"""RDR2 window discovery and management (Win32, no external dependencies)."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.config import WindowConfig

log = logging.getLogger(__name__)

user32 = ctypes.windll.user32
shcore = getattr(ctypes.windll, "shcore", None)

SW_RESTORE = 9
MONITOR_DEFAULTTONEAREST = 2
PROCESS_PER_MONITOR_DPI_AWARE = 2

_WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


@dataclass(frozen=True)
class Rect:
    """Rectangle in screen coordinates: [left, top, right, bottom)."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def clamp(self, bounds: Rect) -> Rect:
        return Rect(
            max(self.left, bounds.left),
            max(self.top, bounds.top),
            min(self.right, bounds.right),
            min(self.bottom, bounds.bottom),
        )


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    rect: Rect
    client: Rect
    focused: bool
    minimized: bool
    visible: bool

    @property
    def width(self) -> int:
        return self.client.width

    @property
    def height(self) -> int:
        return self.client.height

    def to_dict(self) -> dict[str, object]:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "client": list(self.client.as_tuple()),
            "size": [self.client.width, self.client.height],
            "focused": self.focused,
            "minimized": self.minimized,
        }


def enable_dpi_awareness() -> str:
    """Make Win32 coordinates physical pixels so capture regions line up."""
    try:
        awareness = ctypes.c_void_p(-4)
        if user32.SetProcessDpiAwarenessContext(awareness):
            return "per-monitor-v2"
    except Exception:  # pragma: no cover - older Windows
        pass
    if shcore is not None:
        try:
            if shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE) in (0, 1):
                return "per-monitor"
        except Exception:  # pragma: no cover
            pass
    try:
        if user32.SetProcessDPIAware():
            return "system"
    except Exception:  # pragma: no cover
        pass
    return "unaware"


_BLOCKED_CLASSES = frozenset({
    "consolewindowclass",          # cmd / PowerShell / python consoles
    "cascadia_hosting_window_class",  # Windows Terminal
    "chrome_widgetwin_1",          # Chrome / Edge / any chromium browser
    "mozillawindowclass",          # Firefox
    "applicationframewindow",      # UWP shell windows
    "#32770",                      # common dialog boxes
    "cabinetwclass",               # file explorer
})


def _is_blocked_class(hwnd: int) -> bool:
    """True for console/terminal windows that only mention the game in a path."""
    buf = ctypes.create_unicode_buffer(128)
    user32.GetClassNameW(hwnd, buf, 128)
    return buf.value.lower() in _BLOCKED_CLASSES


class GameWindowManager:
    """Finds and tracks the RDR2 window; never guesses when it is missing."""

    def __init__(self, cfg: WindowConfig) -> None:
        self._cfg = cfg
        self._patterns = [p.lower() for p in cfg.title_patterns]
        self._excluded = [p.lower() for p in cfg.exclude_patterns]
        self._cache: WindowInfo | None = None
        self._last_poll = 0.0
        self._poll_interval = cfg.poll_interval_s

    def _match(self, title: str) -> bool:
        low = title.lower()
        if any(pat in low for pat in self._excluded):
            return False
        if self._cfg.match == "exact":
            return low in self._patterns
        return any(pat in low for pat in self._patterns)

    def _read_info(self, hwnd: int) -> WindowInfo | None:
        if not user32.IsWindow(hwnd) or _is_blocked_class(hwnd):
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if not title or not self._match(title):
            return None

        wr = wt.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(wr)):
            return None
        cr = wt.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(cr)):
            return None
        origin = wt.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(origin))
        client = Rect(origin.x, origin.y, origin.x + cr.right, origin.y + cr.bottom)
        rect = Rect(wr.left, wr.top, wr.right, wr.bottom)
        return WindowInfo(
            hwnd=hwnd,
            title=title,
            rect=rect,
            client=client,
            focused=user32.GetForegroundWindow() == hwnd,
            minimized=bool(user32.IsIconic(hwnd)),
            visible=bool(user32.IsWindowVisible(hwnd)),
        )

    def find(self) -> WindowInfo | None:
        """Enumerate top-level windows and return the best match, if any."""
        matches: list[int] = []

        def _cb(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd) or _is_blocked_class(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if self._match(buf.value):
                matches.append(hwnd)
            return True

        try:
            user32.EnumWindows(_WNDENUMPROC(_cb), 0)
        except Exception:
            log.exception("EnumWindows failed")
            return None

        best: WindowInfo | None = None
        for hwnd in matches:
            info = self._read_info(hwnd)
            if info is None:
                continue
            if info.minimized:
                continue
            if best is None or info.client.area > best.client.area:
                best = info
        return best

    def current(self, force: bool = False) -> WindowInfo | None:
        """Cached find() limited to ``window.poll_interval_s``; None when gone."""
        now = time.monotonic()
        if not force and self._cache is not None and (now - self._last_poll) < self._poll_interval:
            cached = self._read_info(self._cache.hwnd)
            if cached is not None and not cached.minimized:
                self._cache = cached
                return cached
            if cached is not None and cached.minimized:
                return None
            self._cache = None
            self._last_poll = now
            return None
        if not force and self._cache is None and (now - self._last_poll) < self._poll_interval:
            return None
        info = self.find()
        self._cache = info
        self._last_poll = now
        return info

    def wait_for_window(self, timeout_s: float, poll_s: float = 0.25) -> WindowInfo | None:
        """Block up to *timeout_s* (<=0 waits forever) until the game window exists."""
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        while True:
            info = self.find()
            if info is not None:
                return info
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(poll_s)

    def is_alive(self, hwnd: int) -> bool:
        return bool(user32.IsWindow(hwnd))

    def is_foreground(self, hwnd: int) -> bool:
        return user32.GetForegroundWindow() == hwnd

    def focus(self, hwnd: int, timeout_s: float | None = None) -> bool:
        """Bring the window to the foreground, with the standard Alt-key workaround."""
        if not self.is_alive(hwnd):
            return False
        timeout = self._cfg.focus_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        if self.is_foreground(hwnd):
            return True

        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 0x0002, 0)
        while time.monotonic() < deadline:
            try:
                user32.SetForegroundWindow(hwnd)
            except Exception:
                pass
            if self.is_foreground(hwnd):
                return True
            time.sleep(0.05)
        return self.is_foreground(hwnd)


def make_focus_check(
    manager: GameWindowManager,
    focus_before_input: bool,
    focus_timeout_s: float,
) -> Callable[[], bool]:
    """Build the InputController focus guard: only allow sends when it returns True."""
    last_attempt = 0.0

    def _check() -> bool:
        nonlocal last_attempt
        info = manager.current()
        if info is None:
            return False
        if not focus_before_input:
            return True
        if info.focused:
            return True
        now = time.monotonic()
        if now - last_attempt < 1.0:
            return False
        last_attempt = now
        ok = manager.focus(info.hwnd, focus_timeout_s)
        info = manager.current(force=True)
        focused = ok or (info is not None and info.focused)
        if not focused:
            log.warning("input refused: '%s' is not focused", info.title if info else "?")
        return focused

    return _check
