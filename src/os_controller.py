"""OS input injection, plus the action types the gesture engine emits.

The gesture engine never calls pynput directly; it returns these small action objects and this
module executes them. Two payoffs: the engine is testable without a display, and this is the one
place that knows what is currently held down, so ``release_all()`` can guarantee that no button
or modifier key stays stuck when the program exits or crashes.

pynput is used instead of pyautogui: pyautogui sleeps 0.1 s after every call by default
(``pyautogui.PAUSE``), which alone caps a loop at ~10 FPS unless you remember to disable it.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)


# -- actions ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class MoveTo:
    x: int
    y: int


@dataclass(frozen=True)
class MoveBy:
    dx: int
    dy: int


@dataclass(frozen=True)
class Click:
    button: str = "left"
    count: int = 1


@dataclass(frozen=True)
class Press:
    button: str = "left"


@dataclass(frozen=True)
class Release:
    button: str = "left"


@dataclass(frozen=True)
class Scroll:
    dx: int = 0  # wheel ticks, + = right
    dy: int = 0  # wheel ticks, + = up


@dataclass(frozen=True)
class Zoom:
    steps: int  # + = zoom in (modifier + wheel up), - = zoom out


Action = MoveTo | MoveBy | Click | Press | Release | Scroll | Zoom


# -- backends --------------------------------------------------------------------------------
class InputBackend(Protocol):
    def move_to(self, x: int, y: int) -> None: ...
    def move_by(self, dx: int, dy: int) -> None: ...
    def press(self, button: str) -> None: ...
    def release(self, button: str) -> None: ...
    def click(self, button: str, count: int) -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...


class PynputBackend:
    """Real input: SendInput on Windows, Quartz events on macOS, XTest on X11."""

    def __init__(self) -> None:
        from pynput import keyboard, mouse  # lazy: on Linux this needs a running X server

        self._mouse: Any = mouse.Controller()
        self._keyboard: Any = keyboard.Controller()
        self._buttons = {
            "left": mouse.Button.left,
            "right": mouse.Button.right,
            "middle": mouse.Button.middle,
        }
        self._keys = {"ctrl": keyboard.Key.ctrl, "cmd": keyboard.Key.cmd}

    def move_to(self, x: int, y: int) -> None:
        self._mouse.position = (x, y)

    def move_by(self, dx: int, dy: int) -> None:
        self._mouse.move(dx, dy)

    def press(self, button: str) -> None:
        self._mouse.press(self._buttons[button])

    def release(self, button: str) -> None:
        self._mouse.release(self._buttons[button])

    def click(self, button: str, count: int) -> None:
        self._mouse.click(self._buttons[button], count)

    def scroll(self, dx: int, dy: int) -> None:
        self._mouse.scroll(dx, dy)  # integer ticks: X11 emulates the wheel with button clicks

    def key_down(self, key: str) -> None:
        self._keyboard.press(self._keys[key])

    def key_up(self, key: str) -> None:
        self._keyboard.release(self._keys[key])


class DryRunBackend:
    """Logs instead of injecting: tune gestures without the cursor running away from you."""

    def move_to(self, x: int, y: int) -> None:
        log.debug("[dry-run] move to (%d, %d)", x, y)

    def move_by(self, dx: int, dy: int) -> None:
        log.debug("[dry-run] move by (%d, %d)", dx, dy)

    def press(self, button: str) -> None:
        log.info("[dry-run] press %s", button)

    def release(self, button: str) -> None:
        log.info("[dry-run] release %s", button)

    def click(self, button: str, count: int) -> None:
        log.info("[dry-run] click %s x%d", button, count)

    def scroll(self, dx: int, dy: int) -> None:
        log.info("[dry-run] scroll dx=%d dy=%d", dx, dy)

    def key_down(self, key: str) -> None:
        log.info("[dry-run] key down %s", key)

    def key_up(self, key: str) -> None:
        log.info("[dry-run] key up %s", key)


# -- controller ------------------------------------------------------------------------------
class OSController:
    def __init__(self, backend: InputBackend, zoom_modifier: str = "ctrl") -> None:
        self._backend = backend
        self._zoom_modifier = zoom_modifier
        self._held_buttons: set[str] = set()
        self._held_keys: set[str] = set()
        self.last_error: str | None = None

    @classmethod
    def create(cls, dry_run: bool = False, zoom_modifier: str = "ctrl") -> OSController:
        return cls(DryRunBackend() if dry_run else PynputBackend(), zoom_modifier)

    @property
    def held_buttons(self) -> frozenset[str]:
        return frozenset(self._held_buttons)

    @property
    def held_keys(self) -> frozenset[str]:
        return frozenset(self._held_keys)

    def apply(self, actions: Iterable[Action]) -> None:
        for action in actions:
            try:
                self._dispatch(action)
            except Exception as exc:
                # One failed injection must not kill the loop (and with it the safety release).
                self.last_error = f"{type(action).__name__}: {exc}"
                log.warning("Input action %r failed: %s", action, exc)

    def release_all(self) -> None:
        """Release everything we hold. Idempotent; meant for ``finally`` blocks."""
        for button in sorted(self._held_buttons):
            try:
                self._backend.release(button)
            except Exception as exc:
                log.error("Could not release mouse button %s: %s", button, exc)
        self._held_buttons.clear()
        for key in sorted(self._held_keys):
            try:
                self._backend.key_up(key)
            except Exception as exc:
                log.error("Could not release key %s: %s", key, exc)
        self._held_keys.clear()

    def _dispatch(self, action: Action) -> None:
        b = self._backend
        match action:
            case MoveTo(x, y):
                b.move_to(x, y)
            case MoveBy(dx, dy):
                b.move_by(dx, dy)
            case Click(button, count):
                b.click(button, count)
            case Press(button):
                if button not in self._held_buttons:  # never double-press
                    b.press(button)
                    self._held_buttons.add(button)
            case Release(button):
                if button in self._held_buttons:  # never release what we did not press
                    b.release(button)
                    self._held_buttons.discard(button)
            case Scroll(dx, dy):
                if dx or dy:
                    b.scroll(dx, dy)
            case Zoom(steps):
                if steps:
                    self._zoom(steps)

    def _zoom(self, steps: int) -> None:
        key = self._zoom_modifier
        self._backend.key_down(key)
        self._held_keys.add(key)
        try:
            self._backend.scroll(0, steps)
        finally:
            self._backend.key_up(key)
            self._held_keys.discard(key)


# -- screen geometry -------------------------------------------------------------------------
def enable_dpi_awareness() -> None:
    """Windows: make the process DPI-aware so screen size and cursor positions are both in
    physical pixels. Must run before the first coordinate query, otherwise a library that flips
    awareness later makes the measured screen size and the cursor coordinates disagree."""
    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
        except (AttributeError, OSError):
            with contextlib.suppress(AttributeError, OSError):
                ctypes.windll.user32.SetProcessDPIAware()  # Windows 7 fallback


def detect_screen_size() -> tuple[int, int] | None:
    """Size of the primary monitor in the coordinate system pynput uses, or None if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes

            enable_dpi_awareness()
            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        if sys.platform == "darwin":
            import Quartz  # installed together with pynput on macOS

            bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            return int(bounds.size.width), int(bounds.size.height)
        from Xlib.display import Display  # installed together with pynput on Linux

        display = Display()
        try:
            screen = display.screen()
            return int(screen.width_in_pixels), int(screen.height_in_pixels)
        finally:
            display.close()
    except Exception as exc:
        log.debug("Native screen size detection failed: %s", exc)
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        try:
            return int(root.winfo_screenwidth()), int(root.winfo_screenheight())
        finally:
            root.destroy()
    except Exception as exc:
        log.debug("tkinter screen size detection failed: %s", exc)
    return None
