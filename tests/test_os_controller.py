from __future__ import annotations

import sys

import pytest

from src.os_controller import (
    Click,
    MoveBy,
    MoveTo,
    OSController,
    Press,
    Release,
    Scroll,
    Zoom,
)


class RecordingBackend:
    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.fail_on = fail_on or set()

    def _record(self, *call: object) -> None:
        if call[0] in self.fail_on:
            raise RuntimeError(f"{call[0]} failed")
        self.calls.append(call)

    def move_to(self, x: int, y: int) -> None:
        self._record("move_to", x, y)

    def move_by(self, dx: int, dy: int) -> None:
        self._record("move_by", dx, dy)

    def press(self, button: str) -> None:
        self._record("press", button)

    def release(self, button: str) -> None:
        self._record("release", button)

    def click(self, button: str, count: int) -> None:
        self._record("click", button, count)

    def scroll(self, dx: int, dy: int) -> None:
        self._record("scroll", dx, dy)

    def key_down(self, key: str) -> None:
        self._record("key_down", key)

    def key_up(self, key: str) -> None:
        self._record("key_up", key)


def test_actions_are_dispatched_in_order() -> None:
    backend = RecordingBackend()
    ctl = OSController(backend)
    ctl.apply([MoveTo(1, 2), MoveBy(3, 4), Click("right"), Click("left", 2), Scroll(0, -3)])
    assert backend.calls == [
        ("move_to", 1, 2),
        ("move_by", 3, 4),
        ("click", "right", 1),
        ("click", "left", 2),
        ("scroll", 0, -3),
    ]


def test_press_is_tracked_idempotent_and_released_by_release_all() -> None:
    backend = RecordingBackend()
    ctl = OSController(backend)
    ctl.apply([Press("left"), Press("left"), Release("right")])
    assert backend.calls == [("press", "left")]  # no double press, no unpaired release
    assert ctl.held_buttons == {"left"}
    ctl.release_all()
    ctl.release_all()  # idempotent
    assert backend.calls == [("press", "left"), ("release", "left")]
    assert ctl.held_buttons == frozenset()


def test_zoom_wraps_scroll_in_modifier_even_when_scroll_fails() -> None:
    backend = RecordingBackend(fail_on={"scroll"})
    ctl = OSController(backend, zoom_modifier="ctrl")
    ctl.apply([Zoom(2)])
    assert backend.calls == [("key_down", "ctrl"), ("key_up", "ctrl")]
    assert ctl.held_keys == frozenset()
    assert ctl.last_error is not None


def test_a_failing_action_does_not_stop_the_following_ones() -> None:
    backend = RecordingBackend(fail_on={"click"})
    ctl = OSController(backend)
    ctl.apply([Click("left"), Press("left")])
    assert backend.calls == [("press", "left")]


def test_dry_run_never_imports_pynput(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynput", None)  # any import of pynput now fails
    ctl = OSController.create(dry_run=True)
    ctl.apply([MoveTo(0, 0), Press("left"), Zoom(1)])
    ctl.release_all()
