"""Action types the gesture engine emits and the OS controller executes."""

from __future__ import annotations

from dataclasses import dataclass


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
