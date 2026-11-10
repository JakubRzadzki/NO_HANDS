"""Draws the gesture pictures of the README (docs/gestures/*.png).

They are drawings, not photos: a simple flat hand whose fingers are placed the way each gesture
needs them, with arrows for the movement. Run ``python docs/make_gesture_images.py`` to redraw
them. To use your own photos, put files with the same names into docs/gestures/ and do not run
this script again.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]
Point = tuple[float, float]

OUT = Path(__file__).resolve().parent / "gestures"
SIZE = 640  # final picture, px
SS = 2  # drawn at this multiple, then scaled down: smooth edges

# BGR colours
BACKGROUND = (250, 247, 244)
FRAME = (226, 218, 210)
INK = (60, 52, 46)
MUTED = (140, 130, 122)
SKIN = (176, 205, 240)
SKIN_SHADE = (150, 183, 224)
SKIN_LINE = (84, 112, 150)
ACTIVE = (232, 140, 40)  # the fingers that make the gesture
MOVE = (60, 140, 235)  # arrows: movement
TOUCH = (90, 190, 60)  # where fingers touch

# Hand model, in palm lengths (wrist -> middle knuckle = 1), x to the thumb side, y up.
MCP = {"index": (0.28, 0.94), "middle": (0.06, 1.0), "ring": (-0.17, 0.94), "pinky": (-0.37, 0.83)}
LENGTHS = {
    "index": (0.50, 0.28, 0.24),
    "middle": (0.55, 0.33, 0.26),
    "ring": (0.52, 0.31, 0.25),
    "pinky": (0.39, 0.22, 0.22),
}
WIDTH = {"index": 0.185, "middle": 0.195, "ring": 0.185, "pinky": 0.155}  # full width
CMC = (0.30, 0.28)
THUMB = (0.62, 0.52)  # segment lengths


class Canvas:
    """Palm-length coordinates on a picture of SIZE px."""

    def __init__(self, scale: float = 0.25, origin: Point = (-0.33, 0.0)) -> None:
        n = SIZE * SS
        self.img: Image = np.full((n, n, 3), BACKGROUND, np.uint8)
        cv2.rectangle(self.img, (6, 6), (n - 7, n - 7), FRAME, 4 * SS, cv2.LINE_AA)
        self.s = scale * n
        self.ox = n / 2 + origin[0] * self.s
        self.oy = n * 0.77 + origin[1] * self.s

    def px(self, p: Point) -> tuple[int, int]:
        return round(self.ox + p[0] * self.s), round(self.oy - p[1] * self.s)

    def r(self, length: float) -> int:
        return max(1, round(length * self.s))


def fk(base: Point, lengths: tuple[float, ...], angles: tuple[float, ...]) -> list[Point]:
    """Joints of a finger. Angles are measured from straight up, positive towards the thumb."""
    pts = [base]
    x, y = base
    for length, a in zip(lengths, angles, strict=True):
        x += length * math.sin(math.radians(a))
        y += length * math.cos(math.radians(a))
        pts.append((x, y))
    return pts


def ik(base: Point, target: Point, l1: float, l2: float, elbow: float = -1.0) -> list[Point]:
    """Two-segment chain from ``base`` whose end reaches ``target`` (as near as it can)."""
    dx, dy = target[0] - base[0], target[1] - base[1]
    d = min(math.hypot(dx, dy), (l1 + l2) * 0.999)
    d = max(d, abs(l1 - l2) * 1.001)
    a = math.atan2(dy, dx)
    cos_k = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    k = math.acos(max(-1.0, min(1.0, cos_k)))
    joint = (base[0] + l1 * math.cos(a + elbow * k), base[1] + l1 * math.sin(a + elbow * k))
    end = (base[0] + d * math.cos(a), base[1] + d * math.sin(a))
    return [base, joint, end]


class Hand:
    """A flat hand: which finger does what is set with ``up``, ``curl``, ``bend`` and ``thumb``."""

    def __init__(self) -> None:
        self.chains: dict[str, list[Point]] = {}
        self.curled: set[str] = set()
        self.active: set[str] = set()
        self.thumb: list[Point] = []
        self.thumb_front = False  # the thumb lies over the fingers (fist)
        self.angle = 0.0  # the whole hand turned by this many degrees (counter-clockwise)

    def up(self, name: str, spread: float = 0.0) -> Hand:
        self.chains[name] = fk(MCP[name], LENGTHS[name], (spread, spread, spread))
        return self

    def curl(self, name: str) -> Hand:
        self.curled.add(name)
        self.chains[name] = fk(MCP[name], (LENGTHS[name][0] * 0.55,), (0,))
        return self

    def bend(self, name: str, angles: tuple[float, float, float]) -> Hand:
        self.chains[name] = fk(MCP[name], LENGTHS[name], angles)
        return self

    def thumb_out(self, a1: float = 62.0, a2: float = 40.0) -> Hand:
        self.thumb = fk(CMC, THUMB, (a1, a2))
        return self

    def thumb_tuck(self) -> Hand:
        self.thumb = [CMC, (0.16, 0.58), (-0.1, 0.7)]
        self.thumb_front = True
        return self

    def thumb_to(self, finger: str) -> Hand:
        self.thumb = ik(CMC, self.chains[finger][-1], *THUMB)
        return self

    def tip(self, finger: str) -> Point:
        return self.chains[finger][-1]

    # -- drawing -----------------------------------------------------------------------------
    def _rotated(self, p: Point) -> Point:
        a = math.radians(self.angle)
        return (p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a))

    def draw(self, c: Canvas) -> None:
        palm = [(-0.34, 0.0), (0.30, 0.0), (0.40, 0.3), (0.40, 0.92), (-0.46, 0.86), (-0.46, 0.3)]
        shapes: list[tuple[list[Point], float, tuple[int, int, int], bool]] = []
        for name, chain in self.chains.items():
            shapes.append(
                (
                    chain,
                    WIDTH[name],
                    SKIN_SHADE if name in self.curled else SKIN,
                    name in self.active,
                )
            )
        if self.thumb:
            shapes.append((self.thumb, 0.2, SKIN, "thumb" in self.active))
        back = [s for s in shapes if s[0] is not self.thumb or not self.thumb_front]
        front = [s for s in shapes if s[0] is self.thumb and self.thumb_front]
        poly = np.array([c.px(self._rotated(p)) for p in palm], np.int32)

        def stroke(
            chain: list[Point], width: float, colour: tuple[int, int, int], extra: int
        ) -> None:
            pts = [c.px(self._rotated(p)) for p in chain]
            t = c.r(width) + extra
            for a, b in itertools.pairwise(pts):
                cv2.line(c.img, a, b, colour, t, cv2.LINE_AA)
            for p in pts:
                cv2.circle(c.img, p, t // 2, colour, -1, cv2.LINE_AA)

        outline = 3 * SS
        cv2.fillPoly(c.img, [poly], SKIN_LINE, cv2.LINE_AA)
        cv2.polylines(c.img, [poly], True, SKIN_LINE, c.r(0.16), cv2.LINE_AA)
        cv2.fillPoly(c.img, [poly], SKIN, cv2.LINE_AA)
        cv2.polylines(c.img, [poly], True, SKIN, c.r(0.13), cv2.LINE_AA)
        for layer in (back, front):
            for chain, width, colour, active in layer:
                stroke(chain, width, ACTIVE if active else SKIN_LINE, outline * 2)
                stroke(chain, width, colour, 0)
                if colour == SKIN_SHADE:  # a folded finger: show where it folds
                    x, y = c.px(self._rotated(chain[-1]))
                    cv2.circle(c.img, (x, y), c.r(width) // 3, SKIN_LINE, 2 * SS, cv2.LINE_AA)


def text(
    c: Canvas, s: str, y: float, size: float, colour: tuple[int, int, int], bold: bool
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = size * SS
    thickness = round((2.4 if bold else 1.2) * SS)
    (w, _), _ = cv2.getTextSize(s, font, scale, thickness)
    cv2.putText(
        c.img, s, ((SIZE * SS - w) // 2, round(y * SS)), font, scale, colour, thickness, cv2.LINE_AA
    )


def label(c: Canvas, s: str, p: Point, colour: tuple[int, int, int] = MOVE) -> None:
    font, scale, th = cv2.FONT_HERSHEY_DUPLEX, 0.9 * SS, 2 * SS
    (w, h), _ = cv2.getTextSize(s, font, scale, th)
    x, y = c.px(p)
    cv2.putText(c.img, s, (x - w // 2, y + h // 2), font, scale, colour, th, cv2.LINE_AA)


def caption(c: Canvas, title: str, *lines: str) -> None:
    text(c, title, 62, 1.25, INK, True)
    for i, line in enumerate(lines):
        text(c, line, SIZE - 62 + 34 * (i - len(lines) + 1) + 20, 0.78, MUTED, False)


def arrow(c: Canvas, a: Point, b: Point, colour: tuple[int, int, int] = MOVE, w: float = 7) -> None:
    cv2.arrowedLine(c.img, c.px(a), c.px(b), colour, round(w * SS), cv2.LINE_AA, tipLength=0.28)


def arc(
    c: Canvas, centre: Point, radius: float, start: float, end: float, flip: bool = False
) -> None:
    """A curved arrow around ``centre`` (angles in degrees, counter-clockwise from +x)."""
    pts = []
    n = 24
    for i in range(n + 1):
        a = math.radians(start + (end - start) * i / n)
        pts.append(c.px((centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a))))
    cv2.polylines(c.img, [np.array(pts[:-1], np.int32)], False, MOVE, 7 * SS, cv2.LINE_AA)
    cv2.arrowedLine(c.img, pts[-4], pts[-1], MOVE, 7 * SS, cv2.LINE_AA, tipLength=1.2)


def touch(c: Canvas, p: Point) -> None:
    x, y = c.px(p)
    cv2.circle(c.img, (x, y), c.r(0.13), TOUCH, 5 * SS, cv2.LINE_AA)


def badge(c: Canvas, s: str, centre: Point) -> None:
    x, y = c.px(centre)
    cv2.circle(c.img, (x, y), 40 * SS, MOVE, -1, cv2.LINE_AA)
    font, scale, th = cv2.FONT_HERSHEY_DUPLEX, 1.3 * SS, 3 * SS
    (w, h), _ = cv2.getTextSize(s, font, scale, th)
    cv2.putText(c.img, s, (x - w // 2, y + h // 2), font, scale, (255, 255, 255), th, cv2.LINE_AA)


def save(c: Canvas, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = cv2.resize(c.img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f"{name}.png"), img)


# -- the hands ---------------------------------------------------------------------------------
def open_hand(spread: bool = True) -> Hand:
    h = Hand()
    for i, name in enumerate(("index", "middle", "ring", "pinky")):
        h.up(name, (5, 1, -3, -8)[i] if spread else 0)
    return h.thumb_out()


def pinch_hand(finger: str = "index", others_up: bool = True) -> Hand:
    h = Hand()
    for i, name in enumerate(("index", "middle", "ring", "pinky")):
        if name == finger:
            continue
        h.up(name, (5, 1, -3, -8)[i] if others_up else 0)
    angles = {"index": (38.0, 70.0, 100.0), "middle": (30.0, 62.0, 92.0)}[finger]
    h.bend(finger, angles)
    h.thumb_to(finger)
    h.active |= {finger, "thumb"}
    return h


def fist() -> Hand:
    h = Hand()
    for name in ("index", "middle", "ring", "pinky"):
        h.curl(name)
    return h.thumb_tuck()


def build() -> dict[str, Any]:
    pics: dict[str, Any] = {}

    def move() -> Canvas:
        c = Canvas()
        h = open_hand()
        h.active |= {"index", "middle", "ring", "pinky"}
        h.draw(c)
        arrow(c, (1.25, 1.55), (1.7, 1.55))
        arrow(c, (-0.65, 1.3), (-1.1, 1.3))
        arrow(c, (0.0, 2.05), (0.0, 2.4))
        caption(c, "Move the cursor", "open hand, fingers apart", "the cursor follows the knuckles")
        return c

    def left_button() -> Canvas:
        c = Canvas()
        h = pinch_hand()
        h.draw(c)
        touch(c, h.tip("index"))
        caption(c, "Left button", "thumb + index touching = button down", "let go = click")
        return c

    def drag() -> Canvas:
        c = Canvas()
        h = pinch_hand()
        h.draw(c)
        touch(c, h.tip("index"))
        for a, b in (
            ((1.35, 1.6), (1.75, 1.6)),
            ((1.35, 1.6), (0.95, 1.6)),
            ((1.35, 1.6), (1.35, 2.0)),
            ((1.35, 1.6), (1.35, 1.2)),
        ):
            arrow(c, a, b)
        caption(c, "Drag and select", "keep touching and move the hand", "let go = drop")
        return c

    def double() -> Canvas:
        c = Canvas()
        h = pinch_hand()
        h.draw(c)
        touch(c, h.tip("index"))
        badge(c, "x2", (1.25, 1.55))
        caption(c, "Double click", "pinch, let go, pinch again", "within about 0.4 s")
        return c

    def right_click() -> Canvas:
        c = Canvas()
        h = pinch_hand("middle")
        h.draw(c)
        touch(c, h.tip("middle"))
        caption(c, "Right click", "thumb + middle finger touching", "one click per pinch")
        return c

    def scroll() -> Canvas:
        c = Canvas()
        h = Hand().up("index", -1).up("middle", 1).curl("ring").curl("pinky").thumb_tuck()
        h.active |= {"index", "middle"}
        h.draw(c)
        arrow(c, (1.05, 1.5), (1.05, 2.05))
        arrow(c, (1.05, 1.3), (1.05, 0.75))
        caption(
            c,
            "Scroll",
            "index + middle together, ring + pinky folded",
            "move up / down: further = faster",
        )
        return c

    def zoom() -> Canvas:
        c = Canvas()
        h = Hand().up("index", 6).curl("middle").curl("ring").up("pinky", -8).thumb_out(75, 55)
        h.active |= {"index", "pinky"}
        h.draw(c)
        arrow(c, (1.5, 1.5), (1.5, 2.0))
        arrow(c, (1.5, 1.3), (1.5, 0.8))
        label(c, "in", (1.5, 2.2))
        label(c, "out", (1.5, 0.5))
        caption(c, "Zoom", "horns: index + pinky up", "move up = zoom in, down = zoom out")
        return c

    def clutch() -> Canvas:
        c = Canvas()
        h = fist()
        h.active |= {"index", "middle", "ring", "pinky"}
        h.draw(c)
        caption(c, "Pause (clutch)", "fist: the cursor does not move", "open the hand to go on")
        return c

    def flick() -> Canvas:
        c = Canvas(origin=(-0.33, 0.28))
        h = open_hand(False)
        h.active |= {"index", "middle", "ring", "pinky"}
        h.draw(c)
        arrow(c, (-0.4, 2.3), (1.2, 2.3), w=5)
        arrow(c, (1.2, 2.58), (-0.4, 2.58), w=5)
        caption(
            c,
            "Flick = previous / next slide",
            "open hand: quick sideways and straight back",
            "to the right = next, to the left = previous",
        )
        return c

    def swipe() -> Canvas:
        c = Canvas(origin=(-0.33, 0.28))
        h = open_hand(False)
        h.active |= {"index", "middle", "ring", "pinky"}
        h.draw(c)
        arrow(c, (-0.4, 2.3), (1.2, 2.3), w=5)
        arrow(c, (1.2, 2.58), (-0.4, 2.58), w=5)
        caption(c, "Swipe", "open hand: quick sideways move", "left / right")
        return c

    def volume() -> Canvas:
        c = Canvas(origin=(-0.2, 0.12))
        h = open_hand(False)
        h.angle = -25.0
        h.active |= {"index", "middle", "ring", "pinky"}
        h.draw(c)
        arc(c, (0.0, 0.9), 1.45, 105, 45)
        arc(c, (0.0, 0.9), 1.45, 75, 135)
        label(c, "louder", (1.15, 2.45))
        label(c, "quieter", (-1.05, 2.45))
        caption(c, "Volume", "turn the open hand like a knob", "clockwise = louder")
        return c

    def play_pause() -> Canvas:
        c = Canvas()
        h = fist()
        h.active |= {"index", "middle", "ring", "pinky"}
        h.draw(c)
        caption(c, "Play / pause", "close the hand into a fist", "once per closing")
        return c

    def shaka() -> Canvas:
        c = Canvas()
        h = Hand().curl("index").curl("middle").curl("ring").up("pinky", -10).thumb_out(75, 60)
        h.active |= {"pinky", "thumb"}
        h.draw(c)
        caption(c, "Mouse <-> media mode", "shaka: thumb + pinky out", "hold for 0.6 s")
        return c

    def slide_pinch() -> Canvas:
        c = Canvas()
        h = pinch_hand()
        h.draw(c)
        touch(c, h.tip("index"))
        caption(c, "Next slide", "quick pinch thumb + index", "like a mouse click")
        return c

    def slide_zoom() -> Canvas:
        c = Canvas()
        h = pinch_hand()
        h.draw(c)
        touch(c, h.tip("index"))
        arrow(c, (1.5, 1.6), (1.5, 2.1))
        arrow(c, (1.5, 1.4), (1.5, 0.9))
        caption(c, "Zoom in / out", "pinch thumb + index, hold,", "move up = in, down = out")
        return c

    pics.update(
        hand_move=move,
        hand_left_button=left_button,
        hand_drag=drag,
        hand_double_click=double,
        hand_right_click=right_click,
        hand_scroll=scroll,
        hand_zoom=zoom,
        hand_clutch=clutch,
        hand_flick=flick,
        hand_swipe=swipe,
        hand_volume=volume,
        hand_play_pause=play_pause,
        hand_shaka=shaka,
        slides_pinch=slide_pinch,
        slides_zoom=slide_zoom,
    )
    return pics


# -- faces -------------------------------------------------------------------------------------
def face(
    c: Canvas, yaw: float = 0.0, pitch: float = 0.0, left: str = "open", right: str = "open"
) -> None:
    """A schematic face as the mirrored preview shows it: your left eye is on the left."""
    n = SIZE * SS
    cx, cy = n / 2 + yaw * 0.9 * SS, n * 0.5 + pitch * 0.9 * SS
    cv2.ellipse(
        c.img,
        (round(cx), round(cy)),
        (round(n * 0.2), round(n * 0.27)),
        0,
        0,
        360,
        SKIN,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        c.img,
        (round(cx), round(cy)),
        (round(n * 0.2), round(n * 0.27)),
        0,
        0,
        360,
        SKIN_LINE,
        4 * SS,
        cv2.LINE_AA,
    )
    for side, state in ((-1, left), (1, right)):
        ex, ey = round(cx + side * n * 0.085 + yaw * 0.3 * SS), round(cy - n * 0.04)
        if state == "open":
            cv2.ellipse(
                c.img,
                (ex, ey),
                (round(n * 0.04), round(n * 0.022)),
                0,
                0,
                360,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(c.img, (ex, ey), round(n * 0.014), INK, -1, cv2.LINE_AA)
            cv2.ellipse(
                c.img,
                (ex, ey),
                (round(n * 0.04), round(n * 0.022)),
                0,
                0,
                360,
                INK,
                3 * SS,
                cv2.LINE_AA,
            )
        else:
            cv2.ellipse(
                c.img,
                (ex, ey),
                (round(n * 0.04), round(n * 0.02)),
                0,
                10,
                170,
                INK,
                5 * SS,
                cv2.LINE_AA,
            )
    cv2.line(
        c.img,
        (round(cx), round(cy - n * 0.005)),
        (round(cx - 6 * SS), round(cy + n * 0.06)),
        SKIN_LINE,
        4 * SS,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        c.img,
        (round(cx), round(cy + n * 0.13)),
        (round(n * 0.05), round(n * 0.022)),
        0,
        10,
        170,
        INK,
        4 * SS,
        cv2.LINE_AA,
    )


def build_faces() -> dict[str, Any]:
    pics: dict[str, Any] = {}

    def make(title: str, line1: str, line2: str, **kw: Any) -> Any:
        def draw() -> Canvas:
            c = Canvas()
            arrows = kw.pop("arrows", None)
            face(c, **kw)
            for a, b in arrows or []:
                cv2.arrowedLine(c.img, a, b, MOVE, 8 * SS, cv2.LINE_AA, tipLength=0.28)
            caption(c, title, line1, line2)
            return c

        return draw

    n = SIZE * SS
    pics["head_turn"] = make(
        "Left / Right arrow",
        "a quick turn of the head",
        "to the left / right",
        arrows=[
            ((round(n * 0.33), round(n * 0.2)), (round(n * 0.12), round(n * 0.2))),
            ((round(n * 0.67), round(n * 0.2)), (round(n * 0.88), round(n * 0.2))),
        ],
    )
    pics["head_nod"] = make(
        "Up / Down arrow",
        "a quick tilt up or nod down",
        "about 12 degrees",
        arrows=[
            ((round(n * 0.9), round(n * 0.45)), (round(n * 0.9), round(n * 0.2))),
            ((round(n * 0.9), round(n * 0.55)), (round(n * 0.9), round(n * 0.8))),
        ],
    )
    pics["eyes_blink"] = make(
        "Left click",
        "a deliberate blink with both eyes",
        "0.35 - 1 s; twice = double click",
        left="closed",
        right="closed",
    )
    pics["eyes_wink_right"] = make(
        "Right click", "wink with the right eye", "at least 0.3 s", right="closed"
    )
    pics["eyes_wink_left"] = make(
        "Scroll mode on / off",
        "wink with the left eye",
        "then look above / below the middle",
        left="closed",
    )
    pics["eyes_close"] = make(
        "Pause / resume", "close both eyes", "for 1.5 s", left="closed", right="closed"
    )
    return pics


def main(argv: list[str]) -> int:
    pictures = {**build(), **build_faces()}
    wanted = set(argv) or set(pictures)
    for name, draw in pictures.items():
        if name in wanted:
            save(draw(), name)
            print("drew", OUT / f"{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
