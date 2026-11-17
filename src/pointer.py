"""Hand anchor -> cursor: Active ROI, edge overshoot, 1€ smoothing, absolute/relative modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import FilterConfig
from src.os_controller import MoveBy, MoveTo
from src.smoother import FloatArray, OneEuroFilter


@dataclass(frozen=True)
class Roi:
    """Active region of the camera frame (normalized coordinates) that maps onto the screen.

    Using a box smaller than the frame means the hand never has to reach the image border
    (where MediaPipe loses fingers that leave the frame), and moving past the box simply pins
    the cursor to the screen edge.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def for_screen(
        cls,
        frame_size: tuple[int, int],
        screen_size: tuple[int, int],
        width: float = 0.6,
        center: tuple[float, float] = (0.5, 0.55),
        sensitivity: float = 1.0,
        max_extent: float = 0.96,
    ) -> Roi:
        """Build an ROI whose pixel aspect ratio matches the screen's.

        Then a centimetre of hand travel moves the cursor equally far horizontally and
        vertically. A naive 0.6 x 0.6 box on a 4:3 camera mapped to a 16:9 screen makes
        horizontal movement 33% more sensitive than vertical.
        """
        frame_w, frame_h = frame_size
        screen_w, screen_h = screen_size
        w = width / sensitivity
        h = w * (frame_w / frame_h) / (screen_w / screen_h)
        shrink = min(1.0, max_extent / w, max_extent / h)  # keep it inside the frame
        w, h = w * shrink, h * shrink
        cx = min(max(center[0], w / 2), 1 - w / 2)
        cy = min(max(center[1], h / 2), 1 - h / 2)
        return cls(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def to_unit(self, x: float, y: float, clamp: bool = True) -> tuple[float, float]:
        """Frame coordinates -> position inside the box (0..1 when inside)."""
        u = (x - self.x0) / (self.x1 - self.x0)
        v = (y - self.y0) / (self.y1 - self.y0)
        if clamp:
            u, v = min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)
        return u, v

    def to_pixels(self, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
        w, h = frame_size
        return round(self.x0 * w), round(self.y0 * h), round(self.x1 * w), round(self.y1 * h)


class CursorMapper:
    """Turns the hand anchor into cursor commands.

    absolute: the ROI is stretched over the screen. The ROI edge maps slightly *beyond* the
              screen edge (``edge_overshoot``), so the smoothed cursor crosses the edge quickly
              and gets clamped there. Without it, an exponential filter approaches the last
              pixel asymptotically, which makes corner targets (close buttons, start menu)
              frustratingly slow to hit.
    relative: like a real mouse, only *changes* of the filtered position move the cursor.
              While frozen (clutch, scroll, pinch) deltas are discarded, so the hand can be
              repositioned in the air like lifting a mouse off its pad.

    ``speed`` below 1 slows the cursor down (precision). In absolute mode the cursor then moves
    only part of the way the hand moves, which leaves it off the spot the hand points at. Back
    at full speed that gap closes by ``catch_up`` times the hand's travel (0.5: the cursor
    moves up to 1.5x / 0.5x as fast as the hand until it is back), so it never jumps.
    """

    def __init__(
        self,
        roi: Roi,
        screen_size: tuple[int, int],
        filter_cfg: FilterConfig,
        mode: str = "absolute",
        edge_overshoot: float = 0.03,
        relative_gain: float = 1.0,
        catch_up: float = 0.5,
    ) -> None:
        if mode not in ("absolute", "relative"):
            raise ValueError(f"unknown pointer mode {mode!r}")
        self.roi = roi
        self.screen_w, self.screen_h = screen_size
        self.mode = mode
        self.edge_overshoot = edge_overshoot
        self.relative_gain = relative_gain
        self._filter = OneEuroFilter(filter_cfg.min_cutoff, filter_cfg.beta, filter_cfg.d_cutoff)
        self._target: FloatArray | None = None  # filtered position, screen px, may be off-screen
        self._last_pos: tuple[int, int] | None = None  # absolute: last position sent
        self._residual = np.zeros(2)  # relative: sub-pixel motion carried to the next frame
        self._was_frozen = False
        self.catch_up = catch_up
        self._offset = np.zeros(2)  # absolute: cursor - hand target, built up by slow movement

    @property
    def target(self) -> FloatArray | None:
        """Latest cursor target in screen pixels (None before the first update)."""
        return None if self._target is None else self._target + self._offset

    def reset(self) -> None:
        """Call when tracking is lost: the next hand position starts a fresh trajectory."""
        self._filter.reset()
        self._target = None
        self._last_pos = None
        self._residual[:] = 0.0
        self._was_frozen = False
        self._offset[:] = 0.0

    def update(
        self, anchor: tuple[float, float], t: float, frozen: bool = False, speed: float = 1.0
    ) -> MoveTo | MoveBy | None:
        """Feed the anchor (normalized frame coords). Always updates the filter so it stays
        converged, but emits nothing while ``frozen``. ``speed`` < 1 = precision."""
        raw = self._raw_target(anchor)
        if self.mode == "absolute":
            previous = self._target
            self._target = self._filter(raw, t)
            if previous is not None:
                self._absorb(self._target - previous, speed)
            if frozen:
                return None
            cursor = self._target + self._offset
            x = round(min(max(float(cursor[0]), 0.0), self.screen_w - 1.0))
            y = round(min(max(float(cursor[1]), 0.0), self.screen_h - 1.0))
            if (x, y) == self._last_pos:
                return None  # also stops us fighting a physical mouse while the hand is still
            self._last_pos = (x, y)
            return MoveTo(x, y)

        # Relative mode. While frozen the filter still trails the moving hand; on unfreezing,
        # that lag would "leak out" as cursor motion nobody made. Restart from the hand instead.
        unfreezing = self._was_frozen and not frozen
        self._was_frozen = frozen
        if unfreezing:
            self._filter.reset()
        previous = None if unfreezing else self._target
        self._target = self._filter(raw, t)
        if frozen or previous is None:
            self._residual[:] = 0.0
            return None
        # Keep the fractional part: slow movements of 0.3 px/frame must still add up.
        self._residual += (self._target - previous) * self.relative_gain * speed
        step = np.trunc(self._residual)
        self._residual -= step
        dx, dy = int(step[0]), int(step[1])
        return MoveBy(dx, dy) if (dx or dy) else None

    def hold_at(self, position: FloatArray | None) -> None:
        """Absolute mode: carry on from ``position`` (where the cursor was held) instead of
        jumping to where the hand points now. The gap closes as the hand moves (catch_up)."""
        if self.mode != "absolute" or self._target is None or position is None:
            return
        self._offset = np.asarray(position, dtype=np.float64) - self._target

    def _absorb(self, delta: FloatArray, speed: float) -> None:
        """Absolute mode: the cursor moves ``speed`` times as far as the hand target."""
        if speed < 1.0:
            self._offset -= (1.0 - speed) * delta
            # The cursor never leaves the screen, so neither does the offset it may build up.
            limit = np.array([self.screen_w, self.screen_h], dtype=np.float64)
            self._offset = np.clip(self._offset, -limit, limit)
        elif self._offset.any():
            # The gap closes by catch_up x the hand's travel: the cursor moves a little faster
            # or slower than the hand until it is back where the hand points. Never a jump.
            gap = float(np.linalg.norm(self._offset))
            closed = self.catch_up * float(np.linalg.norm(delta))
            self._offset *= max(0.0, gap - closed) / gap

    def _raw_target(self, anchor: tuple[float, float]) -> FloatArray:
        if self.mode == "absolute":
            u, v = self.roi.to_unit(*anchor, clamp=True)
            # Clamp *before* filtering so overshooting the box never builds up a backlog the
            # hand has to "unwind" before the cursor leaves the edge again.
            o = self.edge_overshoot
            u, v = u * (1 + 2 * o) - o, v * (1 + 2 * o) - o
        else:
            u, v = self.roi.to_unit(*anchor, clamp=False)  # no dead zone outside the box
        return np.array([u * (self.screen_w - 1), v * (self.screen_h - 1)])
