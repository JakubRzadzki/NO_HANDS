"""Pinch primitives shared by the hand mouse, --slides and the two-hand zoom: a hysteresis
latch and the thumb-to-fingertip distance."""

from __future__ import annotations

import math

from src.hand_tracker import HandObservation


class HysteresisLatch:
    """Boolean latch with separate engage/release thresholds plus frame debouncing.

    low_active=True:  engages when value < engage_at, releases when value > release_at.
    low_active=False: engages when value > engage_at, releases when value < release_at.

    Two thresholds stop a signal that hovers around one boundary from toggling every frame (the
    classic "machine-gun click"); ``confirm_frames`` additionally ignores one-frame outliers.
    ``release_frames`` (default: the same) can make letting go slower than engaging: a held
    mouse button must survive a couple of bad frames while the hand moves.
    """

    def __init__(
        self,
        engage_at: float,
        release_at: float,
        low_active: bool = True,
        confirm_frames: int = 1,
        release_frames: int | None = None,
    ) -> None:
        if (low_active and engage_at >= release_at) or (not low_active and engage_at <= release_at):
            raise ValueError("the release threshold must lie on the inactive side of engage_at")
        self.engage_at = engage_at
        self.release_at = release_at
        self.low_active = low_active
        self.confirm_frames = max(1, confirm_frames)
        self.release_frames = self.confirm_frames if release_frames is None else release_frames
        self.active = False
        self._streak = 0

    def update(self, value: float) -> bool:
        if self.active:
            flip = value > self.release_at if self.low_active else value < self.release_at
        else:
            flip = value < self.engage_at if self.low_active else value > self.engage_at
        if flip:
            self._streak += 1
            if self._streak >= (self.release_frames if self.active else self.confirm_frames):
                self.active = not self.active
                self._streak = 0
        else:
            self._streak = 0
        return self.active

    def reset(self) -> None:
        self.active = False
        self._streak = 0


def pinch_distance(obs: HandObservation, a: int, b: int, depth_weight: float) -> float:
    """Distance between two fingertips in palm lengths, depth counted at ``depth_weight``.

    MediaPipe's depth (z) is its least reliable output, and its noise is as large as a pinch is
    small. On real photos of touching fingers the image distance was 0.09-0.18 palm lengths
    while the full 3D distance read 0.13-0.26. Some depth is still worth having: seen from
    the side, fingers that overlap in the image can be far apart."""
    d = obs.px[a] - obs.px[b]
    return float(math.sqrt(d[0] ** 2 + d[1] ** 2 + (depth_weight * d[2]) ** 2)) / obs.palm_scale
