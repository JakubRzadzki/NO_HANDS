"""Gesture recognition building blocks: modes, hand features and hysteresis latches."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.hand_tracker import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    HandObservation,
)
from src.os_controller import Action


class Mode(str, Enum):
    NO_HAND = "NO HAND"
    POINTER = "POINTER"
    SCROLL = "SCROLL"
    ZOOM = "ZOOM"
    CLUTCH = "CLUTCH"


class LeftState(str, Enum):
    IDLE = "IDLE"
    PENDING = "PENDING"  # pinched, not yet decided between click and drag
    DRAG = "DRAG"  # button is held down


class Finger(Enum):
    INDEX = (INDEX_PIP, INDEX_TIP)
    MIDDLE = (MIDDLE_PIP, MIDDLE_TIP)
    RING = (RING_PIP, RING_TIP)
    PINKY = (PINKY_PIP, PINKY_TIP)

    @property
    def pip(self) -> int:
        return self.value[0]

    @property
    def tip(self) -> int:
        return self.value[1]


# Thumb pinches: INDEX = left click / drag, MIDDLE = right click, RING = double click.
PINCH_FINGERS = (Finger.INDEX, Finger.MIDDLE, Finger.RING)


class HysteresisLatch:
    """Boolean latch with separate engage/release thresholds plus frame debouncing.

    low_active=True:  engages when value < engage_at, releases when value > release_at.
    low_active=False: engages when value > engage_at, releases when value < release_at.

    Two thresholds stop a signal that hovers around one boundary from toggling every frame (the
    classic "machine-gun click"); ``confirm_frames`` additionally ignores one-frame outliers.
    """

    def __init__(
        self,
        engage_at: float,
        release_at: float,
        low_active: bool = True,
        confirm_frames: int = 1,
    ) -> None:
        if (low_active and engage_at >= release_at) or (not low_active and engage_at <= release_at):
            raise ValueError("the release threshold must lie on the inactive side of engage_at")
        self.engage_at = engage_at
        self.release_at = release_at
        self.low_active = low_active
        self.confirm_frames = max(1, confirm_frames)
        self.active = False
        self._streak = 0

    def update(self, value: float) -> bool:
        if self.active:
            flip = value > self.release_at if self.low_active else value < self.release_at
        else:
            flip = value < self.engage_at if self.low_active else value > self.engage_at
        if flip:
            self._streak += 1
            if self._streak >= self.confirm_frames:
                self.active = not self.active
                self._streak = 0
        else:
            self._streak = 0
        return self.active

    def reset(self) -> None:
        self.active = False
        self._streak = 0


@dataclass(frozen=True)
class HandFeatures:
    palm_scale: float  # px
    pinch: dict[Finger, float]  # thumb tip <-> finger tip, palm lengths
    extension: dict[Finger, float]  # dist(wrist, tip) / dist(wrist, pip)
    index_middle_gap: float  # index tip <-> middle tip, palm lengths
    anchor: tuple[float, float]  # cursor anchor, normalized frame coordinates
    scroll_point: tuple[float, float]  # px, midpoint of the index and middle tips

    @classmethod
    def from_observation(
        cls, obs: HandObservation, anchor_index_weight: float = 0.5
    ) -> HandFeatures:
        pinch = {finger: obs.ndist(THUMB_TIP, finger.tip) for finger in PINCH_FINGERS}
        # A straight finger puts its tip much farther from the wrist than its PIP joint (~1.3x+),
        # a curled one folds the tip back towards the palm (<0.9x). Ratios of distances do not
        # care how the hand is rotated in the image, unlike the common `tip.y < pip.y` test,
        # which silently breaks as soon as the hand tilts sideways.
        extension = {
            finger: obs.dist(WRIST, finger.tip) / max(obs.dist(WRIST, finger.pip), 1e-6)
            for finger in Finger
        }
        w = anchor_index_weight
        anchor = w * obs.norm[INDEX_MCP, :2] + (1.0 - w) * obs.norm[MIDDLE_MCP, :2]
        scroll_point = (obs.px[INDEX_TIP, :2] + obs.px[MIDDLE_TIP, :2]) / 2.0
        return cls(
            palm_scale=obs.palm_scale,
            pinch=pinch,
            extension=extension,
            index_middle_gap=obs.ndist(INDEX_TIP, MIDDLE_TIP),
            anchor=(float(anchor[0]), float(anchor[1])),
            scroll_point=(float(scroll_point[0]), float(scroll_point[1])),
        )


@dataclass
class EngineOutput:
    mode: Mode
    label: str
    actions: list[Action] = field(default_factory=list)
    cursor_frozen: bool = False
    left_state: LeftState = LeftState.IDLE
    features: HandFeatures | None = None
    extended: dict[Finger, bool] = field(default_factory=dict)
    pending_mode: Mode | None = None  # posture change waiting for confirmation
    scroll_origin: tuple[float, float] | None = None  # px, for the HUD
    scroll_rate: tuple[float, float] = (0.0, 0.0)  # ticks/s, for the HUD
