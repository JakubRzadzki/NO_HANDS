"""Synthetic hand landmarks for tests: a tiny forward-kinematics hand model.

Model space is metres: x to the right, y up towards the fingertips, z negative towards the
camera (MediaPipe convention: smaller z = closer). Fingers curl towards the camera, so a fist
seen from the front is foreshortened the way a real one is.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from config import AppConfig
from src.gesture_engine import EngineOutput, GestureEngine
from src.hand_tracker import HandObservation
from src.os_controller import Action
from src.pointer import CursorMapper, Roi

FloatArray = npt.NDArray[np.float64]

# name: (MCP position, segment lengths proximal/middle/distal, landmark ids MCP..TIP)
FINGERS: dict[str, tuple[tuple[float, float], tuple[float, float, float], tuple[int, ...]]] = {
    "index": ((0.025, 0.085), (0.045, 0.025, 0.022), (5, 6, 7, 8)),
    "middle": ((0.005, 0.090), (0.050, 0.030, 0.024), (9, 10, 11, 12)),
    "ring": ((-0.015, 0.085), (0.047, 0.028, 0.023), (13, 14, 15, 16)),
    "pinky": ((-0.033, 0.075), (0.035, 0.020, 0.020), (17, 18, 19, 20)),
}
TIPS = {name: ids[3] for name, (_, _, ids) in FINGERS.items()}
JOINT_LIMITS = (math.radians(85), math.radians(100), math.radians(70))  # MCP, PIP, DIP
PALM_M = math.hypot(0.005, 0.090)  # wrist -> middle MCP
THUMB_CMC = np.array([0.030, 0.025, -0.005])
CENTER = (0.5, 0.55)  # default anchor position = centre of the default ROI

# Poses -------------------------------------------------------------------------------------
OPEN: dict[str, Any] = {}
FIST: dict[str, Any] = {
    "curl": {"index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "thumb": "tucked",
}
SCROLL: dict[str, Any] = {  # index + middle up and touching, ring + pinky curled
    "curl": {"ring": 1.0, "pinky": 1.0},
    "spread": {"index": -6.0, "middle": 4.0},
    "thumb": "tucked",
}
VICTORY: dict[str, Any] = {  # same fingers as SCROLL but spread apart -> plain pointer
    "curl": {"ring": 1.0, "pinky": 1.0},
    "spread": {"index": 10.0, "middle": -8.0},
    "thumb": "tucked",
}


def pinch(finger: str, ratio: float = 0.05) -> dict[str, Any]:
    """Open hand with the thumb tip ``ratio`` palm lengths from ``finger``'s tip."""
    return {"pinch": finger, "pinch_ratio": ratio}


def rock(thumb_index_ratio: float) -> dict[str, Any]:
    """Zoom posture (index + pinky up, middle + ring curled) with a given thumb-index spread."""
    return {
        "curl": {"middle": 1.0, "ring": 1.0},
        "pinch": "index",
        "pinch_ratio": thumb_index_ratio,
    }


# Geometry ----------------------------------------------------------------------------------
def hand_points(
    curl: dict[str, float] | None = None,
    spread: dict[str, float] | None = None,
    pinch: str | None = None,
    pinch_ratio: float = 0.0,
    thumb: str = "open",
    thumb_tip: npt.ArrayLike | None = None,
) -> FloatArray:
    """21 landmarks in model space (see module docstring)."""
    curl = curl or {}
    spread = spread or {}
    pts = np.zeros((21, 3))
    for name, ((mx, my), segments, ids) in FINGERS.items():
        bend, side = curl.get(name, 0.0), math.radians(spread.get(name, 0.0))
        pos = np.array([mx, my, 0.0])
        pts[ids[0]] = pos
        phi = 0.0
        for k, length in enumerate(segments):
            phi += bend * JOINT_LIMITS[k]
            direction = np.array(
                [math.sin(side) * math.cos(phi), math.cos(side) * math.cos(phi), -math.sin(phi)]
            )
            pos = pos + length * direction
            pts[ids[k + 1]] = pos

    if thumb_tip is not None:
        tip = np.asarray(thumb_tip, dtype=np.float64)
    elif pinch is not None:  # thumb tip in front of the fingertip (towards the camera)
        tip = pts[TIPS[pinch]] + np.array([0.0, 0.0, -pinch_ratio * PALM_M])
    elif thumb == "tucked":  # folded over the middle/ring knuckles
        tip = (pts[10] + pts[14]) / 2 + np.array([0.0, -0.01, -0.02])
    else:  # sticking out sideways
        tip = THUMB_CMC + np.array([0.055, 0.045, -0.010])
    bend_offset = np.array([0.012, -0.006, -0.004])
    pts[1] = THUMB_CMC
    pts[2] = THUMB_CMC + 0.40 * (tip - THUMB_CMC) + bend_offset
    pts[3] = THUMB_CMC + 0.75 * (tip - THUMB_CMC) + 0.5 * bend_offset
    pts[4] = tip
    return pts


def project(
    points: FloatArray,
    *,
    center: tuple[float, float] = CENTER,
    palm_px: float = 150.0,
    roll_deg: float = 0.0,
    frame: tuple[int, int] = (1280, 720),
) -> FloatArray:
    """Model space -> MediaPipe normalized coordinates.

    ``center`` is where the cursor anchor (midpoint of landmarks 5 and 9) lands in the frame,
    ``palm_px`` the palm length in pixels (hand distance), ``roll_deg`` an in-plane rotation.
    """
    width, height = frame
    k = palm_px / PALM_M  # px per metre
    r = math.radians(roll_deg)
    rotation = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    xy = points[:, :2] @ rotation.T
    anchor = (xy[5] + xy[9]) / 2
    px_x = center[0] * width + (xy[:, 0] - anchor[0]) * k
    px_y = center[1] * height - (xy[:, 1] - anchor[1]) * k  # image y grows downwards
    z = (points[:, 2] - points[0, 2]) * k
    return np.column_stack([px_x / width, px_y / height, z / width])


def make_obs(
    t: float = 0.0,
    *,
    center: tuple[float, float] = CENTER,
    palm_px: float = 150.0,
    roll_deg: float = 0.0,
    frame: tuple[int, int] = (1280, 720),
    **pose: Any,
) -> HandObservation:
    norm = project(
        hand_points(**pose), center=center, palm_px=palm_px, roll_deg=roll_deg, frame=frame
    )
    return HandObservation.from_normalized(norm, frame, t)


# Engine harness ----------------------------------------------------------------------------
def make_engine(
    mode: str = "absolute",
    screen: tuple[int, int] = (1920, 1080),
    frame: tuple[int, int] = (1280, 720),
    **gesture_overrides: Any,
) -> GestureEngine:
    cfg = AppConfig()
    for name, value in gesture_overrides.items():
        if not hasattr(cfg.gesture, name):
            raise AttributeError(name)
        setattr(cfg.gesture, name, value)
    cfg.validate()
    p = cfg.pointer
    roi = Roi.for_screen(frame, screen, p.roi_width, p.roi_center, p.sensitivity)
    mapper = CursorMapper(roi, screen, cfg.filter, mode, p.edge_overshoot, p.relative_gain)
    return GestureEngine(cfg.gesture, mapper, p.anchor_index_mcp_weight)


class Driver:
    """Feeds poses to an engine at a fixed frame rate and records every output."""

    def __init__(self, engine: GestureEngine, fps: float = 60.0) -> None:
        self.engine = engine
        self.dt = 1.0 / fps
        self.t = 0.0
        self.outputs: list[EngineOutput] = []

    def step(self, pose: dict[str, Any] | None, **view: Any) -> EngineOutput:
        """One frame. ``pose=None`` simulates a frame without a detected hand."""
        obs = None if pose is None else make_obs(self.t, **view, **pose)
        out = self.engine.update(obs, self.t)
        self.outputs.append(out)
        self.t += self.dt
        return out

    def hold(self, pose: dict[str, Any] | None, seconds: float, **view: Any) -> list[Action]:
        """Repeat a pose for ``seconds``; returns the actions emitted meanwhile."""
        actions: list[Action] = []
        for _ in range(max(1, round(seconds / self.dt))):
            actions += self.step(pose, **view).actions
        return actions

    def sweep(
        self,
        pose: dict[str, Any],
        start: tuple[float, float],
        end: tuple[float, float],
        seconds: float,
        **view: Any,
    ) -> list[Action]:
        """Move the hand linearly from ``start`` to ``end`` (anchor positions) over ``seconds``."""
        n = max(1, round(seconds / self.dt))
        actions: list[Action] = []
        for i in range(1, n + 1):
            a = i / n
            center = (start[0] + (end[0] - start[0]) * a, start[1] + (end[1] - start[1]) * a)
            actions += self.step(pose, center=center, **view).actions
        return actions
