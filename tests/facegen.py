"""Synthetic MediaPipe face landmarks for the eye and head tests.

Only the landmarks the code reads are placed (eye outlines, irises, nose tip); all other points
sit at the face centre. Coordinates follow a mirrored (selfie) frame, so the user's left eye is
on the image's left, as in a mirror.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.eye_engine import FaceOutput
from src.face_tracker import EYE_OUTLINES, IRIS_CENTERS, NOSE_TIP, FaceObservation
from src.gaze import GazeCalibration
from src.os_controller import Action

FRAME = (1280, 720)
SCREEN = (1920, 1080)
OPEN_ASPECT = 0.32  # eye aspect ratio of an open eye (real MediaPipe data: 0.29-0.40)
NOSE_DROP = 0.7  # neutral nose tip position below the eye line, in eye distances
GAZE_RANGE = (0.4, 0.24)  # synthetic iris offset range (eye widths) across the screen


def make_face(
    t: float = 0.0,
    *,
    openness: tuple[float, float] = (1.0, 1.0),
    gaze: tuple[float, float] = (0.0, 0.0),
    center: tuple[float, float] = (640.0, 330.0),
    eye_distance: float = 120.0,
    eye_width: float = 50.0,
    eye_aspect: float = OPEN_ASPECT,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll_deg: float = 0.0,
    frame: tuple[int, int] = FRAME,
) -> FaceObservation:
    """openness: (user's left, user's right) eye, 1 = open, 0 = shut.
    gaze: iris offset from the eye centre in eye widths (+x right, +y down).
    yaw / pitch: nose offset from its neutral position, in eye distances (+ = right / down)."""
    r = math.radians(roll_deg)
    u = np.array([math.cos(r), math.sin(r)])
    v = np.array([-math.sin(r), math.cos(r)])
    c = np.asarray(center, dtype=np.float64)
    pts = np.tile(np.append(c, 0.0), (478, 1))
    for outline, iris, side, eye_open in zip(
        EYE_OUTLINES, IRIS_CENTERS, (-1, 1), openness, strict=True
    ):
        eye = c + u * side * eye_distance / 2
        half = eye_aspect * eye_open * eye_width / 2
        first, second = outline["corners"]
        pts[first, :2] = eye - u * eye_width / 2
        pts[second, :2] = eye + u * eye_width / 2
        for upper, lower, dx in zip(outline["upper"], outline["lower"], (-1, 1), strict=True):
            pts[upper, :2] = eye + u * dx * eye_width / 6 - v * half
            pts[lower, :2] = eye + u * dx * eye_width / 6 + v * half
        pts[iris, :2] = eye + u * gaze[0] * eye_width + v * gaze[1] * eye_width
    pts[NOSE_TIP, :2] = c + u * yaw * eye_distance + v * (NOSE_DROP + pitch) * eye_distance
    width, height = frame
    return FaceObservation.from_normalized(pts / np.array([width, height, width]), frame, t)


def gaze_for(x: float, y: float, screen: tuple[int, int] = SCREEN) -> tuple[float, float]:
    """The synthetic iris offset of someone looking at screen pixel (x, y)."""
    return (
        (x / (screen[0] - 1) - 0.5) * GAZE_RANGE[0],
        (y / (screen[1] - 1) - 0.5) * GAZE_RANGE[1],
    )


def linear_calibration(screen: tuple[int, int] = SCREEN) -> GazeCalibration:
    """A calibration fitted on a 3x3 grid of the synthetic gaze above."""
    targets = [
        (fx * (screen[0] - 1), fy * (screen[1] - 1))
        for fy in (0.1, 0.5, 0.9)
        for fx in (0.1, 0.5, 0.9)
    ]
    return GazeCalibration.fit([gaze_for(x, y, screen) for x, y in targets], targets, screen)


class FaceDriver:
    """Feeds synthetic faces to an EyeEngine / HeadArrows at a fixed frame rate."""

    def __init__(self, engine: Any, fps: float = 30.0) -> None:
        self.engine = engine
        self.dt = 1.0 / fps
        self.t = 0.0
        self.outputs: list[FaceOutput] = []

    def step(self, present: bool = True, **face: Any) -> FaceOutput:
        obs = make_face(self.t, **face) if present else None
        out: FaceOutput = self.engine.update(obs, self.t)
        self.outputs.append(out)
        self.t += self.dt
        return out

    def hold(self, seconds: float, present: bool = True, **face: Any) -> list[Action]:
        actions: list[Action] = []
        for _ in range(max(1, round(seconds / self.dt))):
            actions += self.step(present, **face).actions
        return actions
