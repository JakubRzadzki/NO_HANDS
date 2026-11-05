"""Head pose from MediaPipe's facial transformation matrix, and the gaze features."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.face_tracker import gaze_features, head_from_matrix, measure_eyes
from tests.facegen import make_face


def transform(
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    position: tuple[float, float, float] = (0.0, 0.0, -60.0),
) -> np.ndarray:
    """A face transform as MediaPipe reports it: camera frame x right, y up, z towards the
    camera; the canonical face looks along +z (at the camera). Angles in degrees, in the
    image's view: yaw > 0 = the face turns to the image's right, pitch > 0 = it looks down."""
    a, b, c = (math.radians(v) for v in (yaw, pitch, roll))
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx @ rz
    m[:3, 3] = position
    return m


def test_head_angles_and_position_from_the_matrix() -> None:
    (yaw, pitch, roll), (x, y, distance) = head_from_matrix(
        transform(20, 0, 0, (5.0, -3.0, -62.0)), mirrored=True
    )
    assert (yaw, pitch, roll) == pytest.approx((20.0, 0.0, 0.0), abs=1e-6)
    assert (x, y, distance) == pytest.approx((5.0, -3.0, 62.0))
    (yaw, pitch, _), _ = head_from_matrix(transform(0, 15, 0), mirrored=True)
    assert (yaw, pitch) == pytest.approx((0.0, 15.0), abs=1e-6)  # nod down = positive pitch
    (yaw, pitch, roll), _ = head_from_matrix(transform(0, 0, 25), mirrored=True)
    assert (yaw, pitch, roll) == pytest.approx((0.0, 0.0, 25.0), abs=1e-6)
    (yaw, pitch, _), _ = head_from_matrix(transform(10, 10, 0), mirrored=True)
    assert yaw == pytest.approx(10.0, abs=0.5) and pitch == pytest.approx(10.0, abs=0.5)


def test_without_mirroring_the_sides_are_swapped_into_the_users_view() -> None:
    """In a raw (unmirrored) image the user's right is the image's left."""
    (yaw, pitch, roll), (x, y, _) = head_from_matrix(
        transform(20, 10, 5, (5.0, -3.0, -60.0)), mirrored=False
    )
    assert (yaw, pitch, roll) == pytest.approx((-20.0, 10.0, -5.0), abs=0.5)
    assert (x, y) == pytest.approx((-5.0, -3.0))


def test_gaze_features_are_in_the_users_view_and_screen_oriented() -> None:
    obs = make_face(gaze=(0.1, 0.05), yaw=4.0, pitch=-3.0, head_position=(2.0, 6.0, 55.0))
    left, right = measure_eyes(obs, mirrored=True)
    ix, iy, yaw, pitch, hx, hy = gaze_features(obs, left, right, mirrored=True)
    assert (ix, iy) == pytest.approx((0.1, 0.05), abs=1e-6)  # looking right and down
    assert (yaw, pitch, hx) == pytest.approx((4.0, -3.0, 2.0))
    assert hy == pytest.approx(-6.0)  # 6 cm up = -6 in screen orientation (y down)
    # The same image without mirroring: the iris offset to the image's right is the user's left.
    left, right = measure_eyes(obs, mirrored=False)
    assert gaze_features(obs, left, right, mirrored=False)[0] == pytest.approx(-0.1, abs=1e-6)
