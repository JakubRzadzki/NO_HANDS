"""--head on synthetic faces: quick head turns and nods press arrow keys."""

from __future__ import annotations

from typing import Any

from config import FaceConfig, HeadConfig
from src.eye_engine import FaceMode
from src.head_engine import HeadArrows
from src.os_controller import Action, KeyTap
from tests.facegen import FaceDriver

CLOSED = (0.05, 0.05)


def keys(actions: list[Action]) -> list[str]:
    return [a.key for a in actions if isinstance(a, KeyTap) for _ in range(a.count)]


def driver(mirrored: bool = True, **overrides: Any) -> FaceDriver:
    cfg = HeadConfig()
    for name, value in overrides.items():
        setattr(cfg, name, value)
    d = FaceDriver(HeadArrows(cfg, FaceConfig(), mirrored))
    d.hold(0.7)  # learn the neutral pose and the open-eye baseline
    assert d.outputs[-1].mode is FaceMode.HEAD
    return d


def turn(d: FaceDriver, seconds: float, **goal: float) -> list[Action]:
    """Move the head from neutral to ``goal`` (degrees) over ``seconds``, linearly."""
    frames = max(1, round(seconds / d.dt))
    actions: list[Action] = []
    for i in range(1, frames + 1):
        pose: dict[str, Any] = {axis: value * i / frames for axis, value in goal.items()}
        actions += d.step(**pose).actions
    return actions


def test_quick_turns_and_nods_press_the_matching_arrow_once() -> None:
    d = driver()
    pressed: list[str] = []
    moves: list[dict[str, Any]] = [{"yaw": 22}, {"yaw": -22}, {"pitch": 18}, {"pitch": -18}]
    for move in moves:
        pressed += keys(turn(d, 0.25, **move) + d.hold(0.5, **move) + d.hold(0.4))
    # Coming back to the centre never fires the opposite arrow, holding does not repeat.
    assert pressed == ["right", "left", "down", "up"]


def test_slowly_turning_to_look_elsewhere_presses_nothing() -> None:
    d = driver()
    assert keys(turn(d, 3.0, yaw=30.0) + d.hold(1.0, yaw=30.0)) == []  # 10 deg/s
    assert keys(turn(d, 3.0, pitch=25.0) + d.hold(1.0, pitch=25.0)) == []
    d.hold(0.5)  # back in the centre: armed again
    assert keys(turn(d, 0.2, yaw=-22.0)) == ["left"]


def test_holding_the_head_there_can_repeat_the_key() -> None:
    d = driver(repeat_delay=0.6)
    pressed = keys(turn(d, 0.2, yaw=22.0) + d.hold(1.5, yaw=22.0))
    # once at once, then every 0.25 s after 0.6 s
    assert set(pressed) == {"right"} and len(pressed) in (5, 6)


def test_small_moves_and_sideways_tilt_do_nothing() -> None:
    d = driver()
    actions = turn(d, 0.2, yaw=10.0) + d.hold(1.0, yaw=10.0) + d.hold(0.5)
    actions += turn(d, 0.2, pitch=-9.0) + d.hold(1.0, pitch=-9.0) + d.hold(0.5)
    actions += d.hold(1.0, roll_deg=25.0)
    assert keys(actions) == []


def test_distance_from_the_camera_does_not_matter() -> None:
    for distance in (40.0, 100.0):
        d = FaceDriver(HeadArrows(HeadConfig(), FaceConfig()))
        d.hold(0.7, head_position=(0.0, 0.0, distance))
        assert keys(turn(d, 0.2, yaw=22.0)) == ["right"]


def test_closing_the_eyes_pauses_head_control() -> None:
    d = driver()
    d.hold(2.0, openness=CLOSED)
    assert d.step().mode is FaceMode.PAUSED
    assert keys(turn(d, 0.2, yaw=22.0) + d.hold(0.3)) == []
    d.hold(2.0, openness=CLOSED)
    d.hold(0.7)  # resumed: the neutral pose is learned again
    assert d.outputs[-1].mode is FaceMode.HEAD
    assert keys(turn(d, 0.2, yaw=-22.0)) == ["left"]


def test_slow_posture_drift_is_absorbed() -> None:
    d = driver()
    pressed: list[str] = []
    for i in range(300):  # 10 s in which the resting pose drifts by 12 degrees
        pressed += keys(d.step(yaw=12.0 * i / 300).actions)
    assert pressed == []
    d.hold(0.5, yaw=12.0)
    # A quick 17-degree turn from the new resting pose fires; from the old one it is 29.
    actions = []
    for i in range(1, 7):
        actions += d.step(yaw=12.0 + 17.0 * i / 6).actions
    assert keys(actions) == ["right"]


def test_head_pose_is_already_in_the_users_view() -> None:
    """Mirroring is handled where the pose is computed (head_from_matrix), so the engine
    presses the same arrow for the same head turn with or without a mirrored image."""
    for mirrored in (True, False):
        d = driver(mirrored=mirrored)
        assert keys(turn(d, 0.2, yaw=22.0)) == ["right"]
