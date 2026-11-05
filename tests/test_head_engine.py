"""--head on synthetic faces: head turns and nods press arrow keys."""

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


def test_turns_and_nods_press_the_matching_arrow_once() -> None:
    d = driver(repeat_delay=0.0)
    pressed: list[str] = []
    moves: list[dict[str, Any]] = [{"yaw": 0.2}, {"yaw": -0.2}, {"pitch": 0.15}, {"pitch": -0.15}]
    for move in moves:
        pressed += keys(d.hold(0.3, **move) + d.hold(0.3))  # move, then back to neutral
    # Coming back to the centre never fires the opposite arrow.
    assert pressed == ["right", "left", "down", "up"]


def test_holding_the_head_there_repeats_the_key() -> None:
    d = driver()
    pressed = keys(d.hold(1.5, yaw=0.2))
    # once at once, then every 0.25 s after 0.6 s
    assert set(pressed) == {"right"} and len(pressed) in (4, 5)


def test_small_moves_and_sideways_tilt_do_nothing() -> None:
    d = driver()
    actions = d.hold(1.0, yaw=0.05) + d.hold(1.0, pitch=0.03) + d.hold(1.0, roll_deg=20.0)
    assert keys(actions) == []


def test_distance_from_the_camera_does_not_matter() -> None:
    for eye_distance in (60.0, 200.0):
        d = driver(repeat_delay=0.0)
        assert keys(d.hold(0.3, yaw=0.2, eye_distance=eye_distance)) == ["right"]


def test_closing_the_eyes_pauses_head_control() -> None:
    d = driver(repeat_delay=0.0)
    d.hold(2.0, openness=CLOSED)
    assert d.step().mode is FaceMode.PAUSED
    assert keys(d.hold(0.5, yaw=0.2) + d.hold(0.3)) == []
    d.hold(2.0, openness=CLOSED)
    d.hold(0.7)  # resumed: the neutral pose is learned again
    assert d.outputs[-1].mode is FaceMode.HEAD
    assert keys(d.hold(0.3, yaw=-0.2)) == ["left"]


def test_slow_posture_drift_is_absorbed() -> None:
    d = driver(repeat_delay=0.0)
    pressed: list[str] = []
    for i in range(300):  # 10 s in which the resting pose drifts by most of a threshold
        pressed += keys(d.step(yaw=0.1 * i / 300).actions)
    assert pressed == []
    assert keys(d.hold(0.3, yaw=0.3)) == ["right"]  # a real turn from the new resting pose


def test_without_mirroring_the_raw_image_sides_are_swapped() -> None:
    d = driver(mirrored=False, repeat_delay=0.0)
    assert keys(d.hold(0.3, yaw=0.2)) == ["left"]
