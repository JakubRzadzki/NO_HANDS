"""The gaze model, its calibration and the fixation filter, on a simulated laptop user."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.gaze import HEAD_MOVES, CalibrationSession, FixationFilter, GazeModel
from tests.facegen import SCREEN, linear_calibration
from tests.gazesim import REST, Pose, Viewer, calibrate


def calibrated(
    viewer: Viewer,
    head: Callable[[float, str], Pose] | None = None,
    glitch: Callable[[float], bool] | None = None,
    eye_only: bool = False,
) -> GazeModel:
    session = CalibrationSession(SCREEN, settle=0.7, sample=1.0, head_seconds=6.0)
    calibrate(session, viewer, head=head, glitch=glitch)
    if eye_only:  # the same recording with all head information removed
        for step in session.steps:
            step.samples = [((*f[:2], 0.0, 0.0, 0.0, REST["y"]), 60.0) for f, _ in step.samples]
    return session.result()


def errors(
    model: GazeModel, viewer: Viewer, poses: list[Pose], seed: int = 1, eye_only: bool = False
) -> np.ndarray:
    """px between where the user looks and the prediction, one random target per pose."""
    rng = np.random.default_rng(seed)
    out = []
    for pose in poses:
        target = (rng.uniform(0, SCREEN[0] - 1), rng.uniform(0, SCREEN[1] - 1))
        features, distance = viewer.features(target, **pose)
        if eye_only:  # what an iris-only model sees: no head information at all
            features = (*features[:2], 0.0, 0.0, 0.0, REST["y"])
            distance = REST["distance"]
        x, y = model.predict(features, distance)
        out.append(math.hypot(x - target[0], y - target[1]))
    return np.array(out)


def random_poses(viewer: Viewer, n: int = 300, seed: int = 2) -> list[Pose]:
    rng = np.random.default_rng(seed)
    return [viewer.random_pose(rng) for _ in range(n)]


# -- the model ------------------------------------------------------------------------------------
def test_head_movement_is_compensated() -> None:
    """The point of the head-aware model: turn, lean or shift the head while looking at the
    same spot and the prediction stays there. An iris-only model, calibrated on exactly the
    same data, is off by centimetres."""
    viewer = Viewer()
    model = calibrated(viewer)
    poses = random_poses(viewer)
    head_aware = errors(model, viewer, poses)
    assert np.median(head_aware) < 40 and np.percentile(head_aware, 90) < 80

    iris_only = errors(calibrated(viewer, eye_only=True), viewer, poses, eye_only=True)
    assert np.median(iris_only) > 4 * np.median(head_aware)


def test_rotation_and_position_moving_together_do_not_confuse_the_fit() -> None:
    """In the head phase turning the head also moves the eyes, so rotation and position are
    almost the same signal there. Fitted as separate features (an earlier version) their
    weights were arbitrary and a later sideways lean without a turn was ~6 cm off."""
    viewer = Viewer()
    model = calibrated(viewer)  # the simulated user follows the arrows: rotation moves the eyes
    lean = [{"x": x} for x in (-5.0, -2.5, 2.5, 5.0)] * 50  # lean sideways, no turn
    assert np.median(errors(model, viewer, lean)) < 40


def test_sitting_closer_or_farther_is_handled() -> None:
    viewer = Viewer()
    model = calibrated(viewer)
    for distance in (45.0, 75.0):
        assert np.median(errors(model, viewer, [{"distance": distance}] * 200)) < 70


def test_tracking_glitches_during_calibration_are_rejected() -> None:
    viewer = Viewer()
    clean = calibrated(viewer)
    glitchy = calibrated(viewer, glitch=lambda t: int(t * 30) % 17 == 0)  # ~6% garbage frames
    poses = random_poses(viewer, 200)
    assert np.median(errors(glitchy, viewer, poses)) < 1.5 * np.median(errors(clean, viewer, poses))


def test_realistic_noise_accuracy_and_the_reported_numbers_are_honest() -> None:
    viewer = Viewer(iris_noise=0.012, angle_noise=0.3, rng=np.random.default_rng(5))
    model = calibrated(viewer)
    # The systematic error that is left, judged without per-frame noise:
    assert np.median(errors(model, Viewer(), random_poses(viewer))) < 70
    # Jitter: one frame's prediction while staring at one point, vs the estimate.
    target = (900.0, 500.0)
    frames = np.array([model.predict(*viewer.features(target)) for _ in range(400)])
    measured = float(np.mean(frames.std(axis=0)))
    assert model.noise == pytest.approx(measured, rel=0.3)
    assert model.validation_error is not None and 10 < model.validation_error < 120


def test_a_lazy_head_phase_is_reported() -> None:
    viewer = Viewer()

    def barely(t: float, phase: str) -> Pose:
        return {"yaw": math.sin(t), "pitch": 0.5 * math.sin(t)} if phase == "head" else {}

    good, lazy = calibrated(viewer), calibrated(viewer, head=barely)
    assert good.head_range[0] > 5 and good.head_range[1] > 3
    assert lazy.head_range[0] < 1 and lazy.head_range[1] < 0.5


def test_validation_catches_a_user_who_looked_elsewhere() -> None:
    """The check dots are not used to judge the fit they were part of."""
    viewer = Viewer()
    session = CalibrationSession(SCREEN, settle=0.7, sample=1.0, head_seconds=6.0)
    t = 0.0
    while not session.done:
        target = session.target
        assert target is not None
        if session.phase == "check":  # stares at the screen centre instead of the dots
            target = (960.0, 540.0)
        session.update(t, viewer.features(target))
        t += 1 / 30
    model = session.result()
    assert model.validation_error is not None and model.validation_error > 250


def test_fit_needs_enough_samples() -> None:
    with pytest.raises(ValueError):
        GazeModel.fit([(0.0,) * 6] * 10, [(0.0, 0.0)] * 10, SCREEN)


def test_model_round_trip_and_rejections(tmp_path: Path) -> None:
    model = linear_calibration(noise=42.0)
    model.validation_error = 55.0
    path = tmp_path / "models" / "gaze.json"
    model.save(path)
    loaded = GazeModel.load(path, SCREEN)
    assert loaded is not None
    features = (0.05, -0.02, 3.0, -2.0, 1.0, 5.0)
    assert loaded.predict(features, 55.0) == pytest.approx(model.predict(features, 55.0))
    assert (loaded.noise, loaded.validation_error) == (42.0, 55.0)
    assert GazeModel.load(path, (2560, 1440)) is None  # made for another screen
    assert GazeModel.load(tmp_path / "missing.json", SCREEN) is None
    (tmp_path / "bad.json").write_text("{not json")
    assert GazeModel.load(tmp_path / "bad.json", SCREEN) is None
    old = {"mean": [0, 0], "scale": [1, 1], "coef": [[0, 0]] * 6, "screen_size": list(SCREEN)}
    (tmp_path / "old.json").write_text(json.dumps(old))  # the first version's calibration
    assert GazeModel.load(tmp_path / "old.json", SCREEN) is None


# -- the calibration procedure --------------------------------------------------------------------
def test_session_runs_grid_head_and_check_phases_in_order() -> None:
    session = CalibrationSession(SCREEN, settle=0.7, sample=1.0, head_seconds=6.0)
    phases: list[str] = []
    hints: list[str] = []
    t = 0.0
    while not session.done:
        if not phases or phases[-1] != session.phase:
            phases.append(session.phase)
        hint = session.head_hint
        if hint is not None and (not hints or hints[-1] != hint[0]):
            hints.append(hint[0])
        assert session.instruction
        target = session.target
        assert target is not None
        session.update(t, Viewer().features(target))
        t += 1 / 30
    assert phases == ["grid", "head", "check"]
    assert hints == [name for name, _ in HEAD_MOVES]  # left, right, up, down
    assert t == pytest.approx(9 * 1.7 + 6.7 + 4 * 1.7, abs=0.5)
    sizes = [len(step.samples) for step in session.steps]
    assert all(28 <= n <= 32 for i, n in enumerate(sizes) if i != 9)  # 1 s at 30 FPS
    assert 178 <= sizes[9] <= 182  # 6 s of head movement
    assert session.phase == "done" and session.instruction == ""


def test_session_without_usable_samples_gives_no_model() -> None:
    session = CalibrationSession(SCREEN, settle=0.2, sample=0.3, head_seconds=0.5)
    t = 0.0
    while not session.done:
        session.update(t, None)  # face never found / eyes closed
        t += 1 / 30
    with pytest.raises(ValueError, match="dots got enough samples"):
        session.result()


def test_skipping_the_head_phase_still_calibrates() -> None:
    session = CalibrationSession(SCREEN, settle=0.5, sample=0.8, head_seconds=0.0)
    calibrate(session, Viewer())
    model = session.result()
    assert [s.phase for s in session.steps].count("head") == 0
    assert model.head_range == (0.0, 0.0)
    assert np.median(errors(model, Viewer(), [{}] * 100)) < 30  # fine while the head is still


# -- the fixation filter --------------------------------------------------------------------------
def correlated_noise(n: int, sigma: float, rho: float, seed: int) -> np.ndarray:
    """Webcam gaze noise wanders: each frame is mostly the previous frame's error."""
    rng = np.random.default_rng(seed)
    e = np.zeros((n, 2))
    e[0] = rng.normal(0, sigma, 2)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, sigma * math.sqrt(1 - rho**2), 2)
    return e


def run_filter(f: FixationFilter, points: np.ndarray, fps: float = 30.0) -> tuple[np.ndarray, int]:
    out, jumps = [], 0
    for i, p in enumerate(points):
        out.append(f.update((float(p[0]), float(p[1])), i / fps))
        jumps += f.jumped
    return np.array(out), jumps


def test_staring_at_one_point_keeps_the_cursor_still() -> None:
    sigma, false_jumps = 90.0, 0
    for seed in range(4):  # 4 x 2 minutes of staring
        noise = correlated_noise(30 * 120, sigma, 0.7, seed=seed)
        out, jumps = run_filter(FixationFilter(3 * sigma), np.array([800.0, 400.0]) + noise)
        false_jumps += jumps - 1  # the first fixation is not a false jump
        assert np.linalg.norm(out[-1] - (800, 400)) < 160  # about one sigma: settled, not lost
        settled = out[30:]
        moving = np.count_nonzero(np.linalg.norm(np.diff(settled, axis=0), axis=1) > 0)
        assert moving < 0.03 * len(settled)  # frozen almost all the time
    assert false_jumps <= 8  # at most one per minute, although the raw gaze jitters by 90 px


def test_saccades_move_the_cursor_quickly() -> None:
    sigma = 60.0
    f = FixationFilter(3 * sigma)
    rng = np.random.default_rng(4)
    targets = [(400.0, 300.0), (1500.0, 800.0), (1500.0, 250.0), (700.0, 900.0)]
    t, latencies = 0.0, []
    for target in targets:
        start = t
        landed = None
        for _ in range(45):  # 1.5 s per target
            p = np.asarray(target) + rng.normal(0, sigma, 2)
            f.update((float(p[0]), float(p[1])), t)
            if f.jumped and landed is None:
                landed = t
            t += 1 / 30
        assert landed is not None
        latencies.append(landed - start)
        assert f.center is not None
        assert math.dist(f.center, target) < 70
    assert max(latencies[1:]) <= 0.1  # two frames (+ one for the step) at 30 FPS


def test_scattered_outliers_do_not_move_the_cursor() -> None:
    f = FixationFilter(150.0)
    for i in range(30):
        f.update((960.0, 540.0), i / 30)
    t = 1.0
    for p in [(1500.0, 540.0), (300.0, 540.0), (960.0, 1000.0), (960.0, 100.0)]:  # disagree
        f.update(p, t)
        t += 1 / 30
        assert not f.jumped
    assert f.center == pytest.approx((960.0, 540.0))


def test_looking_at_the_cursor_does_not_drag_it_away() -> None:
    """The classic runaway: the cursor lands 120 px off; you look at it; the estimate moves
    120 px further... A frozen fixation ignores looks within the radius."""
    f = FixationFilter(200.0)
    t = 0.0
    for _ in range(30):
        f.update((1120.0, 540.0), t)  # looking at a target at 1000, the estimate says 1120
        t += 1 / 30
    for _ in range(60):
        f.update((1240.0, 540.0), t)  # looking at the cursor (1120): the estimate says 1240
        t += 1 / 30
    assert f.center == pytest.approx((1120.0, 540.0))


def test_a_small_real_gaze_shift_is_caught_by_dwell() -> None:
    sigma = 60.0
    f = FixationFilter(3 * sigma, dwell=0.6)
    noise = correlated_noise(150, sigma, 0.7, seed=6)
    points = np.array([[960.0, 540.0]] * 60 + [[960.0 + 260, 540.0]] * 90) + noise
    out, _ = run_filter(f, points)
    assert abs(out[-1][0] - 1220) < 90  # moved to the new spot within the 3 s


def test_slowly_wandering_estimate_does_not_jump() -> None:
    f = FixationFilter(150.0, dwell=0.0)
    t, jumps = 0.0, 0
    for i in range(90):  # drifts 200 px in 3 s without a fast step
        f.update((960.0 + 200.0 * i / 90, 540.0), t)
        jumps += f.jumped
        t += 1 / 30
    assert jumps == 1  # only the first fixation
