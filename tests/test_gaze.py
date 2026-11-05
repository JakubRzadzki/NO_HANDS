from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.gaze import CALIBRATION_TARGETS, CalibrationSession, FixationFilter, GazeCalibration
from tests.facegen import SCREEN, gaze_for, linear_calibration


def quadratic(f: np.ndarray) -> np.ndarray:
    """A smooth, nonlinear stand-in for the true eye -> screen mapping."""
    x, y = f[:, 0], f[:, 1]
    return np.column_stack([900 + 4000 * x + 3000 * x * x, 500 + 3500 * y - 2000 * x * y])


def test_fit_recovers_a_quadratic_mapping() -> None:
    rng = np.random.default_rng(0)
    features = rng.uniform(-0.2, 0.2, size=(20, 2))
    cal = GazeCalibration.fit(features, quadratic(features), SCREEN, ridge=0.0)
    probe = rng.uniform(-0.2, 0.2, size=(5, 2))
    predicted = np.array([cal.predict((float(x), float(y))) for x, y in probe])
    assert np.allclose(predicted, quadratic(probe), atol=1e-6)
    assert cal.rms_error < 1e-6


def test_ridge_keeps_a_noisy_nine_point_fit_sane_between_the_dots() -> None:
    rng = np.random.default_rng(1)
    targets = [(fx * (SCREEN[0] - 1), fy * (SCREEN[1] - 1)) for fx, fy in CALIBRATION_TARGETS]
    features = np.array([gaze_for(x, y) for x, y in targets]) + rng.normal(0, 0.004, (9, 2))
    cal = GazeCalibration.fit(features, targets, SCREEN)
    for x, y in [(400, 300), (1500, 800), (960, 200)]:
        px, py = cal.predict(gaze_for(x, y))
        assert abs(px - x) < 120 and abs(py - y) < 120


def test_fit_needs_six_points() -> None:
    with pytest.raises(ValueError):
        GazeCalibration.fit([(0.0, 0.0)] * 5, [(0.0, 0.0)] * 5, SCREEN)


def test_calibration_round_trip_and_screen_check(tmp_path: Path) -> None:
    cal = linear_calibration()
    path = tmp_path / "models" / "gaze.json"
    cal.save(path)
    loaded = GazeCalibration.load(path, SCREEN)
    assert loaded is not None
    assert loaded.predict((0.05, -0.02)) == pytest.approx(cal.predict((0.05, -0.02)))
    assert GazeCalibration.load(path, (2560, 1440)) is None  # made for another screen
    assert GazeCalibration.load(tmp_path / "missing.json", SCREEN) is None
    (tmp_path / "bad.json").write_text("{not json")
    assert GazeCalibration.load(tmp_path / "bad.json", SCREEN) is None


def run_session(
    session: CalibrationSession,
    feature_for: Callable[[tuple[float, float]], tuple[float, float] | None],
    fps: float = 30.0,
) -> float:
    t = 0.0
    while not session.done:
        target = session.target
        assert target is not None
        session.update(t, feature_for(target))
        t += 1.0 / fps
    return t


def test_session_samples_only_after_settling_and_fits_a_perfect_user() -> None:
    session = CalibrationSession(SCREEN, settle=0.7, sample=1.0)
    duration = run_session(session, lambda target: gaze_for(*target))
    assert duration == pytest.approx(9 * 1.7, abs=0.5)
    assert all(28 <= len(samples) <= 32 for samples in session.samples)  # 1 s at 30 FPS
    cal = session.result()
    for x, y in session.targets:
        px, py = cal.predict(gaze_for(x, y))
        assert abs(px - x) < 5 and abs(py - y) < 5


def test_session_without_usable_samples_gives_no_calibration() -> None:
    session = CalibrationSession(SCREEN, settle=0.2, sample=0.3)
    run_session(session, lambda target: None)  # face never found / eyes closed
    with pytest.raises(ValueError, match="dots got enough samples"):
        session.result()


def test_fixation_filter_holds_still_and_follows_saccades() -> None:
    rng = np.random.default_rng(2)
    f = FixationFilter(radius_px=134, saccade_frames=2, smoothing=0.12)
    out = np.array(
        [f.update((800 + rng.normal(0, 40), 400 + rng.normal(0, 40))) for _ in range(120)]
    )
    assert np.std(out[30:], axis=0).max() < 15  # raw gaze noise is 40 px
    assert np.allclose(out[-1], (800, 400), atol=20)
    assert np.allclose(f.update((1500, 900)), out[-1])  # one far sample: could be a glitch
    assert np.allclose(f.update((1500, 900)), (1500, 900))  # the second one: a saccade
