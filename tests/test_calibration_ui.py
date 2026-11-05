"""The calibration window's loop and drawing, with the camera, the face tracker and HighGUI
replaced: a simulated laptop user looks at the dots and follows the arrows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import src.calibration_ui as ui
from config import EyeConfig
from src.camera import FramePacket
from src.face_tracker import FaceObservation
from src.gaze import CalibrationSession
from tests.facegen import SCREEN, linear_calibration, make_face
from tests.gazesim import REST, Pose, Viewer, follow_arrows


class FakeCamera:
    is_running = True

    def __init__(self) -> None:
        self.seq = -1

    def read(self, after_seq: int = -1, timeout: float = 1.0) -> FramePacket | None:
        self.seq += 1
        return FramePacket(np.zeros((8, 8, 3), np.uint8), self.seq / 30.0, self.seq)


class SimulatedUser:
    """Looks at whatever dot is shown; in the head phase turns the head as the arrow says."""

    def __init__(self, sessions: list[CalibrationSession]) -> None:
        self.sessions = sessions
        self.viewer = Viewer(iris_noise=0.008, angle_noise=0.2, rng=np.random.default_rng(0))
        self.pose: Pose = dict(REST)

    def process(self, frame: Any, t: float) -> FaceObservation | None:
        session = self.sessions[-1]
        target = session.target or (960.0, 540.0)
        if session.phase == "head":
            self.pose = follow_arrows(session, self.pose, 30.0)
        else:
            self.pose = dict(REST)
        f, d = self.viewer.features(target, **self.pose)
        return make_face(t, gaze=(f[0], f[1]), yaw=f[2], pitch=f[3], head_position=(f[4], -f[5], d))


@pytest.fixture
def no_gui(monkeypatch: pytest.MonkeyPatch) -> list[CalibrationSession]:
    """HighGUI calls become no-ops; returns the sessions the UI creates."""
    sessions: list[CalibrationSession] = []

    class Recording(CalibrationSession):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            sessions.append(self)

    monkeypatch.setattr(ui, "CalibrationSession", Recording)
    monkeypatch.setattr(ui, "RESULT_SECONDS", 0.0)
    for name in ("namedWindow", "setWindowProperty", "imshow", "destroyWindow"):
        monkeypatch.setattr(ui.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(ui.cv2, "waitKey", lambda *a: -1)
    return sessions


def test_calibration_runs_all_phases_and_returns_a_good_model(
    no_gui: list[CalibrationSession],
) -> None:
    model = ui.run_calibration(FakeCamera(), SimulatedUser(no_gui), EyeConfig(), SCREEN)
    assert model is not None
    assert no_gui[-1].done
    assert model.validation_error is not None and model.validation_error < 80
    assert model.head_range[0] > 3 and model.head_range[1] > 2  # the arrows were followed
    assert ui.quality_warnings(model) == []


def test_escape_cancels(no_gui: list[CalibrationSession], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui.cv2, "waitKey", lambda *a: 27)
    assert ui.run_calibration(FakeCamera(), SimulatedUser(no_gui), EyeConfig(), SCREEN) is None


def test_quality_is_described_and_problems_are_named() -> None:
    model = linear_calibration(noise=60.0)
    model.validation_error, model.head_range = 45.0, (6.0, 4.0)
    assert ui.describe(model) == "accuracy ~45 px (2.3% of the screen width), jitter ~60 px"
    assert ui.quality_warnings(model) == []
    model.validation_error, model.noise, model.head_range = 300.0, 200.0, (1.0, 4.0)
    warnings = " ".join(ui.quality_warnings(model))
    assert "Poor accuracy" in warnings and "High jitter" in warnings and "barely moved" in warnings


def test_drawing_shows_the_dot_the_arrow_and_a_lost_face() -> None:
    session = CalibrationSession(SCREEN, settle=0.5, sample=0.5, head_seconds=2.0)
    canvas = np.zeros((SCREEN[1], SCREEN[0], 3), np.uint8)
    ui._draw(canvas, session, face_found=True)
    x, y = (round(v) for v in session.target or (0, 0))
    assert tuple(canvas[y, x]) == (0, 0, 255)  # the red dot
    t = 0.0
    while session.phase != "head":
        session.update(t, None)
        t += 0.1
    ui._draw(canvas, session, face_found=False)
    arrow = np.all(canvas == (0, 200, 255), axis=2)
    assert arrow[:, : SCREEN[0] // 2].sum() > 500  # "turn head left": arrow left of the dot
    assert np.all(canvas[60:110, 40:600] == (0, 0, 255), axis=2).sum() > 100  # face not found
