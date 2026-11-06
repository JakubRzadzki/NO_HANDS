"""The preview panels draw without errors and show what they promise (smoke tests)."""

from __future__ import annotations

import numpy as np

from config import AppConfig
from src.eye_engine import FaceMode, FaceOutput
from src.gesture_engine import EngineOutput, Mode
from src.hud import FaceHud, GazeMap, Hud, LoopStats
from tests.facegen import make_face
from tests.handgen import OPEN, make_obs


def test_gaze_map_shows_raw_gaze_fixation_and_cursor() -> None:
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = FaceOutput(
        FaceMode.EYES,
        "EYES",
        pointer=(1500.0, 300.0),
        gaze_raw=(1450.0, 330.0),
        fixation=(1480.0, 300.0),
        radius=200.0,
    )
    GazeMap((1920, 1080)).draw(frame, out)
    corner = frame[720 - 150 :, 1280 - 260 :]
    red = (corner[..., 2] > 200) & (corner[..., 1] < 80)
    white = corner.min(axis=2) > 200
    assert red.any() and white.any()  # a raw gaze dot and the cursor cross
    assert frame[:400, :800].max() == 0  # nothing outside the map's corner


def test_face_hud_with_gaze_map_draws_on_a_real_size_frame() -> None:
    cfg = AppConfig()
    hud = FaceHud("eyes", cfg.face, cfg.head, "accuracy ~45 px", screen_size=(1920, 1080))
    frame = np.zeros((1080, 1920, 3), np.uint8)
    out = FaceOutput(FaceMode.EYES, "EYES", pointer=(960.0, 540.0), gaze_raw=(970.0, 545.0))
    hud.draw(frame, make_face(), out, LoopStats(), paused=False)
    assert frame.any()


def test_slides_panel_draws() -> None:
    cfg = AppConfig()
    hud = Hud(cfg.gesture, slides=cfg.slides)
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = EngineOutput(Mode.SLIDES, "READY", trail=[(600.0, 400.0), (700.0, 405.0)])
    hud.draw(frame, None, make_obs(**OPEN), out, LoopStats(), paused=False)
    assert frame.any()
