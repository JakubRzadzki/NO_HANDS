"""CLI and main-loop wiring, with the camera and MediaPipe replaced by fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

import main
from config import AppConfig, CameraConfig, FaceConfig, TrackerConfig
from src.camera import FramePacket, ProbeResult
from src.face_tracker import FaceObservation
from src.hand_tracker import HandObservation
from src.os_controller import OSController
from tests.facegen import gaze_for, linear_calibration, make_face
from tests.handgen import OPEN, make_obs, pinch
from tests.test_os_controller import RecordingBackend


class FakeCamera:
    def __init__(self, cfg: CameraConfig, frames: int = 10**6) -> None:
        self.frame_size = (1280, 720)
        self.measured_fps = 60.0
        self.dropped_frames = 0
        self.error: str | None = None
        self.frames = frames
        self.seq = -1

    def __enter__(self) -> FakeCamera:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    @property
    def is_running(self) -> bool:
        return self.seq < self.frames

    def read(self, after_seq: int = -1, timeout: float = 1.0) -> FramePacket | None:
        if self.seq + 1 >= self.frames:
            self.seq = self.frames
            self.error = "unplugged"
            return None
        self.seq += 1
        return FramePacket(np.zeros((720, 1280, 3), np.uint8), self.seq / 60.0, self.seq)


class ScriptedTracker:
    """Open hand, then a long pinch (a drag), then the user hits Ctrl+C mid-drag."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.last_inference_ms = 5.0
        self.frames = 0

    def __enter__(self) -> ScriptedTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def process(self, frame: Any, timestamp: float) -> HandObservation | None:
        self.frames += 1
        if self.frames <= 30:
            return make_obs(timestamp, **OPEN)
        if self.frames <= 70:
            return make_obs(timestamp, **pinch("index"))
        raise KeyboardInterrupt


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> RecordingBackend:
    recording = RecordingBackend()
    monkeypatch.setattr(main, "HandTracker", ScriptedTracker)
    monkeypatch.setattr(
        main.OSController, "create", staticmethod(lambda *a, **k: OSController(recording))
    )
    return recording


def headless_config() -> AppConfig:
    cfg = AppConfig()
    cfg.hud.enabled = False
    cfg.hotkeys.enabled = False
    cfg.output.screen_size = (1920, 1080)
    return cfg


def test_ctrl_c_mid_drag_releases_the_button(
    backend: RecordingBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "ThreadedCamera", FakeCamera)
    with pytest.raises(KeyboardInterrupt):
        main.run(headless_config())
    assert [c for c in backend.calls if c[0] == "press"] == [("press", "left")]
    assert backend.calls[-1] == ("release", "left")


def test_camera_failure_exits_with_error(
    backend: RecordingBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "ThreadedCamera", lambda cfg: FakeCamera(cfg, frames=50))
    assert main.run(headless_config()) == 1
    # The camera died mid-drag (the pinch started at frame 31): the button must not stay down.
    assert ("press", "left") in backend.calls
    assert backend.calls[-1] == ("release", "left")


def test_main_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert main.main(["--sensitivity", "0"]) == 2  # invalid configuration

    def interrupted(cfg: AppConfig) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "run", interrupted)
    assert main.main([]) == 130


def test_cli_flags_map_onto_config() -> None:
    argv = [
        *("--camera-id", "2", "--no-hud", "--sensitivity", "1.5", "--relative", "--dry-run"),
        *("--screen", "2560x1440", "--width", "640", "--height", "480", "--fps", "30"),
        "--no-mirror",
    ]
    args = main.build_parser().parse_args(argv)
    cfg = main.config_from_args(args)
    assert cfg.camera.device_id == 2 and not cfg.hud.enabled
    assert cfg.pointer.sensitivity == 1.5 and cfg.pointer.mode == "relative"
    assert cfg.output.dry_run and cfg.output.screen_size == (2560, 1440)
    assert (cfg.camera.width, cfg.camera.height, cfg.camera.fps) == (640, 480, 30)
    assert not cfg.camera.mirror


@pytest.mark.parametrize("bad", ["big", "1920", "0x1080", "1x2x3"])
def test_bad_screen_argument_is_rejected(bad: str) -> None:
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["--screen", bad])


def test_default_config_is_valid_and_bad_values_are_reported() -> None:
    AppConfig().validate()
    cfg = AppConfig()
    cfg.gesture.pinch_enter = 0.5  # above pinch_exit: hysteresis inverted
    cfg.pointer.mode = "joystick"
    with pytest.raises(ValueError) as info:
        cfg.validate()
    assert "pinch_enter" in str(info.value) and "pointer mode" in str(info.value)


class ScriptedFace:
    """FaceTracker stand-in: plays a list of (frames, face kwargs) and then hits Ctrl+C."""

    script: ClassVar[list[tuple[int, dict[str, Any]]]] = []

    def __init__(self, cfg: FaceConfig, mirrored: bool = True) -> None:
        self.last_inference_ms = 4.0
        self.frames = 0

    def __enter__(self) -> ScriptedFace:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def process(self, frame: Any, timestamp: float) -> FaceObservation | None:
        self.frames += 1
        seen = 0
        for count, face in self.script:
            seen += count
            if self.frames <= seen:
                return make_face(timestamp, **face)
        raise KeyboardInterrupt


def run_face_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    script: list[tuple[int, dict[str, Any]]],
    cfg: AppConfig | None = None,
) -> None:
    monkeypatch.setattr(main, "ThreadedCamera", FakeCamera)
    monkeypatch.setattr(main, "FaceTracker", ScriptedFace)
    monkeypatch.setattr(ScriptedFace, "script", script)
    if cfg is None:
        cfg = headless_config()
        cfg.eyes.pointer = "nose"  # the gaze pointer would open the calibration window
    cfg.input_mode = mode
    with pytest.raises(KeyboardInterrupt):
        main.run(cfg)


def test_head_mode_end_to_end(backend: RecordingBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    run_face_mode(monkeypatch, "head", [(30, {}), (10, {"yaw": 22.0})])  # 60 FPS fake camera
    assert ("tap", "right") in backend.calls


def test_gaze_mode_end_to_end_with_a_saved_calibration(
    backend: RecordingBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = headless_config()
    cfg.eyes.calibration_path = str(tmp_path / "gaze.json")
    linear_calibration().save(cfg.eyes.calibration_path)  # no calibration window needed
    look = {"gaze": gaze_for(1500, 300)}
    run_face_mode(monkeypatch, "eyes", [(60, {}), (60, look)], cfg)
    moves = [c[1:] for c in backend.calls if c[0] == "move_to"]
    assert moves and moves[-1] == pytest.approx((1500, 300), abs=25)


def test_eyes_mode_end_to_end(backend: RecordingBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    closed = {"openness": (0.05, 0.05)}
    run_face_mode(monkeypatch, "eyes", [(60, {}), (30, closed), (10, {})])  # 0.5 s blink
    assert ("click", "left", 1) in backend.calls


class StrokeTracker:
    """An open hand rests, makes one quick stroke to the right, rests again, then Ctrl+C."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.last_inference_ms = 5.0
        self.frames = 0

    def __enter__(self) -> StrokeTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def process(self, frame: Any, timestamp: float) -> HandObservation | None:
        self.frames += 1
        x = 0.4 + 0.2 * min(max(self.frames - 30, 0) / 10, 1.0)  # a 0.17 s stroke at 60 FPS
        if self.frames > 80:
            raise KeyboardInterrupt
        return make_obs(timestamp, center=(x, 0.55), **OPEN)


def test_slides_mode_end_to_end(backend: RecordingBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "ThreadedCamera", FakeCamera)
    monkeypatch.setattr(main, "HandTracker", StrokeTracker)
    cfg = headless_config()
    cfg.input_mode = "slides"
    with pytest.raises(KeyboardInterrupt):
        main.run(cfg)
    assert ("tap", "right") in backend.calls
    assert not [c for c in backend.calls if c[0] in ("move_to", "move_by")]  # no cursor


def test_eye_and_head_flags_select_the_input_mode() -> None:
    parse = main.build_parser().parse_args
    eyes = main.config_from_args(parse(["--eyes", "--eyes-pointer", "nose", "--recalibrate"]))
    assert (eyes.input_mode, eyes.eyes.pointer, eyes.eyes.recalibrate) == ("eyes", "nose", True)
    assert main.config_from_args(parse(["--head"])).input_mode == "head"
    assert main.config_from_args(parse(["--slides"])).input_mode == "slides"
    assert main.config_from_args(parse([])).input_mode == "hands"
    with pytest.raises(SystemExit):
        parse(["--eyes", "--head"])  # one input mode at a time
    with pytest.raises(SystemExit):
        parse(["--slides", "--eyes"])
    assert main.main(["--eyes-pointer", "nose"]) == 2  # needs --eyes


def test_probe_camera_flag_prints_the_measured_modes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    results = [
        ProbeResult((1920, 1080, 60), (1920, 1080), 30.1, "MJPG"),
        ProbeResult((1280, 720, 60), (1280, 720), 59.8, "MJPG"),
        ProbeResult((640, 480, 60), None, 0.0, "?"),
    ]
    monkeypatch.setattr(main, "probe_camera", lambda device_id: results)
    assert main.main(["--probe-camera"]) == 0
    out = capsys.readouterr().out
    assert "1920x1080@60" in out and "not opened" in out
    assert "--width 1280 --height 720" in out  # 60 FPS beats more pixels at 30 FPS


def test_camera_settings_flag() -> None:
    args = main.build_parser().parse_args(["--camera-settings"])
    assert main.config_from_args(args).camera.open_settings
