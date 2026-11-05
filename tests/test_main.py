"""CLI and main-loop wiring, with the camera and MediaPipe replaced by fakes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import main
from config import AppConfig, CameraConfig, TrackerConfig
from src.camera import FramePacket
from src.hand_tracker import HandObservation
from src.os_controller import OSController
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
