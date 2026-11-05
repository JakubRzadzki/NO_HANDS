"""AI Virtual Mouse - entry point.

python main.py                     # hand gestures, webcam 0 with the HUD
python main.py --dry-run           # recognise gestures without touching the real mouse
python main.py --relative          # mouse-like relative mode: a fist "lifts" the mouse
python main.py --slides            # presentations: quick hand strokes = slides and zoom
python main.py --eyes              # hands-free: the cursor follows your gaze, blink to click
python main.py --eyes --eyes-pointer nose   # cursor follows the nose, blink to click
python main.py --head              # head movements press the arrow keys
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2

from config import AppConfig, HotkeyConfig
from src.calibration_ui import describe, quality_warnings, run_calibration
from src.camera import ThreadedCamera, probe_camera
from src.eye_engine import EyeEngine, FaceMode, FaceOutput
from src.face_tracker import FaceTracker
from src.gaze import GazeModel
from src.gesture_engine import EngineOutput, GestureEngine, Mode
from src.hand_tracker import HandTracker
from src.head_engine import HeadArrows
from src.hud import FaceHud, Hud, LoopStats
from src.os_controller import OSController, detect_screen_size, enable_dpi_awareness
from src.pointer import CursorMapper, Roi
from src.slide_gestures import SlideGestures

log = logging.getLogger("virtual_mouse")
FALLBACK_SCREEN = (1920, 1080)
HUD_DISPLAY_WIDTH = 1280  # larger frames are shrunk for display only; tracking uses full size


def _screen_size_arg(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 2560x1440") from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("screen size must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hands-free mouse: hand gestures (default), eyes (--eyes) or head (--head)."
    )
    modes = p.add_mutually_exclusive_group()
    modes.add_argument(
        "--slides",
        action="store_true",
        help="presentations: quick hand stroke left/right = slide, up/down = zoom in/out",
    )
    modes.add_argument(
        "--eyes", action="store_true", help="eyes only: look to move the cursor, blink to click"
    )
    modes.add_argument("--head", action="store_true", help="head movements press arrow keys")
    p.add_argument(
        "--eyes-pointer",
        choices=["gaze", "nose"],
        help="with --eyes: cursor follows the gaze (default) or the nose (more precise)",
    )
    p.add_argument("--recalibrate", action="store_true", help="with --eyes: recalibrate the gaze")
    p.add_argument("--camera-id", type=int, default=0, help="webcam index (default: 0)")
    p.add_argument(
        "--probe-camera",
        action="store_true",
        help="measure which resolutions / frame rates the camera really delivers, then exit",
    )
    p.add_argument(
        "--camera-settings",
        action="store_true",
        help="Windows: open the camera driver's settings (exposure, low-light compensation)",
    )
    p.add_argument(
        "--no-hud",
        action="store_true",
        help="no preview window (less CPU); quit with ctrl+alt+q or Ctrl+C in the terminal",
    )
    p.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="cursor gain; >1 = less hand travel for the same cursor travel (default: 1.0)",
    )
    p.add_argument(
        "--relative", action="store_true", help="relative pointer mode; a fist lifts the mouse"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="recognise gestures but never touch the real mouse"
    )
    p.add_argument("--width", type=int, help="requested camera width (default: 1920)")
    p.add_argument("--height", type=int, help="requested camera height (default: 1080)")
    p.add_argument("--fps", type=int, help="requested camera FPS (default: 60)")
    p.add_argument(
        "--screen", type=_screen_size_arg, metavar="WxH", help="override the detected screen size"
    )
    p.add_argument("--model", metavar="PATH", help="hand_landmarker.task (downloaded if missing)")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the camera image")
    p.add_argument("--no-hotkeys", action="store_true", help="disable global pause/quit hotkeys")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def config_from_args(args: argparse.Namespace) -> AppConfig:
    if (args.eyes_pointer or args.recalibrate) and not args.eyes:
        raise ValueError("--eyes-pointer and --recalibrate only work together with --eyes")
    cfg = AppConfig()
    cfg.input_mode = (
        "eyes" if args.eyes else "head" if args.head else "slides" if args.slides else "hands"
    )
    if args.eyes_pointer:
        cfg.eyes.pointer = args.eyes_pointer
    cfg.eyes.recalibrate = args.recalibrate
    cfg.camera.device_id = args.camera_id
    if args.width:
        cfg.camera.width = args.width
    if args.height:
        cfg.camera.height = args.height
    if args.fps:
        cfg.camera.fps = args.fps
    cfg.camera.mirror = not args.no_mirror
    cfg.camera.open_settings = args.camera_settings
    if args.model:
        cfg.tracker.model_path = args.model
    cfg.pointer.sensitivity = args.sensitivity
    if args.relative:
        cfg.pointer.mode = "relative"
    cfg.output.dry_run = args.dry_run
    cfg.output.screen_size = args.screen
    cfg.hud.enabled = not args.no_hud
    cfg.hotkeys.enabled = not args.no_hotkeys
    cfg.validate()
    return cfg


def start_hotkeys(
    cfg: HotkeyConfig, on_pause: Callable[[], None], on_quit: Callable[[], None]
) -> Any | None:
    """System-wide hotkeys: they work without HUD focus, i.e. exactly when the virtual mouse is
    busy clicking somewhere else. Returns the listener, or None if unavailable."""
    if not cfg.enabled:
        return None
    try:
        from pynput import keyboard

        listener = keyboard.GlobalHotKeys({cfg.toggle_pause: on_pause, cfg.quit: on_quit})
        listener.daemon = True
        listener.start()
    except Exception as exc:
        log.warning("Global hotkeys unavailable (%s); use the HUD keys or Ctrl+C.", exc)
        return None
    log.info("Global hotkeys: %s = pause/resume, %s = quit", cfg.toggle_pause, cfg.quit)
    return listener


class _WindowWatch:
    """Detects the HUD window being closed with its X button.

    Some HighGUI backends do not support WND_PROP_VISIBLE and always report -1, so "closed" is
    only trusted after the window was seen visible at least once."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._seen_visible = False

    def closed(self) -> bool:
        try:
            visible = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return self._seen_visible
        if visible >= 1:
            self._seen_visible = True
            return False
        return self._seen_visible


@dataclass
class _Session:
    """What every input mode's loop needs."""

    cfg: AppConfig
    camera: ThreadedCamera
    controller: OSController
    screen: tuple[int, int]
    stop: threading.Event
    pause_toggle: threading.Event  # set by the hotkey thread or the p key


def _show(name: str, frame: Any) -> None:
    height, width = frame.shape[:2]
    if width > HUD_DISPLAY_WIDTH:
        size = (HUD_DISPLAY_WIDTH, round(height * HUD_DISPLAY_WIDTH / width))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(name, frame)


def print_probe(device_id: int) -> int:
    print(f"Probing camera {device_id}: each mode is measured for a moment (~15 s in total)...")
    results = probe_camera(device_id)
    print(f"{'requested':>16}  {'delivered':>11}  {'measured FPS':>12}  format")
    for r in results:
        requested = f"{r.requested[0]}x{r.requested[1]}@{r.requested[2]}"
        delivered = "not opened" if r.size is None else f"{r.size[0]}x{r.size[1]}"
        print(f"{requested:>16}  {delivered:>11}  {r.fps:12.1f}  {r.fourcc}")
    working = [r for r in results if r.size is not None and r.fps > 0]
    if not working:
        print("The camera delivered no frames. Is it in use by another program?")
        return 1
    # Prefer ~60 FPS (latency), then the most pixels (precision, above all for --eyes).
    best = max(working, key=lambda r: (r.fps >= 50, r.size[0] * r.size[1] if r.size else 0))
    assert best.size is not None
    width, height = best.size
    print(f"Best: {width}x{height} at {best.fps:.0f} FPS -> --width {width} --height {height}")
    if best.fps < 50:
        print(
            "No mode reached 60 FPS: the camera is the limit. More light helps (webcams lower "
            "the frame rate to expose longer); a 60 FPS USB webcam helps most."
        )
    return 0


def _describe(cfg: AppConfig) -> str:
    if cfg.input_mode == "eyes":
        return f"eyes ({cfg.eyes.pointer} pointer)"
    if cfg.input_mode == "head":
        return "head (arrow keys)"
    if cfg.input_mode == "slides":
        return "hand strokes (slides and zoom)"
    return f"hands ({cfg.pointer.mode} pointer)"


def run(cfg: AppConfig) -> int:
    enable_dpi_awareness()  # before any coordinate is queried or set (no-op off Windows)
    screen = cfg.output.screen_size or detect_screen_size()
    if screen is None:
        screen = FALLBACK_SCREEN
        log.warning("Screen size unknown, assuming %dx%d (override with --screen).", *screen)
    log.info(
        "Screen %dx%d, input: %s%s",
        *screen,
        _describe(cfg),
        " (dry run: no real input)" if cfg.output.dry_run else "",
    )

    controller = OSController.create(cfg.output.dry_run, cfg.output.zoom_modifier)
    stop = threading.Event()
    pause_toggle = threading.Event()

    with contextlib.ExitStack() as stack:
        previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: stop.set())
        stack.callback(signal.signal, signal.SIGTERM, previous_sigterm)
        listener = start_hotkeys(cfg.hotkeys, pause_toggle.set, stop.set)
        if listener is not None:
            stack.callback(listener.stop)
        if cfg.hud.enabled:
            stack.callback(cv2.destroyAllWindows)
        camera = stack.enter_context(ThreadedCamera(cfg.camera))
        session = _Session(cfg, camera, controller, screen, stop, pause_toggle)
        # release_all is registered after the tracker, so it runs first on *any* exit (normal,
        # exception, Ctrl+C): a mouse button or Ctrl key must never stay held after we are gone.
        if cfg.input_mode in ("hands", "slides"):
            hands = stack.enter_context(HandTracker(cfg.tracker))
            stack.callback(controller.release_all)
            return _hands_loop(session, hands)
        face = stack.enter_context(FaceTracker(cfg.face, cfg.camera.mirror))
        stack.callback(controller.release_all)
        return _face_loop(session, face)


def _hands_loop(s: _Session, tracker: HandTracker) -> int:
    cfg, camera, controller = s.cfg, s.camera, s.controller
    p = cfg.pointer
    roi: Roi | None = None
    engine: GestureEngine | SlideGestures
    if cfg.input_mode == "slides":
        engine = SlideGestures(cfg.slides, cfg.gesture)  # no cursor: the hand only swipes
    else:
        roi = Roi.for_screen(camera.frame_size, s.screen, p.roi_width, p.roi_center, p.sensitivity)
        mapper = CursorMapper(roi, s.screen, cfg.filter, p.mode, p.edge_overshoot, p.relative_gain)
        engine = GestureEngine(cfg.gesture, mapper, p.anchor_index_mcp_weight, cfg.media)
    slides = cfg.slides if cfg.input_mode == "slides" else None
    hud = Hud(cfg.gesture, cfg.hud.show_landmarks, slides) if cfg.hud.enabled else None
    window = _WindowWatch(cfg.hud.window_name)
    if hud is not None:
        cv2.namedWindow(cfg.hud.window_name, cv2.WINDOW_NORMAL)
    stats = LoopStats()
    paused = False
    last_seq = -1

    while not s.stop.is_set():
        packet = camera.read(after_seq=last_seq, timeout=1.0)
        if packet is None:
            if not camera.is_running:
                log.error("Camera stopped: %s", camera.error or "unknown reason")
                return 1
            continue
        last_seq = packet.seq

        if s.pause_toggle.is_set():
            s.pause_toggle.clear()
            paused = not paused
            if paused:
                controller.apply(engine.reset())
                controller.release_all()
            log.info("Paused" if paused else "Resumed")

        if paused:
            # Keep tracking only to show the hand in the HUD; no input is generated.
            obs = tracker.process(packet.frame, packet.timestamp) if hud else None
            out = EngineOutput(Mode.NO_HAND, "PAUSED")
        else:
            obs = tracker.process(packet.frame, packet.timestamp)
            out = engine.update(obs, packet.timestamp)
            controller.apply(out.actions)
        stats.update(
            time.perf_counter(),
            packet.timestamp,
            tracker.last_inference_ms,
            camera.measured_fps,
            camera.dropped_frames,
        )

        if hud is not None:
            hud.draw(packet.frame, roi, obs, out, stats, paused)
            _show(cfg.hud.window_name, packet.frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord("p"):
                s.pause_toggle.set()
            if window.closed():
                break
    return 0


def _gaze_model(s: _Session, tracker: FaceTracker, force: bool) -> GazeModel | None:
    e = s.cfg.eyes
    if not force:
        saved = GazeModel.load(e.calibration_path, s.screen)
        if saved is not None:
            log.info(
                "Saved gaze calibration loaded (%s); --recalibrate or 'c' redoes it.",
                describe(saved),
            )
            for warning in quality_warnings(saved):
                log.warning("%s", warning)
            return saved
    log.info(
        "Gaze calibration (~30 s): follow the dots with your eyes; in the middle, keep looking "
        "at the dot and turn your head where the arrow points."
    )
    model = run_calibration(s.camera, tracker, e, s.screen, s.cfg.camera.mirror)
    if model is None:
        log.error("No gaze calibration. Try again, or use --eyes-pointer nose.")
        return None
    model.save(e.calibration_path)
    return model


def _face_loop(s: _Session, tracker: FaceTracker) -> int:
    cfg, camera, controller = s.cfg, s.camera, s.controller
    mirrored = cfg.camera.mirror
    model: GazeModel | None = None
    engine: EyeEngine | HeadArrows
    if cfg.input_mode == "eyes":
        if cfg.eyes.pointer == "gaze":
            model = _gaze_model(s, tracker, force=cfg.eyes.recalibrate)
            if model is None:
                return 1
        engine = EyeEngine(cfg.eyes, cfg.face, s.screen, cfg.filter, model, mirrored)
        if model is not None:
            log.info("Gaze cursor: jumps when the eyes move > %.0f px.", engine.fixation_radius)
    else:
        engine = HeadArrows(cfg.head, cfg.face, mirrored)
    quality = describe(model) if model is not None else None
    hud = FaceHud(cfg.input_mode, cfg.face, cfg.head, quality) if cfg.hud.enabled else None
    window = _WindowWatch(cfg.hud.window_name)
    if hud is not None:
        cv2.namedWindow(cfg.hud.window_name, cv2.WINDOW_NORMAL)
    stats = LoopStats()
    paused = False
    last_seq = -1

    while not s.stop.is_set():
        packet = camera.read(after_seq=last_seq, timeout=1.0)
        if packet is None:
            if not camera.is_running:
                log.error("Camera stopped: %s", camera.error or "unknown reason")
                return 1
            continue
        last_seq = packet.seq

        if s.pause_toggle.is_set():
            s.pause_toggle.clear()
            paused = not paused
            if paused:
                engine.reset()
                controller.release_all()
            log.info("Paused" if paused else "Resumed")

        obs = tracker.process(packet.frame, packet.timestamp) if (hud or not paused) else None
        if paused:
            out = FaceOutput(FaceMode.PAUSED, "PAUSED")
        else:
            out = engine.update(obs, packet.timestamp)
            controller.apply(out.actions)
        stats.update(
            time.perf_counter(),
            packet.timestamp,
            tracker.last_inference_ms,
            camera.measured_fps,
            camera.dropped_frames,
        )

        if hud is not None:
            hud.draw(packet.frame, obs, out, stats, paused)
            _show(cfg.hud.window_name, packet.frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                s.pause_toggle.set()
            if key == ord("c") and isinstance(engine, EyeEngine) and cfg.eyes.pointer == "gaze":
                new = _gaze_model(s, tracker, force=True)
                if new is not None:
                    engine.set_model(new)
                    hud.calibration = describe(new)
            if window.closed():
                break
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if args.probe_camera:
        return print_probe(cfg.camera.device_id)
    try:
        return run(cfg)
    except KeyboardInterrupt:
        log.info("Interrupted, all buttons released.")
        return 130
    except RuntimeError as exc:  # camera / model problems: a clear message beats a traceback
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
