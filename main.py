"""AI Virtual Mouse - entry point.

python main.py                     # webcam 0 with the HUD
python main.py --dry-run           # recognise gestures without touching the real mouse
python main.py --relative          # mouse-like relative mode: a fist "lifts" the mouse
python main.py --camera-id 1 --sensitivity 1.5 --no-hud
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
from typing import Any

import cv2

from config import AppConfig, HotkeyConfig
from src.camera import ThreadedCamera
from src.gesture_engine import EngineOutput, GestureEngine, Mode
from src.hand_tracker import HandTracker
from src.hud import Hud, LoopStats
from src.os_controller import OSController, detect_screen_size, enable_dpi_awareness
from src.pointer import CursorMapper, Roi

log = logging.getLogger("virtual_mouse")
FALLBACK_SCREEN = (1920, 1080)


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
        description="Control the mouse with hand gestures (MediaPipe + OpenCV + pynput)."
    )
    p.add_argument("--camera-id", type=int, default=0, help="webcam index (default: 0)")
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
    p.add_argument("--width", type=int, help="requested camera width (default: 1280)")
    p.add_argument("--height", type=int, help="requested camera height (default: 720)")
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
    cfg = AppConfig()
    cfg.camera.device_id = args.camera_id
    if args.width:
        cfg.camera.width = args.width
    if args.height:
        cfg.camera.height = args.height
    if args.fps:
        cfg.camera.fps = args.fps
    cfg.camera.mirror = not args.no_mirror
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


def run(cfg: AppConfig) -> int:
    enable_dpi_awareness()  # before any coordinate is queried or set (no-op off Windows)
    screen = cfg.output.screen_size or detect_screen_size()
    if screen is None:
        screen = FALLBACK_SCREEN
        log.warning("Screen size unknown, assuming %dx%d (override with --screen).", *screen)
    log.info(
        "Screen %dx%d, %s pointer%s",
        *screen,
        cfg.pointer.mode,
        " (dry run: no real input)" if cfg.output.dry_run else "",
    )

    controller = OSController.create(cfg.output.dry_run, cfg.output.zoom_modifier)
    stop = threading.Event()
    pause_toggle = threading.Event()  # set by the hotkey thread, consumed by the main loop

    with contextlib.ExitStack() as stack:
        previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: stop.set())
        stack.callback(signal.signal, signal.SIGTERM, previous_sigterm)
        listener = start_hotkeys(cfg.hotkeys, pause_toggle.set, stop.set)
        if listener is not None:
            stack.callback(listener.stop)
        if cfg.hud.enabled:
            stack.callback(cv2.destroyAllWindows)
        camera = stack.enter_context(ThreadedCamera(cfg.camera))
        tracker = stack.enter_context(HandTracker(cfg.tracker))
        # Registered last, so it runs first on *any* exit (normal, exception, Ctrl+C):
        # a mouse button or Ctrl key must never stay held after we are gone.
        stack.callback(controller.release_all)

        p = cfg.pointer
        roi = Roi.for_screen(camera.frame_size, screen, p.roi_width, p.roi_center, p.sensitivity)
        mapper = CursorMapper(roi, screen, cfg.filter, p.mode, p.edge_overshoot, p.relative_gain)
        engine = GestureEngine(cfg.gesture, mapper, p.anchor_index_mcp_weight, cfg.media)
        hud = Hud(cfg.gesture, cfg.hud.show_landmarks) if cfg.hud.enabled else None
        window = _WindowWatch(cfg.hud.window_name)
        if hud is not None:
            cv2.namedWindow(cfg.hud.window_name, cv2.WINDOW_NORMAL)
        stats = LoopStats()
        paused = False
        last_seq = -1

        while not stop.is_set():
            packet = camera.read(after_seq=last_seq, timeout=1.0)
            if packet is None:
                if not camera.is_running:
                    log.error("Camera stopped: %s", camera.error or "unknown reason")
                    return 1
                continue
            last_seq = packet.seq

            if pause_toggle.is_set():
                pause_toggle.clear()
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
                cv2.imshow(cfg.hud.window_name, packet.frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break
                if key == ord("p"):
                    pause_toggle.set()
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
