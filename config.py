"""Application configuration.

Every gesture threshold below is *dimensionless*: a distance between two landmarks divided by
the palm length (wrist -> middle-finger MCP). That is what makes the same numbers work with the
hand 30 cm or 1 m from the camera, and for small or large hands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class CameraConfig:
    device_id: int = 0
    # Full HD: more pixels per eye matters for --eyes. Whether 60 FPS is reachable at this size
    # depends on the camera: many laptop webcams give 1080p (or everything) at 30 FPS only.
    # `python main.py --probe-camera` measures what yours really delivers.
    width: int = 1920
    height: int = 1080
    fps: int = 60
    mirror: bool = True  # selfie view: hand moves right -> cursor moves right
    use_mjpg: bool = (
        True  # USB webcams need MJPG for 1080p/720p at speed (raw YUYV saturates USB 2)
    )
    buffer_size: int = 1  # minimal driver-side queue = lower latency (some backends ignore it)
    open_settings: bool = False  # Windows: open the driver's settings dialog (exposure, ...)


@dataclass
class TrackerConfig:
    model_path: str = "models/hand_landmarker.task"
    model_url: str = MODEL_URL
    num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    # Higher values make MediaPipe re-run the (slower, jumpier) palm detector more often.
    min_tracking_confidence: float = 0.5


@dataclass
class FilterConfig:
    """One Euro filter parameters. The filter runs on *screen pixels*, and beta is unit-dependent:
    filtering normalized 0..1 coordinates would need a ~2000x larger beta for the same effect."""

    min_cutoff: float = 1.0  # Hz: lower = less jitter at rest, more lag when a movement starts
    beta: float = 0.05  # speed coefficient: higher = less lag in fast moves, more jitter
    d_cutoff: float = 1.0  # Hz: cutoff of the velocity estimate


@dataclass
class PointerConfig:
    mode: str = "absolute"  # "absolute": ROI stretched over the screen; "relative": mouse-like
    roi_width: float = 0.60  # fraction of frame width; height is derived for equal x/y gain
    roi_center: tuple[float, float] = (0.50, 0.55)  # slightly low: the arm can rest lower
    sensitivity: float = 1.0  # >1 shrinks the ROI: less hand travel, but jitter is amplified too
    edge_overshoot: float = 0.03  # ROI edge maps 3% beyond the screen edge -> edges reachable
    relative_gain: float = 1.0  # extra multiplier in relative mode
    anchor_index_mcp_weight: float = 0.5  # anchor = w * LM5 + (1 - w) * LM9 (knuckles, not tips)


@dataclass
class GestureConfig:
    # Pinch hysteresis: thumb tip <-> finger tip, in palm lengths.
    pinch_enter: float = 0.22
    pinch_exit: float = 0.32
    pinch_confirm_frames: int = 2  # consecutive frames needed to engage/release (rejects glitches)
    pinch_ambiguity_margin: float = 0.03  # the closest finger must win by at least this much
    approach_freeze: float = 0.32  # freeze the cursor while a pinch is closer than this (0 = off)

    # Left button: one pinch gesture, two meanings.
    click_max_hold: float = 0.25  # s: released sooner -> click, held longer -> drag
    drag_start_px: float = 40.0  # moving this far (screen px) while pinched starts the drag early

    right_click_cooldown: float = 0.40  # s
    double_click_cooldown: float = 0.50  # s

    # Finger posture: dist(wrist, tip) / dist(wrist, pip). Rotation invariant, unlike tip.y < pip.y.
    # Measured on real MediaPipe output: straight fingers >= 1.28, curled <= 0.86.
    finger_extended_above: float = 1.12
    finger_curled_below: float = 1.00
    posture_confirm_frames: int = 3  # a new mode must be seen this many frames in a row

    # Scroll: index + middle extended and touching, ring + pinky curled. Works like a joystick:
    # displacement from where the gesture started sets the scroll *speed*.
    scroll_join_enter: float = 0.28  # index tip <-> middle tip
    scroll_join_exit: float = 0.36
    scroll_deadzone: float = 0.15  # palm lengths of travel before scrolling starts
    scroll_gain: float = 30.0  # wheel ticks per second, per palm length beyond the dead zone
    scroll_max_rate: float = 40.0  # ticks per second
    scroll_horizontal: bool = True
    scroll_invert: bool = False  # True = "natural" (touchpad-like) direction

    # Zoom: index + pinky extended, middle + ring curled; thumb-index spread drives the zoom.
    zoom_step: float = 0.12  # change of the thumb-index ratio per modifier+wheel tick
    zoom_max_steps_per_frame: int = 2

    hand_lost_grace: float = 0.20  # s: short tracking dropouts must not break a drag


@dataclass
class MediaConfig:
    """Media mode: music control instead of the mouse. Toggled by holding the "shaka" (thumb and
    pinky out, other fingers curled). It is a separate mode because in mouse mode moving the open
    hand sideways already *is* moving the cursor."""

    toggle_hold: float = 0.6  # s the shaka must be held to switch mouse <-> media mode
    # Swipe: the wrist travels this many palm lengths sideways within swipe_window.
    swipe_distance: float = 1.2
    swipe_window: float = 0.30  # s
    swipe_cooldown: float = 0.6  # s: swallows the return stroke after a swipe
    swipe_action: str = "tracks"  # "tracks": previous/next track, "arrows": Left/Right keys
    # Volume knob: turn the hand. Works like a spring-loaded jog dial: the further past the dead
    # zone, the faster the volume changes (a wrist only turns about +-70 degrees).
    knob_deadzone_deg: float = 25.0
    knob_rate: float = 0.25  # volume key presses per second, per degree beyond the dead zone
    knob_max_rate: float = 10.0  # presses per second (Windows: 1 press = 2% volume)
    knob_max_wrist_speed: float = 1.5  # palm lengths/s: a faster hand is moving, not turning
    knob_invert: bool = False  # True if "turn right" should lower the volume
    play_pause_cooldown: float = 0.8  # s between two fist (play/pause) gestures


@dataclass
class OutputConfig:
    dry_run: bool = False  # recognise gestures but never touch the real mouse/keyboard
    screen_size: tuple[int, int] | None = None  # None = auto-detect the primary monitor
    zoom_modifier: str = "ctrl"  # key held during a zoom wheel tick ("ctrl" or "cmd")


@dataclass
class HudConfig:
    enabled: bool = True
    window_name: str = "AI Virtual Mouse"
    show_landmarks: bool = True


@dataclass
class HotkeyConfig:
    """System-wide hotkeys. They work even when the HUD window has no focus (or with --no-hud),
    which is exactly when you need an emergency stop for a misbehaving virtual mouse."""

    enabled: bool = True
    toggle_pause: str = "<ctrl>+<alt>+p"
    quit: str = "<ctrl>+<alt>+q"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    pointer: PointerConfig = field(default_factory=PointerConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)

    def validate(self) -> None:
        """Fail fast on inconsistent settings instead of running a subtly broken controller."""
        g, p, f, c, m = self.gesture, self.pointer, self.filter, self.camera, self.media
        checks = [
            (0 < g.pinch_enter < g.pinch_exit, "need 0 < pinch_enter < pinch_exit"),
            (
                g.finger_curled_below < g.finger_extended_above,
                "need finger_curled_below < finger_extended_above",
            ),
            (
                0 < g.scroll_join_enter < g.scroll_join_exit,
                "need 0 < scroll_join_enter < scroll_join_exit",
            ),
            (g.pinch_confirm_frames >= 1, "pinch_confirm_frames must be >= 1"),
            (g.posture_confirm_frames >= 1, "posture_confirm_frames must be >= 1"),
            (g.zoom_step > 0, "zoom_step must be > 0"),
            (g.zoom_max_steps_per_frame >= 1, "zoom_max_steps_per_frame must be >= 1"),
            (g.scroll_deadzone >= 0 and g.scroll_gain > 0, "invalid scroll dead zone / gain"),
            (p.mode in ("absolute", "relative"), "pointer mode must be 'absolute' or 'relative'"),
            (p.sensitivity > 0, "sensitivity must be > 0"),
            (0 < p.roi_width <= 1, "roi_width must be in (0, 1]"),
            (0 <= p.anchor_index_mcp_weight <= 1, "anchor_index_mcp_weight must be in [0, 1]"),
            (f.min_cutoff > 0 and f.d_cutoff > 0, "filter cutoffs must be > 0"),
            (f.beta >= 0, "filter beta must be >= 0"),
            (c.width > 0 and c.height > 0 and c.fps > 0, "camera width/height/fps must be > 0"),
            (self.output.zoom_modifier in ("ctrl", "cmd"), "zoom_modifier must be ctrl or cmd"),
            (m.toggle_hold > 0, "media.toggle_hold must be > 0"),
            (
                m.swipe_distance > 0 and m.swipe_window > 0,
                "media swipe distance/window must be > 0",
            ),
            (m.swipe_action in ("tracks", "arrows"), "media.swipe_action must be tracks or arrows"),
            (0 <= m.knob_deadzone_deg < 90, "media.knob_deadzone_deg must be in [0, 90)"),
            (m.knob_rate > 0 and m.knob_max_rate > 0, "media knob rates must be > 0"),
        ]
        errors = [message for ok, message in checks if not ok]
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
