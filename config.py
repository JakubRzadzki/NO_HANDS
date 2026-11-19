"""Application configuration.

Every gesture threshold below is *dimensionless*: a distance between two landmarks divided by
the palm length (wrist -> middle-finger MCP). That is what makes the same numbers work with the
hand 30 cm or 1 m from the camera, and for small or large hands.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
INPUT_MODES = ("hands", "slides", "eyes", "head")


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
    # A camera that cannot keep up at the requested size is useless however sharp it is: some
    # deliver Full HD only uncompressed, at 5-10 FPS (common on Linux). When the measured rate
    # is below min_fps, smaller sizes are tried and the first usable one is taken. Off when a
    # size is given on the command line.
    # Off on Windows: DirectShow does not always deliver frames after the camera is released and
    # opened again, and the window stayed empty. --width / --height turn it off everywhere.
    auto_fallback: bool = sys.platform != "win32"
    min_fps: float = 20.0


@dataclass
class TrackerConfig:
    model_path: str = "models/hand_landmarker.task"
    model_url: str = MODEL_URL
    num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    # Higher values make MediaPipe re-run the (slower, jumpier) palm detector more often.
    min_tracking_confidence: float = 0.5
    # Far (small) hands: the palm detector sees the whole frame shrunk to 192 px, where a hand
    # far from the camera is only a few pixels. When nothing is found, the frame's centre crop
    # of this size is searched as well: the detector sees it 1/zoom_crop times larger. Costs
    # time only while no hand is found by the normal search. 0 = off.
    zoom_crop: float = 0.6


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
    # Pinch: thumb tip <-> finger tip, in palm lengths, with MediaPipe's depth counted at
    # pinch_depth_weight. Measured on real hands (OK-sign photos, tracked while moving with
    # motion blur): touching fingers read 0.13-0.20, p95 0.30 in motion; spread open hands
    # read >= 0.58, a relaxed open hand 0.41-0.46.
    # The first version used full depth and 0.22 / 0.32: MediaPipe's depth is noisy at exactly
    # this scale, half of the real touching pinches read 0.22-0.26, so the pinch engaged late
    # or not at all and a held one let go.
    pinch_enter: float = 0.27
    pinch_exit: float = 0.36  # below a relaxed open hand, or the button would stay down
    pinch_depth_weight: float = 0.5  # 1 = full 3D distance, 0 = distance in the image only
    pinch_confirm_frames: int = 2  # frames in a row needed to engage (rejects glitches)
    pinch_release_frames: int = 3  # frames in a row apart before a held pinch lets go
    # The closest finger must win by this much. Not needed by the index finger: when the thumb
    # is about as close to the index as to the middle finger, that is a left click (the click
    # people mean), and a right click needs the middle finger clearly closest.
    pinch_ambiguity_margin: float = 0.03
    # A pinch is made with an open hand: the pinching finger reaches for the thumb, bent but not
    # folded (extension ratio 0.98-1.04 on real pinches, 0.56-0.91 for the fingers of real fists),
    # and at least one other finger is not folded. A thumb resting on a folded finger is not a
    # pinch. On 480 real photos the first version's rule "fired" on 30% of the fists and 36% of
    # the V signs; with these two conditions on none of them.
    pinch_min_extension: float = 0.93
    # "Not folded": the first version wanted another finger straight (> 1.12), but people relax
    # the other fingers when they click, and then the click was missed. Real fists: <= 0.99.
    pinch_other_finger_above: float = 1.05
    # Thumb + ring finger = double click. Off: in a V sign or a relaxed hand the thumb often
    # holds the ring fingertip, and a stray double click skips a slide. Two quick thumb + index
    # pinches are a double click anyway.
    ring_double_click: bool = False
    approach_freeze: float = 0.33  # freeze the cursor while a pinch is closer than this (0 = off)

    # Left button = thumb + index, held exactly as long as the fingers touch: pressed when they
    # meet, released when they part. The cursor stays where the press landed until the hand
    # clearly moves; only then does it follow with the button down (a drag), starting from
    # that spot. However long the pinch is held, a still hand makes a click.
    # (It used to start the drag after 0.25 s or 40 screen px: most real clicks last longer than
    # that, and closing the fingers alone moves the knuckles by that much on a large screen, so
    # clicks kept turning into tiny drags and selections.)
    drag_start: float = 0.35  # palm lengths of hand travel (~3 cm) with the button down
    click_settle: float = (
        0.35  # s the cursor stays put after a click: a second pinch = double click
    )

    right_click_cooldown: float = 0.40  # s
    double_click_cooldown: float = 0.50  # s

    # Finger posture: dist(wrist, tip) / dist(wrist, pip). Rotation invariant, unlike tip.y < pip.y.
    # Measured on real MediaPipe output: straight fingers >= 1.28, curled <= 0.86.
    finger_extended_above: float = 1.12
    finger_curled_below: float = 1.00
    posture_confirm_frames: int = 3  # a new mode must be seen this many frames in a row
    # The horns (zoom) need middle + ring folded. Few people fold them all the way while the
    # pinky is out, so for the horns "not straight" (below finger_extended_above) is enough.
    horns_relaxed: bool = True

    # Precision: fold fingers to slow the cursor down. Counted are the middle, ring and pinky
    # fingers (the index makes the pinch, so clicking never changes the speed):
    # all three out = full speed; one folded (e.g. the pinky) = slower; two or three folded
    # (e.g. a V: index + middle apart) = slowest.
    precision_speeds: tuple[float, float, float] = (1.0, 0.5, 0.25)
    precision_confirm_frames: int = 4  # a new speed must be seen this many frames in a row
    # Absolute mode: after slow movement the cursor is off the spot the hand points at. Back at
    # full speed the gap closes by this fraction of the hand's travel (the cursor moves up to
    # 1.5x / 0.5x as fast as the hand until it is back), so the cursor never jumps.
    precision_catch_up: float = 0.5
    # Magnifier: the other hand - the one that does not move the cursor, the left one for most
    # people - shows a V (index + middle out, ring + pinky folded), and the system's screen
    # magnifier is on for as long as it does (Windows: Magnifier, Win + Plus; macOS: Zoom,
    # Option + Cmd + 8, must be enabled in Accessibility; Linux GNOME: Super + Alt + 8). The
    # cursor hand meanwhile moves and clicks as usual, at full speed. (It used to be the V of
    # the cursor hand itself, which then also slowed the cursor to 25%.)
    magnifier: bool = True
    magnifier_confirm_frames: int = 4  # the V must be seen this many frames to turn it on
    magnifier_release_frames: int = 8  # ... and be gone this many to turn it off
    # Slides with the other hand: an open-hand swipe of the hand that does not move the cursor
    # presses Left / Right, with the --slides thresholds (SlidesConfig). The cursor hand only
    # has the strict flick (sideways and straight back), because it also points; in simulation
    # the flick missed every careful swipe, wrist flick and one-way swipe, the other hand's
    # swipe caught 19-20 of 20 of each.
    other_hand_slides: bool = True

    # Zoom with two hands: pinch thumb + index on both hands and pull them apart (zoom in) or
    # together (zoom out), like two fingers on a touchscreen. While the other hand pinches, the
    # cursor holds still and nothing clicks; a second hand that does not pinch changes nothing.
    two_hand_zoom: bool = True
    two_hand_zoom_step: float = 0.12  # one zoom step per 12% change of the hands' distance
    two_hand_grace_frames: int = 5  # the other hand may be lost this long mid-zoom
    # With a second hand in view, a pinch of the cursor hand waits this many frames longer:
    # both hands pinch at about the same time, and the first must not click meanwhile.
    two_hand_click_delay_frames: int = 4

    # Scroll: the whole hand turned sideways, its edge to the camera ("karate chop"), then moved
    # up / down. Works like a joystick: displacement from where the gesture started sets the
    # scroll *speed*. Measured as the palm's width (index knuckle -> pinky knuckle) over its
    # length: 0.62-1.12 on real hands facing the camera (0.38 for one pinching hand), about
    # 0.1-0.25 edge on. (Scroll used to be index + middle together, which the precision V kept
    # turning into.)
    scroll_palm_enter: float = 0.25  # narrower than this = turned sideways = scroll
    scroll_palm_exit: float = 0.35
    scroll_deadzone: float = 0.15  # palm lengths of travel before scrolling starts
    scroll_gain: float = 30.0  # wheel ticks per second, per palm length beyond the dead zone
    scroll_max_rate: float = 40.0  # ticks per second
    scroll_horizontal: bool = True
    scroll_invert: bool = False  # True = "natural" (touchpad-like) direction

    # Zoom: "horns" (index + pinky extended, middle + ring curled), then move the hand up to
    # zoom in, down to zoom out: one modifier+wheel tick per zoom_travel of vertical travel.
    # (Spreading the thumb, the first version, ran out of range: with the index straight the
    # thumb cannot get much closer to it than where it started, so zooming out stopped.)
    zoom_travel: float = 0.25  # palm lengths of vertical hand travel per zoom step (~2.3 cm)
    zoom_max_steps_per_frame: int = 1

    hand_lost_grace: float = 0.20  # s: a short tracking dropout changes nothing
    drag_lost_grace: float = 0.50  # s: ... and while the button is held it may last this long

    # A quick sideways flick of the open hand *that comes straight back* presses Left / Right
    # (previous / next slide). The way back is what tells it from moving the cursor. It has to
    # be longer and faster than a --slides swipe, because here the same hand also points and
    # gestures. False = swipes only in --slides.
    flick_slides: bool = True
    flick_distance: float = 1.5  # palm lengths sideways (~14 cm)
    flick_speed: float = 6.0  # palm lengths per second at the peak


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
class SlidesConfig:
    """Swipes that change slides (src/swipes.py): --slides, and the flick of the hand mouse.

    Distances in palm lengths (wrist -> middle knuckle, ~9 cm), speeds in palm lengths per
    second. Checked on simulated natural swipes (one that comes straight back, one started
    while moving into place, a wrist flick, a slow careful one) and against slow repositioning,
    small talking gestures and raising / lowering the hand."""

    swipe_distance: float = 1.0  # the hand travels at least this far sideways (~9 cm) ...
    swipe_speed: float = 4.0  # ... reaching at least this peak speed (20 cm within ~1 s)
    start_speed: float = 1.5  # faster than this starts a stroke
    axis_ratio: float = 1.5  # sideways must be this much longer than up/down (~34 deg)
    max_stroke: float = 1.0  # s: longer movements are repositioning, not swipes
    min_fingers: int = 2  # fingers not curled when the stroke starts; a moving fist never swipes
    # s: an opposite stroke starting this soon after a swipe is the hand coming back. The way back
    # starts within ~0.3 s; 0.8 (the first value) also swallowed a deliberate "previous" right
    # after a "next" (0 of 20 in simulation, 18 of 20 now).
    return_window: float = 0.45
    arm_time: float = 0.3  # s the hand must be in view before strokes count (raising it)
    # Hand mouse only: the swipe counts once the hand is this much of the way back, this soon.
    return_fraction: float = 0.5
    return_time: float = 0.6  # s
    # --slides only: pinch thumb + index and move up / down
    zoom_travel: float = 0.3  # palm lengths of vertical travel per zoom step


@dataclass
class FaceConfig:
    """Face tracking shared by --eyes and --head.

    Eye openness = eye aspect ratio (lid opening / eye width) divided by that eye's own "fully
    open" value, learned while running, so the thresholds fit narrow and wide eyes alike.
    Looking down lowers the upper lid to roughly 0.6-0.75, which is why "closed" is well
    below that."""

    model_path: str = "models/face_landmarker.task"
    model_url: str = FACE_MODEL_URL
    closed_below: float = 0.40
    open_above: float = 0.60
    pause_hold: float = 1.5  # s: both eyes closed this long = pause / resume


@dataclass
class EyeConfig:
    """--eyes: hands-free mouse. The cursor follows the gaze (or the nose), eye gestures click.

    A laptop webcam sees each eye as ~50-75 px, so one pixel of iris noise is ~100 screen px:
    expect the cursor to land within 1-3 cm of where you look. It jumps when your eyes jump,
    then stays put; a small head turn moves it the last bit (head_nudge). See src/gaze.py."""

    pointer: str = "gaze"  # "gaze": calibrated eye gaze; "nose": the cursor follows the nose
    recalibrate: bool = False  # force a new gaze calibration at start
    calibration_path: str = "models/gaze_calibration.json"
    blink_min: float = 0.35  # s: shorter closures are natural blinks and are ignored
    blink_max: float = 1.00  # s: closures in [blink_min, blink_max] are deliberate = left click
    double_click_window: float = 1.2  # s: a second deliberate blink within this = double click
    wink_min: float = 0.30  # s: one eye closed while the other stays open
    swap_winks: bool = False  # default: right wink = right click, left wink = scroll mode
    # Gaze cursor: fixation filter (see FixationFilter)
    fixation_radius: float = 0.0  # fraction of screen width; 0 = 3x the jitter of calibration
    saccade_frames: int = 2  # frames outside the radius (agreeing, after a fast step) = a jump
    fixation_settle: float = 0.6  # s: the landing point is averaged this long, then frozen
    fixation_dwell: float = 0.6  # s: looking this long outside the radius also moves the cursor
    # Gaze cursor: head nudge ("MAGIC pointing"): after the eyes put the cursor near the target,
    # turning the head moves it the rest of the way. Reset by every gaze jump.
    head_nudge: float = 0.03  # fraction of screen width per degree beyond the dead zone (0 = off)
    head_nudge_max: float = 0.15  # fraction of screen width
    head_nudge_deadzone: float = 1.5  # degrees: smaller turns (natural head drift) do nothing
    # Nose pointer: how far (fraction of frame width) the nose travels to cross the screen
    nose_roi_width: float = 0.12
    # Scroll mode: look above / below the screen centre
    scroll_deadzone: float = 0.15  # fraction of screen height around the centre
    scroll_gain: float = 40.0  # ticks/s per screen height beyond the dead zone
    scroll_max_rate: float = 15.0  # ticks/s
    # Gaze calibration: 9 dots, the head moving while looking at one dot, 4 check dots
    calib_settle: float = 0.7  # s for the eyes to land on a dot before sampling
    calib_sample: float = 1.0  # s of samples per dot
    calib_head: float = 6.0  # s of head movement (0 = skip: no head compensation)


@dataclass
class HeadConfig:
    """--head: quick head movements press the arrow keys (turn left/right, tilt up/nod down).

    Angles in degrees from MediaPipe's 3D head pose, relative to a neutral pose learned at start.
    A key fires only for a quick move: the head must get from the neutral zone past the
    threshold within gesture_time, like a swipe. Slowly turning to look at something else, or
    a posture change, never presses anything."""

    yaw_threshold: float = 15.0  # deg of head turn that presses Left / Right
    yaw_release: float = 7.0  # back within this = neutral, ready for the next move
    pitch_threshold: float = 12.0  # deg of tilt up / nod down that presses Up / Down
    pitch_release: float = 6.0
    gesture_time: float = 0.6  # s from leaving the neutral zone to crossing the threshold
    repeat_delay: float = 0.0  # s: holding the head there repeats the key (0 = no repeat)
    repeat_interval: float = 0.25  # s between repeats
    neutral_adapt: float = 0.02  # per-frame pull of the neutral pose while the head is centred


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
    input_mode: str = "hands"  # "hands", "slides" (--slides), "eyes" (--eyes) or "head" (--head)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    pointer: PointerConfig = field(default_factory=PointerConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    slides: SlidesConfig = field(default_factory=SlidesConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    eyes: EyeConfig = field(default_factory=EyeConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)

    def validate(self) -> None:
        """Fail fast on inconsistent settings instead of running a subtly broken controller."""
        g, p, f, c, m = self.gesture, self.pointer, self.filter, self.camera, self.media
        checks = [
            (0 < g.pinch_enter < g.pinch_exit, "need 0 < pinch_enter < pinch_exit"),
            (0 <= g.pinch_depth_weight <= 1, "pinch_depth_weight must be in [0, 1]"),
            (g.pinch_release_frames >= 1, "pinch_release_frames must be >= 1"),
            (g.flick_distance > 0 and g.flick_speed > 0, "flick distance / speed must be > 0"),
            (g.drag_start > 0, "drag_start must be > 0"),
            (
                g.magnifier_confirm_frames >= 1 and g.magnifier_release_frames >= 1,
                "magnifier_confirm_frames / magnifier_release_frames must be >= 1",
            ),
            (
                len(g.precision_speeds) == 3 and all(0 < v <= 1 for v in g.precision_speeds),
                "precision_speeds: three values in (0, 1]",
            ),
            (0 < g.precision_catch_up <= 1, "precision_catch_up must be in (0, 1]"),
            (g.two_hand_zoom_step > 0, "two_hand_zoom_step must be > 0"),
            (
                g.two_hand_grace_frames >= 0 and g.two_hand_click_delay_frames >= 0,
                "two_hand_grace_frames / two_hand_click_delay_frames must be >= 0",
            ),
            (
                0 <= g.hand_lost_grace <= g.drag_lost_grace,
                "need 0 <= hand_lost_grace <= drag_lost_grace",
            ),
            (
                g.finger_curled_below < g.finger_extended_above,
                "need finger_curled_below < finger_extended_above",
            ),
            (
                0 < g.scroll_palm_enter < g.scroll_palm_exit,
                "need 0 < scroll_palm_enter < scroll_palm_exit",
            ),
            (g.pinch_confirm_frames >= 1, "pinch_confirm_frames must be >= 1"),
            (g.posture_confirm_frames >= 1, "posture_confirm_frames must be >= 1"),
            (g.zoom_travel > 0, "zoom_travel must be > 0"),
            (g.zoom_max_steps_per_frame >= 1, "zoom_max_steps_per_frame must be >= 1"),
            (g.scroll_deadzone >= 0 and g.scroll_gain > 0, "invalid scroll dead zone / gain"),
            (p.mode in ("absolute", "relative"), "pointer mode must be 'absolute' or 'relative'"),
            (p.sensitivity > 0, "sensitivity must be > 0"),
            (0 < p.roi_width <= 1, "roi_width must be in (0, 1]"),
            (0 <= p.anchor_index_mcp_weight <= 1, "anchor_index_mcp_weight must be in [0, 1]"),
            (f.min_cutoff > 0 and f.d_cutoff > 0, "filter cutoffs must be > 0"),
            (f.beta >= 0, "filter beta must be >= 0"),
            (c.width > 0 and c.height > 0 and c.fps > 0, "camera width/height/fps must be > 0"),
            (c.min_fps >= 0, "camera.min_fps must be >= 0"),
            (0 <= self.tracker.zoom_crop < 1, "tracker.zoom_crop must be in [0, 1)"),
            (self.output.zoom_modifier in ("ctrl", "cmd"), "zoom_modifier must be ctrl or cmd"),
            (m.toggle_hold > 0, "media.toggle_hold must be > 0"),
            (
                m.swipe_distance > 0 and m.swipe_window > 0,
                "media swipe distance/window must be > 0",
            ),
            (m.swipe_action in ("tracks", "arrows"), "media.swipe_action must be tracks or arrows"),
            (0 <= m.knob_deadzone_deg < 90, "media.knob_deadzone_deg must be in [0, 90)"),
            (m.knob_rate > 0 and m.knob_max_rate > 0, "media knob rates must be > 0"),
            (self.input_mode in INPUT_MODES, f"input_mode must be one of {INPUT_MODES}"),
            *self._slides_checks(),
            *self._face_checks(),
        ]
        errors = [message for ok, message in checks if not ok]
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    def _slides_checks(self) -> list[tuple[bool, str]]:
        sl = self.slides
        return [
            (sl.swipe_distance > 0 and sl.zoom_travel > 0, "slides distances must be > 0"),
            (sl.axis_ratio >= 1, "slides.axis_ratio must be >= 1"),
            (
                0 < sl.start_speed < sl.swipe_speed,
                "need 0 < slides.start_speed < swipe_speed",
            ),
            (sl.max_stroke > 0, "slides.max_stroke must be > 0"),
            (0 <= sl.min_fingers <= 4, "slides.min_fingers must be in [0, 4]"),
            (sl.return_window >= 0 and sl.arm_time >= 0, "slides return_window / arm_time >= 0"),
            (
                0 < sl.return_fraction <= 1 and sl.return_time > 0,
                "slides return_fraction must be in (0, 1], return_time > 0",
            ),
        ]

    def _face_checks(self) -> list[tuple[bool, str]]:
        fc, e, h = self.face, self.eyes, self.head
        return [
            (0 < fc.closed_below < fc.open_above < 1, "need 0 < closed_below < open_above < 1"),
            (e.pointer in ("gaze", "nose"), "eyes.pointer must be gaze or nose"),
            (
                0 < e.blink_min < e.blink_max < fc.pause_hold,
                "need 0 < blink_min < blink_max < pause_hold",
            ),
            (e.wink_min > 0 and e.double_click_window > 0, "eye wink/double-click times > 0"),
            (0 <= e.fixation_radius < 1, "eyes.fixation_radius must be in [0, 1)"),
            (e.saccade_frames >= 1, "eyes.saccade_frames must be >= 1"),
            (e.fixation_settle >= 0 and e.fixation_dwell >= 0, "eyes fixation times must be >= 0"),
            (
                e.head_nudge >= 0 and e.head_nudge_max >= 0 and e.head_nudge_deadzone >= 0,
                "eyes.head_nudge values must be >= 0",
            ),
            (0 < e.nose_roi_width <= 1, "eyes.nose_roi_width must be in (0, 1]"),
            (0 <= e.scroll_deadzone < 0.5, "eyes.scroll_deadzone must be in [0, 0.5)"),
            (e.scroll_gain > 0 and e.scroll_max_rate > 0, "eye scroll gain/rate must be > 0"),
            (
                e.calib_settle >= 0 and e.calib_sample > 0 and e.calib_head >= 0,
                "invalid gaze calibration timing",
            ),
            (0 < h.yaw_release < h.yaw_threshold, "need 0 < head yaw_release < yaw_threshold"),
            (
                0 < h.pitch_release < h.pitch_threshold,
                "need 0 < head pitch_release < pitch_threshold",
            ),
            (h.gesture_time > 0, "head.gesture_time must be > 0"),
            (h.repeat_delay >= 0 and h.repeat_interval > 0, "invalid head key repeat timing"),
            (0 <= h.neutral_adapt <= 1, "head.neutral_adapt must be in [0, 1]"),
        ]
