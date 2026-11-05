# AI Virtual Mouse

Hands-free control in front of a webcam: with hand gestures, with hand strokes for
presentations (`--slides`), with your eyes (`--eyes`) or with head movements (`--head`).
MediaPipe (hand / face tracking) → gesture state machines → pynput (injects mouse/keyboard
events into the OS). Hand mode: cursor movement, left/right/double click, drag and drop,
vertical and horizontal scrolling, zoom (Ctrl + wheel), a clutch, and a media mode with
previous/next track, volume and play/pause. Runs on the CPU only.

## Quick start

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
python main.py --dry-run         # first without mouse control: check the gestures in the HUD
```

```bash
python main.py                   # the real run
```

```bash
python main.py --slides          # presentations: hand strokes = slides and zoom
```

```bash
python main.py --probe-camera    # which resolutions / frame rates does the camera really deliver?
```

The MediaPipe models (`hand_landmarker.task` ~8 MB, `face_landmarker.task` ~4 MB for `--eyes`
and `--head`) are downloaded automatically into `models/` on first use.

**Emergency exit:** `Ctrl+Alt+Q` (works system-wide, also when the window has no focus and with
`--no-hud`), pause with `Ctrl+Alt+P`. In the HUD window: `q`/`ESC` quits, `p` pauses. On every
exit, including `Ctrl+C` and crashes, all held mouse buttons and keys are released.

## Gestures

| Action | Gesture | Safeguard |
| --- | --- | --- |
| Move cursor | open hand | anchored on the knuckles (LM 5+9), not a fingertip; 1€ filter |
| Left click | quick **thumb + index** pinch (< 0.25 s) | the cursor freezes while the fingers approach, so the click lands where you aimed |
| Drag | hold **thumb + index** (> 0.25 s), or move > 40 px while pinched | separating the fingers drops; a short tracking loss (< 0.2 s) does not break the drag |
| Right click | **thumb + middle** | one click per pinch + 0.4 s cooldown |
| Double click | **thumb + ring** (or two quick left clicks) | 0.5 s cooldown |
| Scroll | **index + middle** extended and together, ring + pinky curled; move the hand up/down/sideways | works like a joystick: distance from the starting point sets the *speed*; dead zone |
| Zoom | "horns": **index + pinky** up, middle + ring curled; spread/close thumb and index | Ctrl + wheel in steps (0.12 palm lengths of spread per wheel tick) |
| Clutch | **fist** | `absolute`: pauses output; `--relative`: "lifts the mouse" |

Pinch like an "OK" sign (other fingers extended). A pinch with the other fingers clenched looks
like a fist to the system.

Every pinch is "armed" only after its fingers were seen apart. That is why leaving zoom with the
fingers together, or a hand coming back into view mid-pinch, does not trigger an accidental
click.

### Media mode

Hold the **shaka** 🤙 (thumb and pinky out, other fingers curled) for 0.6 s to switch between
mouse mode and media mode; the HUD box turns orange. Media mode is a separate mode because in
mouse mode moving the open hand sideways already moves the cursor, so it cannot also mean
"next track".

| Action | Gesture | Notes |
| --- | --- | --- |
| Previous / next track | quick sideways **swipe** of the open hand, left / right | ≥ 1.2 palm lengths within 0.3 s, measured at the wrist; the hand travelling back is ignored for 0.6 s |
| Volume up / down | **turn** the open hand clockwise / counter-clockwise, like a knob | jog dial: 25° dead zone, then the further you turn, the faster the volume changes; back to neutral stops it |
| Play / pause | **fist** | once per closing of the hand |

The volume knob works like a spring-loaded jog dial, not like a real knob, because a wrist only
turns about ±70°: a position-based knob would run out of range, and turning back would undo the
change. Swipes are measured at the wrist because turning the hand swings the knuckles around
the wrist while the wrist itself stays put.

Media keys act system-wide: Spotify, YouTube in a browser and other players react without
having focus. With `MediaConfig.swipe_action = "arrows"` swipes send the Left/Right arrow keys
instead (photo viewers, seeking in YouTube). For presentations there is `--slides`.

## Slides (`--slides`)

A presenter's remote in your hand: PowerPoint, Google Slides, Keynote, PDF viewers, photo
viewers. There is no cursor in this mode; the hand only makes strokes.

| Action | Hand stroke |
| --- | --- |
| Previous / next slide | a quick stroke of the open hand left / right (Left / Right arrow) |
| Zoom in / out | a quick stroke up / down (Ctrl + mouse wheel, 2 steps) |

A stroke counts only if it:

- starts and ends with the hand still, inside the frame;
- is quick (at least ~9 cm, one palm length, in under ~0.6 s);
- runs mostly along one axis;
- is made with an open hand (at least 3 fingers straight).

The event fires ~0.2 s after the hand stops. These rules are what keep a talk free of wrong
slides:
- **Bringing the hand back** after a swipe is ignored. An opposite stroke within 0.8 s of a swipe
  counts as the way back, so swipe, return, swipe, return turns three slides forward.
- **Raising the hand** into view or **lowering it** out of view does nothing. Those strokes never
  start, or never end, with the hand resting in the frame.
- **To move the hand without swiping**, make a fist or move it slowly.
- **Hand distance:** it works the same close to the camera and far from it, because distances
  are in palm lengths.

Thresholds are in `SlidesConfig` (`config.py`). The HUD shows the stroke as it happens and the
wrist's path. Zoom uses Ctrl + wheel: PowerPoint's slide show, browsers and PDF viewers zoom
with it. In mouse mode there is also the "rock" zoom gesture (see the table above).

## Eye control (`--eyes`)

A mouse without hands: look where the cursor should go and click with your eyes.

| Action | Eye gesture |
| --- | --- |
| Move the cursor | look at the spot: the cursor jumps there and stays (or `--eyes-pointer nose`) |
| Fine-tune the cursor | turn the head slightly: ~40 px per degree, up to ~290 px |
| Left click | a deliberate blink with both eyes, 0.35–1 s |
| Double click | a second deliberate blink within 1.2 s |
| Right click | wink with the right eye (≥ 0.3 s) |
| Scroll mode on / off | wink with the left eye; while on, look above / below the screen centre to scroll |
| Pause / resume | close both eyes for 1.5 s |

**What to expect from a laptop webcam.** The camera sees each eye as 50–75 px, and the
iris moves only ~0.1 of the eye's width across the whole screen. One pixel of iris noise is ~100
screen pixels. Expect the cursor to land within about 1–3 cm of where you look. That is fine
for buttons, tabs, taskbar icons and big links, not for small text links. Therefore:

- **The cursor jumps with your eyes, then stays put.** When your eyes jump to a new spot (a
  saccade), the cursor follows within ~0.1 s, averages your gaze for 0.6 s and then freezes.
  Looking around inside that spot, including at the cursor itself, does not move it. That
  stops the classic gaze cursor that "runs away" when you look at it. Small gaze shifts (below
  the jump radius shown in the log, about 3× the measured jitter) move it only if you keep
  looking there for 0.6 s.
- **The last few centimetres are done with the head** ("MAGIC pointing"): after a jump, turn
  the head slightly towards the target and the cursor follows (`EyeConfig.head_nudge`, 0 =
  off). Every new gaze jump resets this, so it never accumulates.
- **Head movement is compensated.** Where you look depends on the eyes *and* on where the head
  points and sits. The gaze model uses MediaPipe's 3D head pose, so turning, leaning or moving
  closer while looking at the same spot keeps the cursor there. In simulation that cut the
  error under normal head movement from ~300 px (eyes only, the previous version) to ~45 px.

**Calibration (~30 s, first run).** Full screen, three phases:

1. 9 dots: look at each until it moves on, head still.
2. One dot in the middle: keep your eyes on it and turn the head where the arrow points
   (left, right, up, down). This teaches the model to compensate head movement, so do turn
   properly. If the head barely moved, the log warns you.
3. 4 check dots: the accuracy is measured on points the fit never saw. It is shown at the end
   and in the HUD (`accuracy ~45 px, jitter ~90 px`).

The result is saved to `models/gaze_calibration.json`. Redo it with `--recalibrate` or `c` in
the HUD when the accuracy is poor, after moving the laptop or the camera, or when the light
changed a lot.

**Tips:**
- Light on your face (a window or a lamp in front of you, not behind you) matters most.
- Use 1920×1080 (the default): more pixels per eye.
- Sit at your usual distance during calibration.
- For precise work there is `--eyes-pointer nose`: the cursor follows your nose, the eye
  gestures still click.

**Blinks and winks:**
- **Natural blinks** (0.1–0.3 s) are ignored. A gesture counts when the eyes open again, and
  the cursor stays frozen while they close, so a click lands where you looked.
- **Openness** is relative to each eye's own open size, learned while running, so narrow and
  wide eyes both work. Looking down lowers the lids to about 0.6–0.75 of open, safely above the
  0.4 "closed" threshold. The HUD shows both eyes' openness live. `EyeConfig.swap_winks = True`
  swaps the winks.

## Head control (`--head`)

Quick head movements press the arrow keys: slides, photo viewers, video players, lists.

| Action | Head movement |
| --- | --- |
| Left / Right arrow | a quick turn left / right (15°) |
| Up / Down arrow | a quick tilt up / nod down (12°) |
| Pause / resume | close both eyes for 1.5 s |

- **Like a swipe:** the head has to get from the neutral pose past the threshold within 0.6 s.
  Slowly turning to look at something, or shifting in your chair, presses nothing.
- Each move presses its key once. It re-arms when the head is back near neutral, so returning
  to the centre never presses the opposite arrow. Holding the head there does not repeat the key
  (`HeadConfig.repeat_delay > 0` turns repeat on).
- The angles come from MediaPipe's 3D head pose in degrees. They do not depend on how far you
  sit, and tilting the head sideways (roll) is ignored. The HUD shows them live.
  - The previous version measured the nose's offset from the eyes in the image. That gave the
    wrong direction as soon as the head was also tilted.
- The neutral pose is learned in the first half second (hold still) and follows slow posture
  changes. A quick look away is a quick turn too: pause before turning to talk to someone.

## Options

| Flag | Meaning |
| --- | --- |
| `--camera-id N` | camera index (default 0) |
| `--no-hud` | no preview window (less CPU) |
| `--sensitivity X` | cursor gain; `1.5` = a third less hand travel for the same cursor travel |
| `--relative` | relative mode like a real mouse (fist = lift the mouse) |
| `--dry-run` | recognize gestures without touching the real mouse |
| `--slides` | presentations: hand strokes left / right = slides, up / down = zoom |
| `--eyes` | eye control (see above) |
| `--eyes-pointer gaze/nose` | with `--eyes`: the cursor follows the gaze (default) or the nose |
| `--recalibrate` | with `--eyes`: redo the gaze calibration |
| `--head` | head movements press the arrow keys |
| `--width/--height/--fps` | requested camera mode (default 1920×1080@60; the camera may deliver less) |
| `--probe-camera` | measure which modes the camera really delivers, suggest the best, exit |
| `--camera-settings` | Windows: open the camera driver's settings (exposure, low-light compensation) |
| `--screen WxH` | manual screen size when auto-detection fails |
| `--model PATH`, `--no-mirror`, `--no-hotkeys`, `--log-level` | as the names say |

The remaining parameters (thresholds, filter, ROI, scroll speed) live in `config.py`.

## Architecture

```text
camera thread ──newest frame──▶ HandTracker ──HandObservation──▶ GestureEngine ──actions──▶ OSController
(src/camera.py)                 (MediaPipe)                      │  (state machine)         (pynput)
                                                                 ▼
                                                           CursorMapper
                                                (ROI → edge overshoot → 1€ filter)
```

| File | Responsibility |
| --- | --- |
| `config.py` | all parameters as dataclasses + consistency validation (e.g. `pinch_enter < pinch_exit`) |
| `src/camera.py` | a thread that keeps draining the camera and holds only the newest frame |
| `src/hand_tracker.py` | MediaPipe Tasks API (`HandLandmarker`, VIDEO mode), second search for far hands, landmarks in isotropic pixel space |
| `src/smoother.py` | 1€ filter for vectors (one shared cutoff frequency for x and y) |
| `src/pointer.py` | active region (ROI) matched to the screen's aspect ratio, absolute/relative mapping |
| `src/gesture_engine.py` | hand features, debounced posture classification, hysteresis latches, clicks, scroll, zoom |
| `src/os_controller.py` | executes actions, tracks held buttons, `release_all()`, screen size |
| `src/media_gestures.py` | media mode: swipes, volume jog dial, play/pause |
| `src/slide_gestures.py` | `--slides`: stroke segmentation (rest → quick stroke → rest), slides and zoom |
| `src/face_tracker.py` | MediaPipe `FaceLandmarker`, eye measurements (openness, iris offset), 3D head pose, gaze features |
| `src/gaze.py` | head-compensated gaze model, the 3-phase calibration procedure, the fixation filter |
| `src/calibration_ui.py` | the full-screen calibration window |
| `src/eye_engine.py` | `--eyes`: blink / wink / closure gestures, gaze (+ head nudge) or nose pointer, scroll, pause |
| `src/head_engine.py` | `--head`: head pose in degrees to arrow keys, swipe-like, with hysteresis |
| `src/hud.py` | preview: ROI in the state's color, pinch proximity lines, FPS, latency, metrics |

The gesture engine never calls the OS: it returns action objects (`Click`, `Press`, `Scroll`…).
That is why the whole state machine is tested on synthetic hands (`tests/handgen.py`), with no
camera and no display. The eye and head modes work the same way: synthetic faces
(`tests/facegen.py`) and a simulated laptop user (`tests/gazesim.py`). The simulated user has
real geometry: a 15.6" screen, the camera above it, eyes that turn inside a moving head, webcam
noise that wanders from frame to frame. The gaze model and the fixation filter are tested and
tuned against it.

## Engineering decisions

- **One measurement space.** Distances are computed in pixels `(x·W, y·H, z·W)`, not on
  normalized coordinates. In a 16:9 frame one unit of x is 1280 px but one unit of y only 720 px,
  so `hypot` on the raw values changes when the hand rotates. Only then is everything divided by
  the palm length (LM 0 → LM 9).
- **Finger extension:** `dist(wrist, tip) / dist(wrist, PIP)`. On real MediaPipe output,
  straight fingers give ≥ 1.28 and curled ones ≤ 0.86, regardless of how the hand is rotated.
  The popular `tip.y < pip.y` test stops working as soon as the hand tilts sideways.
- **Pinch arbitration:** only the finger clearly closest to the thumb may start a pinch. With
  the thumb on the index finger, the middle fingertip is close too; without this rule a single
  pinch can fire a left *and* a right click.
- **1€ filter in screen pixels**, and `beta` depends on the units. Simulation with 2–4 px of
  noise: `beta=0.05` gives ~15 px maximum lag in fast movement and ~1 px jitter at rest;
  `beta=0.01` gives 3× more lag for only half the jitter.
- **Screen edges:** positions are clamped to the ROI *before* filtering (moving the hand past
  the box leaves no backlog to "unwind"), and the ROI edge maps 3% *beyond* the screen edge.
  Without that, an exponential filter approaches the last pixel asymptotically and corners
  (close button, Start menu) are hard to hit.
- **ROI in screen proportions:** the box height follows from its width so that the x and y gain
  are equal (a 0.6×0.6 box from a 4:3 camera on a 16:9 screen would make horizontal movement 33%
  more sensitive).
- **Scroll and zoom depend on time, not on frame count:** rate × dt with a fractional
  accumulator. Scrolling "N ticks per frame" is twice as fast at 60 FPS as at 30 FPS.
- **Image landmarks, not "world" landmarks, for the fingers.** MediaPipe also returns metric 3D
  "world" landmarks, which sound more accurate. Measured on real hand photos tilted by up to
  50° in 3D, with webcam noise, they were worse:
  - 86 of 92 finger states right, against 89 of 92 for the image landmarks (x, y and relative
    z) used here;
  - a fist only just counted as a fist (margin ~0 against +0.2).
- **Far hands get a second search** in a magnified centre crop (see Limitations). It costs
  time only while the normal search finds no hand, and with no hand in view for a second it
  runs on every 3rd frame only.

## Performance

Everything runs on the CPU. The MediaPipe Python package supports its GPU delegate only on
Ubuntu, so on Windows a graphics card is not used at all, and it is not needed.

Measured per frame on one 2.1 GHz Xeon core (a recent laptop Core i7 is faster):

| Stage | 1280×720 | 640×360 |
| --- | --- | --- |
| MJPG decode (camera thread) | 3.8 ms | 0.5 ms |
| MediaPipe hand landmarker | 9.9 ms | 9.7 ms |
| gesture engine | 0.1 ms | 0.1 ms |
| HUD drawing | 0.5 ms | 0.4 ms |
| cursor jitter at rest (sensor-noise test) | 0.7–1.1 px | 1.2–1.9 px |

- MediaPipe's cost does not depend on the camera resolution (its networks have a fixed input
  size) and it runs on a single core. A lower resolution saves little time but adds jitter.
  The face landmarker takes 4–6 ms per frame while it is tracking a face.
- The software takes roughly 15–20 ms per frame. A 30 FPS laptop webcam adds up to 33 ms of
  waiting per frame plus its exposure time, so the camera is the real latency bottleneck: a
  60 FPS camera and good lighting help more than any code change (many webcams lower their
  frame rate in dim light to expose longer).
- On a laptop, plug in or pick a balanced / best-performance power mode; power saving
  throttles the CPU.

### Camera: 1920×1080 at 60 FPS

The app asks for 1920×1080 at 60 FPS, but the camera decides what it delivers. The HUD's `cam`
value is the measured rate, and the log warns when it is far below the request. Most built-in
laptop webcams deliver 30 FPS at most, and many give 1920×1080 only at 30 FPS.

- `python main.py --probe-camera` measures the real frame rate of each common mode and
  suggests the best one, e.g. `--width 1280 --height 720` when that is the only 60 FPS mode.
- Light matters: in dim light webcams expose longer and drop to 30 or even 15 FPS. On Windows,
  `--camera-settings` opens the driver's dialog, where low-light compensation or automatic
  exposure can be switched off.
- 1920×1080 costs more per frame than 1280×720 (MJPG decoding ~13 ms vs ~4 ms on one 2.1 GHz
  Xeon core, in the camera thread) but gives the eye tracker 1.5× more pixels per eye. Larger
  frames are shrunk for the preview window only.

## Tuning

Run with the HUD: the panel shows live pinch distances and finger-extension ratios, so you set
thresholds from numbers, not by feel.

| Symptom | Parameter in `config.py` |
| --- | --- |
| cursor jitters while the hand is still | `FilterConfig.min_cutoff` ↓ (e.g. 0.6) |
| cursor lags behind fast movement | `FilterConfig.beta` ↑ |
| you have to stretch your arm | `--sensitivity 1.3` or `PointerConfig.roi_width` ↓ |
| clicks do not register | `GestureConfig.pinch_enter` ↑ (e.g. 0.26); check the "pinch I" value in the HUD |
| accidental clicks | `pinch_enter` ↓, `pinch_confirm_frames` ↑ |
| cursor "stops" when the thumb rests loosely near a finger | `approach_freeze` ↓ (0 disables freezing) |
| scroll too fast / too slow | `scroll_gain`, `scroll_deadzone`, `scroll_max_rate` |

## Limitations

- **Linux:** X11 sessions only; pynput cannot inject events under Wayland. MediaPipe loads
  `libEGL.so.1`, so minimal systems need the `libegl1` package.
- **macOS:** the terminal needs the Accessibility (control) and Input Monitoring (global
  hotkeys) permissions. Ctrl + wheel does not zoom content in most macOS apps; change the
  modifier in `OutputConfig.zoom_modifier` (`"cmd"`). MediaPipe 1.2.2 has no wheels for Intel
  Macs.
- **Multiple monitors:** Windows and macOS map onto the primary monitor, X11 onto the whole
  virtual desktop. Set a different area with `--screen`.
- **Detection range:** MediaPipe's palm detector sees the whole frame scaled down to 192 px, so a
  hand that is small in the frame (far from the camera, wide-angle lens) can be missed. When the
  normal search finds nothing, the centre 60% of the frame is searched again, magnified
  (`TrackerConfig.zoom_crop`). On real hand photos at several sizes that found a hand in 48%
  more frames, almost all of them small hands. A far hand near the frame edge can still be
  missed; coming closer fixes it.
- Keep one hand in the frame (`num_hands=1`). `--eyes` and `--head` track one face, which
  must be reasonably large in the frame (sit within about 1 m of the camera).

## Tests

```bash
pip install -r requirements-dev.txt
```

```bash
pytest && ruff check . && ruff format --check . && mypy .
```

The tests need no camera and no display. The `HandTracker` test against real MediaPipe runs only
when the model is already in `models/`.
