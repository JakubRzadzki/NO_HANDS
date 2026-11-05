# NO_HANDS

Control your computer through a webcam without touching it. You get a mouse driven by hand
gestures, a presenter's remote in your hand, a cursor that follows your eyes, and arrow keys
from head movements. Everything runs locally on the CPU, no GPU needed. MediaPipe tracks the
hands and the face, and pynput sends the mouse and keyboard input.

| Mode | `python main.py` + | What it does | Good for |
| --- | --- | --- | --- |
| Hand mouse | (nothing) | cursor, clicks, drag, scroll, zoom; a media mode for music | everyday use instead of a mouse |
| Slides | `--slides` | hand strokes: previous / next slide, zoom in / out | presentations |
| Eyes | `--eyes` | the cursor follows your gaze, blinks click | hands busy, accessibility; large targets |
| Head | `--head` | quick head turns and nods press the arrow keys | slides, photos, lists, video |

Runs on Windows, Linux (X11) and macOS; see [Limitations](#limitations).

## Install

You need Python 3.10 or newer (tested with 3.13) and a webcam.

```bash
git clone https://github.com/JakubRzadzki/NO_HANDS.git
```

```bash
cd NO_HANDS
```

```bash
python -m venv .venv
```

Activate the environment. On Windows (PowerShell or cmd):

```powershell
.venv\Scripts\activate
```

On Linux / macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, allow local scripts once, then activate
again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The MediaPipe models are downloaded into `models/` on first use, so the first start needs
internet:
- `hand_landmarker.task`, ~8 MB;
- `face_landmarker.task`, ~4 MB, for `--eyes` and `--head`.

## First run

```bash
python main.py --probe-camera    # what the camera really delivers (~15 s), and the best mode
```

```bash
python main.py --dry-run         # gestures are recognised and shown, the mouse is not touched
```

```bash
python main.py                   # the real thing
```

**Emergency exit:** `Ctrl+Alt+Q`. **Pause:** `Ctrl+Alt+P`. Both work system-wide, even when the
window has no focus and with `--no-hud`. In the preview window, `q` / `ESC` quits and `p`
pauses. On every exit, including `Ctrl+C` and crashes, all held mouse buttons and keys are
released.

## Hand mouse (default)

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

- **Pinch like an "OK" sign**, with the other fingers extended. A pinch with the other fingers
  clenched looks like a fist to the system.
- **A pinch is "armed" only after its fingers were seen apart.** So leaving zoom with the
  fingers together, or a hand coming back into view mid-pinch, does not click.
- **The preview's box** is the active area: the hand inside it covers the whole screen. Its colour
  shows the mode: cyan = moving, green = click / drag, yellow = scroll, magenta = zoom,
  red = clutch, orange = media mode.

### Media mode

Hold the **shaka** 🤙 (thumb and pinky out, other fingers curled) for 0.6 s to switch between
the mouse and media mode. It is a separate mode because in mouse mode moving the open hand
sideways already moves the cursor, so it cannot also mean "next track".

| Action | Gesture | Notes |
| --- | --- | --- |
| Previous / next track | quick sideways **swipe** of the open hand, left / right | ≥ 1.2 palm lengths within 0.3 s, measured at the wrist; the hand travelling back is ignored for 0.6 s |
| Volume up / down | **turn** the open hand clockwise / counter-clockwise, like a knob | 25° dead zone, then the further you turn, the faster the volume changes; back to neutral stops it |
| Play / pause | **fist** | once per closing of the hand |

- **The volume knob is a spring-loaded jog dial, not a real knob.** A wrist only turns about
  ±70°: a position-based knob would run out of range, and turning back would undo the change.
- **Media keys act system-wide.** Spotify, YouTube in a browser and other players react without
  having focus.
- **Arrow keys instead of tracks:** set `MediaConfig.swipe_action = "arrows"`. For
  presentations, use `--slides` instead.

## Slides (`--slides`)

A presenter's remote in your hand, for PowerPoint, Google Slides, Keynote, PDF viewers and
photo viewers. There is no cursor in this mode: the hand only makes strokes.

| Action | Hand stroke |
| --- | --- |
| Previous / next slide | a quick stroke of the open hand left / right (Left / Right arrow) |
| Zoom in / out | a quick stroke up / down (Ctrl + mouse wheel, 2 steps) |

A stroke counts only if it:
- starts and ends with the hand still, inside the frame;
- is quick (at least one palm length, ~9 cm, in under ~0.6 s);
- runs mostly along one axis;
- is made with an open hand (at least 3 fingers straight).

The slide changes ~0.2 s after the hand stops. These rules keep a talk free of wrong slides:
- **The way back is ignored.** An opposite stroke within 0.8 s of a swipe counts as the hand
  returning. Swipe right, bring the hand back, swipe right again: two slides forward.
- **Raising the hand into view or lowering it out of view does nothing.** Those strokes never
  start, or never end, with the hand resting in the frame.
- **To move the hand without swiping**, make a fist or move it slowly.
- **Close or far does not matter:** distances are measured in palm lengths.

The HUD shows the wrist's path, the stroke as it happens and whether the hand counts as open.
Zoom uses Ctrl + wheel, which PowerPoint's slide show, browsers and PDF viewers understand. The
thresholds are in `SlidesConfig` in `config.py`.

## Eyes (`--eyes`)

A mouse without hands: look where the cursor should go and click with your eyes.

| Action | How |
| --- | --- |
| Move the cursor | look at the spot: the cursor jumps there and stays |
| Fine-tune the cursor | turn the head slightly towards the target (2% of the screen width per degree, up to 15%) |
| Left click | a deliberate blink with both eyes, 0.35–1 s |
| Double click | a second deliberate blink within 1.2 s |
| Right click | wink with the right eye (≥ 0.3 s) |
| Scroll mode on / off | wink with the left eye; while on, look above / below the screen centre to scroll |
| Pause / resume | close both eyes for 1.5 s |
| Recalibrate | `c` in the preview window |

**Calibration** takes ~30 s, full screen, on the first run and with `--recalibrate`:

1. **9 dots:** look at each until it moves on, head still.
2. **Head phase:** keep your eyes on the middle dot and turn the head where the arrow points
   (left, right, up, down). This is what later lets you move your head without the cursor
   moving, so turn properly. The result screen warns you if the head barely moved.
3. **4 check dots:** the accuracy is measured on points the fit has never seen.

The result screen and the HUD then show the accuracy and the jitter, for example `accuracy ~45 px
(2.3% of the screen width), jitter ~60 px`, and say what to fix if something is off. The
calibration is saved to `models/gaze_calibration.json`. Redo it after moving the laptop, the
camera or your chair, or when the light has changed a lot.

**What a laptop webcam can do.** The camera sees each eye as only 50–75 px wide, and across the
whole screen the iris moves by about a fifth of that. One pixel of iris noise is therefore more
than 100 screen pixels. Expect the cursor to land within 1–3 cm of where you look. That is
enough for buttons, tabs, taskbar icons and large links, not for small text links. The cursor
is built around that:

- **It jumps with your eyes, then stays put.** When your eyes jump to a new spot (a saccade),
  the cursor follows within ~0.1 s. It averages your gaze there for 0.6 s and then freezes.
  - Looking around inside that spot, including at the cursor itself, does not move it, so the
    cursor never "runs away" from your gaze.
  - A smaller shift, below the jump radius printed in the log (~3× the jitter), moves the
    cursor after 0.6 s of looking there.
- **The last centimetres are done with the head.** After a jump, turn the head slightly towards
  the target. The next gaze jump resets this, so it never adds up. `EyeConfig.head_nudge = 0`
  turns it off.
- **Head movement is compensated.** Where you look depends on the eyes and on where the head
  points and sits. The model uses MediaPipe's 3D head pose, so turning, leaning or moving closer
  while looking at the same spot keeps the cursor there.

**Tips:**
- Light on your face matters most: a window or a lamp in front of you, not behind you.
- Keep 1920×1080 (the default): more pixels per eye.
- Calibrate at your usual distance.
- For precise work use `--eyes-pointer nose`: the cursor follows your nose, the eye gestures
  still click.

**Blinks and winks:**
- **Natural blinks** (0.1–0.3 s) are ignored. A gesture counts when the eyes open again, and
  the cursor stays frozen while they close, so a click lands where you looked.
- **Openness is relative** to each eye's own open size, learned while running, so narrow and
  wide eyes both work. Looking down lowers the lids to about 0.6–0.75 of open, well above the
  0.4 "closed" threshold. The HUD shows both eyes' openness live.
- `EyeConfig.swap_winks = True` swaps the winks.

## Head (`--head`)

Quick head movements press the arrow keys: slides, photo viewers, video players, lists.

| Action | Head movement |
| --- | --- |
| Left / Right arrow | a quick turn left / right (15°) |
| Up / Down arrow | a quick tilt up / nod down (12°) |
| Pause / resume | close both eyes for 1.5 s |

- **Like a swipe:** the head has to get from the neutral pose past the threshold within 0.6 s.
  Slowly turning to look at something, or shifting in your chair, presses nothing. A *quick*
  look away is a gesture too, so pause before turning to talk to someone.
- **One press per move.** A move re-arms when the head is back near neutral, so returning to the
  centre never presses the opposite arrow. Holding the head there does not repeat the key;
  `HeadConfig.repeat_delay > 0` turns repeat on.
- **Neutral pose:** learned in the first half second (hold still), then it follows slow posture
  changes.
- **Angles in degrees** from MediaPipe's 3D head pose. They do not depend on how far you sit,
  and tilting the head sideways (roll) is ignored. The HUD shows them live.

## Options

| Flag | Meaning |
| --- | --- |
| `--slides` | presentations: hand strokes left / right = slides, up / down = zoom |
| `--eyes` | eye control |
| `--eyes-pointer gaze/nose` | with `--eyes`: the cursor follows the gaze (default) or the nose |
| `--recalibrate` | with `--eyes`: redo the gaze calibration |
| `--head` | head movements press the arrow keys |
| `--dry-run` | recognise gestures without touching the real mouse and keyboard |
| `--relative` | hand mouse: relative mode like a real mouse (fist = lift the mouse) |
| `--sensitivity X` | hand mouse: cursor gain; `1.5` = a third less hand travel for the same cursor travel |
| `--camera-id N` | camera index (default 0) |
| `--width/--height/--fps` | requested camera mode (default 1920×1080@60; the camera may deliver less) |
| `--probe-camera` | measure which modes the camera really delivers, suggest the best, exit |
| `--camera-settings` | Windows: open the camera driver's settings (exposure, low-light compensation) |
| `--no-hud` | no preview window (less CPU) |
| `--screen WxH` | manual screen size when auto-detection fails |
| `--model PATH`, `--no-mirror`, `--no-hotkeys`, `--log-level` | as the names say |

Everything else (thresholds, filters, speeds) is in `config.py`. It is checked at start, and
inconsistent values are reported by name.

## Tuning and troubleshooting

The HUD shows the measured values live: pinch distances, finger-extension ratios, the slide
stroke, eye openness, head angles, gaze accuracy. Tune from numbers, not by feel. All
parameters are in `config.py`.

| Symptom | What to change |
| --- | --- |
| **Hand mouse:** cursor jitters while the hand is still | `FilterConfig.min_cutoff` ↓ (e.g. 0.6) |
| **Hand mouse:** cursor lags behind fast movement | `FilterConfig.beta` ↑ |
| **Hand mouse:** you have to stretch your arm | `--sensitivity 1.3` or `PointerConfig.roi_width` ↓ |
| **Hand mouse:** clicks do not register | `GestureConfig.pinch_enter` ↑ (e.g. 0.26); check "pinch I" in the HUD |
| **Hand mouse:** accidental clicks | `pinch_enter` ↓, `pinch_confirm_frames` ↑ |
| **Hand mouse:** cursor stops when the thumb rests near a finger | `approach_freeze` ↓ (0 disables freezing) |
| **Hand mouse:** scroll too fast / too slow | `scroll_gain`, `scroll_deadzone`, `scroll_max_rate` |
| **Hands:** the hand is not found from far away | come closer, or more light; `TrackerConfig.zoom_crop` (smaller = more magnified, narrower) |
| **Slides:** strokes are not recognised | does the HUD say "open hand"? Then `SlidesConfig.swipe_distance` ↓ (e.g. 0.8), `move_speed` ↓ |
| **Slides:** slides change by accident | `swipe_distance` ↑, `min_fingers = 4` |
| **Slides:** the way back changes the slide back | `return_window` ↑ (e.g. 1.2) |
| **Eyes:** the cursor lands far from where you look | recalibrate (`c`) with light on your face; check the accuracy it reports |
| **Eyes:** the cursor jumps on its own | `EyeConfig.fixation_radius` = 0.12 (fixed; default: automatic, 3× the jitter) |
| **Eyes:** small gaze shifts are ignored | nudge with the head, or `fixation_dwell` ↓ |
| **Eyes:** the head nudge is too strong / too weak | `head_nudge` (0.02 = 2% of the screen width per degree; 0 = off) |
| **Eyes:** blinks click by accident | `blink_min` ↑ (e.g. 0.45) |
| **Head:** arrows fire too easily | `HeadConfig.yaw_threshold` / `pitch_threshold` ↑ |
| **Head:** quick turns are missed | `gesture_time` ↑ (e.g. 0.8) or the thresholds ↓ |
| **Any mode:** low FPS in the HUD (`cam`) | more light; `--probe-camera` for a faster mode |

## Limitations

- **Linux:** X11 sessions only; pynput cannot inject events under Wayland. MediaPipe loads
  `libEGL.so.1`, so minimal systems need the `libegl1` package.
- **macOS:**
  - The terminal needs the Accessibility (control) and Input Monitoring (global hotkeys)
    permissions.
  - Ctrl + wheel does not zoom in most macOS apps; set `OutputConfig.zoom_modifier = "cmd"`.
  - MediaPipe 1.2.2 has no wheels for Intel Macs.
- **Multiple monitors:** Windows and macOS map onto the primary monitor, X11 onto the whole
  virtual desktop. Set a different area with `--screen`.
- **Far hands:** MediaPipe's palm detector sees the whole frame scaled down to 192 px, so a hand
  that is small in the frame (far from the camera, wide-angle lens) can be missed.
  - When the normal search finds nothing, the centre 60% of the frame is searched again,
    magnified (`TrackerConfig.zoom_crop`). On real hand photos at several sizes that found a
    hand in 48% more frames.
  - A far hand near the frame edge can still be missed.
- **One hand, one face:** keep one hand in the frame (`num_hands=1`). `--eyes` and `--head`
  track one face, which must be reasonably large in the frame: sit within about 1 m.
- **Eyes:** 1–3 cm of accuracy is what a laptop webcam allows; see [Eyes](#eyes---eyes).
- **Slides:** a stroke that leaves the frame does nothing; it has to end with the hand in view.

## How it works

### Architecture

```text
                ┌─▶ HandTracker ─┬─▶ GestureEngine  (mouse, media) ──┐
camera thread ──┤   (MediaPipe)  └─▶ SlideGestures  (--slides)       │
(newest frame)  │                                                    ├─▶ OSController ─▶ OS
                └─▶ FaceTracker ─┬─▶ EyeEngine      (--eyes)         │   (pynput)
                    (MediaPipe)  └─▶ HeadArrows     (--head)       ──┘
```

Each run uses one tracker and one engine, chosen by the flag. The engines never call the OS:
they return action objects (`Click`, `Press`, `Scroll`, …). That is why the state machines are
tested without a camera and a display:
- **Hands:** synthetic hands (`tests/handgen.py`).
- **Eyes and head:** synthetic faces (`tests/facegen.py`) and a simulated laptop user
  (`tests/gazesim.py`). The simulated user has real geometry: a 15.6" screen with the camera
  above it, eyes that turn inside a moving head, and webcam noise that wanders from frame to
  frame. The gaze model and the fixation filter are tuned against it.

| File | Responsibility |
| --- | --- |
| `main.py` | command line, the main loops, guaranteed release of held buttons and keys |
| `config.py` | all parameters as dataclasses + consistency validation (e.g. `pinch_enter < pinch_exit`) |
| `src/camera.py` | a thread that keeps draining the camera and holds only the newest frame; `--probe-camera` |
| `src/hand_tracker.py` | MediaPipe `HandLandmarker` (VIDEO mode), the second search for far hands, landmarks in isotropic pixel space |
| `src/smoother.py` | 1€ filter for vectors (one shared cutoff frequency for x and y) |
| `src/pointer.py` | active region (ROI) matched to the screen's aspect ratio, absolute/relative mapping |
| `src/gesture_engine.py` | hand features, debounced posture classification, hysteresis latches, clicks, scroll, zoom |
| `src/media_gestures.py` | media mode: swipes, volume jog dial, play/pause |
| `src/slide_gestures.py` | `--slides`: stroke segmentation (rest → quick stroke → rest), slides and zoom |
| `src/face_tracker.py` | MediaPipe `FaceLandmarker`, eye measurements (openness, iris offset), 3D head pose, gaze features |
| `src/gaze.py` | head-compensated gaze model, the 3-phase calibration procedure, the fixation filter |
| `src/calibration_ui.py` | the full-screen calibration window |
| `src/eye_engine.py` | `--eyes`: blink / wink / closure gestures, gaze (+ head nudge) or nose pointer, scroll, pause |
| `src/head_engine.py` | `--head`: head pose in degrees to arrow keys, swipe-like, with hysteresis |
| `src/os_controller.py` | executes actions, tracks held buttons and keys, `release_all()`, screen size |
| `src/hud.py` | the preview window: landmarks, mode colours, live measurements, FPS, latency |

### Engineering decisions

- **One measurement space.** Distances are computed in pixels `(x·W, y·H, z·W)`, not on
  normalized coordinates. In a 16:9 frame one unit of x is 1280 px but one unit of y only 720 px,
  so `hypot` on the raw values changes when the hand rotates. Only then is everything divided by
  the palm length (LM 0 → LM 9).
- **Finger extension:** `dist(wrist, tip) / dist(wrist, PIP)`. On real MediaPipe output,
  straight fingers give ≥ 1.28 and curled ones ≤ 0.86, regardless of how the hand is rotated.
  The popular `tip.y < pip.y` test stops working as soon as the hand tilts sideways.
- **Image landmarks, not "world" landmarks, for the fingers.** MediaPipe also returns metric 3D
  "world" landmarks, which sound more accurate. Measured on real hand photos tilted by up to
  50° in 3D, with webcam noise, they were worse:
  - 86 of 92 finger states right, against 89 of 92 for the image landmarks used here;
  - a fist only just counted as a fist (margin ~0 against +0.2).
- **Pinch arbitration:** only the finger clearly closest to the thumb may start a pinch. With
  the thumb on the index finger, the middle fingertip is close too; without this rule a single
  pinch can fire a left *and* a right click.
- **1€ filter in screen pixels**, and `beta` depends on the units. Simulation with 2–4 px of
  noise:
  - `beta=0.05` gives ~15 px maximum lag in fast movement and ~1 px jitter at rest;
  - `beta=0.01` gives 3× more lag for only half the jitter.
- **Screen edges:** positions are clamped to the ROI *before* filtering, so moving the hand past
  the box leaves no backlog to "unwind". The ROI edge also maps 3% *beyond* the screen edge.
  Without that, an exponential filter approaches the last pixel asymptotically and corners
  (close button, Start menu) are hard to hit.
- **ROI in screen proportions:** the box height follows from its width, so the x and y gain
  are equal. A 0.6×0.6 box from a 4:3 camera on a 16:9 screen would make horizontal movement
  33% more sensitive.
- **Independent of the frame rate:** scroll and volume speeds are rate × dt with a fractional
  accumulator, and zoom steps follow the hand's spread. "N ticks per frame" would scroll twice
  as fast at 60 FPS as at 30 FPS.
- **Slides by stroke segmentation:** rest → quick stroke → rest, instead of "the hand moved far
  in the last 0.3 s". A moving window also fires when a hand is raised into view or lowered out
  of it. A stroke that has to start and end at rest never does.
- **Head pose from the 3D face transform.** MediaPipe fits a 3D face model to all 478 landmarks,
  so the angles are steady and in degrees. An earlier 2D measure, the nose's offset from the
  eyes in the image, gave the wrong direction as soon as the head was also tilted.
- **Gaze = eyes + head.**
  - **One head term:** the model folds the head's rotation and position into one physical term,
    where the head's straight-ahead line meets the screen. During calibration rotation and
    position move together, so fitted separately their weights came out arbitrary.
  - **Measured in simulation:** under normal head movement, the eyes alone gave ~300 px of
    error; with the head pose it is ~45 px.
  - **Iris centring:** the iris offset is centred on its calibration mean before it is scaled
    by the distance. On real faces the iris sits high between the eye corners. Uncentred,
    sitting 15 cm closer moved the cursor ~150 px.
- **Gaze fixation filter:** a jump needs samples that agree with each other *and* got there in
  a fast step. Webcam noise wanders slowly from frame to frame. Without the step test it caused
  ~30 false jumps per minute in simulation; now it is under 1.

### Performance

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

- **MediaPipe's cost does not depend on the camera resolution:** its networks have a fixed input
  size, and it runs on a single core. A lower resolution saves little time but adds jitter.
- **Face landmarker:** 4–6 ms per frame while it is tracking a face.
- **Far-hand search:** runs only while no hand is found. With no hand in view for a second, it
  runs on every 3rd frame.
- **The camera is the real latency bottleneck.** The software takes roughly 15–20 ms per frame,
  while a 30 FPS webcam adds up to 33 ms of waiting per frame plus its exposure time. A 60 FPS
  camera and good lighting help more than any code change: many webcams lower their frame rate
  in dim light to expose longer.
- **On a laptop**, plug in or pick a balanced / best-performance power mode: power saving
  throttles the CPU.

### Camera: 1920×1080 at 60 FPS

The app asks for 1920×1080 at 60 FPS, but the camera decides what it delivers. The HUD's `cam`
value is the measured rate, and the log warns when it is far below the request. Most built-in
laptop webcams deliver 30 FPS at most, and many give 1920×1080 only at 30 FPS.

- **Measure it:** `python main.py --probe-camera` measures the real frame rate of each common
  mode. It suggests the best one, e.g. `--width 1280 --height 720` when that is the only 60 FPS
  mode.
- **Light matters:** in dim light webcams expose longer and drop to 30 or even 15 FPS. On
  Windows, `--camera-settings` opens the driver's dialog, where low-light compensation or
  automatic exposure can be switched off.
- **Resolution trade-off:** 1920×1080 costs more per frame than 1280×720 (MJPG decoding ~13 ms
  vs ~4 ms on one 2.1 GHz Xeon core, in the camera thread). In return the eye tracker gets 1.5×
  more pixels per eye. Larger frames are shrunk for the preview window only.

## Development

```bash
pip install -r requirements-dev.txt
```

```bash
pytest && ruff check . && ruff format --check . && mypy .
```

The tests need no camera and no display. The `HandTracker` test against real MediaPipe runs only
when the model is already in `models/`.
