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

| Action | Gesture | Notes |
| --- | --- | --- |
| Move cursor | open hand | anchored on the knuckles (LM 5+9), not a fingertip; 1€ filter |
| Left button | **thumb + index** touching | the button is down for exactly as long as the fingers touch |
| Click | touch and let go | the cursor stays frozen while pinching, so the click lands where you aimed |
| Drag, select, draw | touch, keep touching and move | the cursor follows after 0.25 s or 40 px of movement, button still down |
| Double click | two quick pinches | the cursor waits 0.35 s after a click, so both land on the same pixel |
| Right click | **thumb + middle** | one click per pinch + 0.4 s cooldown |
| Previous / next slide | **flick** the open hand left / right **and straight back** | Left / Right arrow; the cursor ends where it was |
| Scroll | **index + middle** extended and together, ring + pinky curled; move the hand up/down/sideways | works like a joystick: distance from the starting point sets the *speed*; dead zone |
| Zoom | "horns": **index + pinky** up, middle + ring curled; then move the hand **up** (zoom in) or **down** (zoom out) | Ctrl + wheel, one step per ~2 cm of travel, one step per frame at most; the thumb is free |
| Clutch | **fist** | `absolute`: pauses output; `--relative`: "lifts the mouse" |

- **The left button is held like a real one.** It goes down the moment thumb and index meet and
  up when they part. A held button survives a few bad frames and up to 0.5 s of lost tracking.
- **Pinch with an open hand** (like an "OK" sign): at least one other finger straight. Once
  the button is down, the other fingers may relax. With every finger folded, a pinch looks
  exactly like a fist, and a fist never clicks.
- **The flick has to come straight back.** That is what tells it from moving the cursor:
  nobody points far, fast, and straight back. It also needs to be longer and faster than a
  `--slides` swipe (about 14 cm, in a snap). `GestureConfig.flick_slides = False` turns it off.
- **A pinch is "armed" only after its fingers were seen apart.** So a hand coming back into
  view mid-pinch does not click.
- **The preview's box** is the active area: the hand inside it covers the whole screen. Its colour
  shows the mode: cyan = moving, green = button down / slide changed, yellow = scroll,
  magenta = zoom, red = clutch, orange = media mode. The panel lists the gestures.

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
photo viewers. There is no cursor in this mode, so the hand can simply move.

| Action | Gesture |
| --- | --- |
| Previous / next slide | move the open hand left / right (Left / Right arrow) |
| Next slide | a quick **pinch** thumb + index, like a click |
| Zoom in / out | **pinch** thumb + index and move the hand up / down (Ctrl + mouse wheel, one step per ~3 cm) |

**Swipes the way people make them.** The slide changes as soon as the fast part of the movement
ends: when the hand slows down or turns back. You do not have to hold the hand still before or
after. These all work:
- a careful, unhurried move (about 10 cm; 20 cm taking most of a second is fast enough);
- a swipe that comes straight back;
- a swipe started while the hand is still moving into place;
- a flick of the wrist with the arm almost still.

What does not change the slide:
- **The way back.** An opposite stroke within 0.8 s of a swipe is the hand returning. Swipe
  right, back, right, back: two slides forward.
- **Small and slow movements:** bobbing while talking, up/down beats, slowly moving the hand.
- **Raising the hand into view, lowering it out of view, or swiping off the frame edge.**
- **A fist:** make one to move the hand without swiping.

Close or far does not matter: distances are measured in palm lengths. A wide, quick sideways
gesture made while talking *is* a swipe here; `SlidesConfig.swipe_speed` and `swipe_distance`
set how big and fast a swipe has to be.

**Zoom is a pinch**, not a vertical swipe. Up/down strokes were tried first: in simulation every
up/down beat made while talking zoomed the slide. Nobody pinches by accident. While pinched,
sideways movement never changes the slide. Zoom uses Ctrl + wheel: PowerPoint's slide show
zooms with it (up to 400%), and so do browsers and PDF viewers.

The HUD shows the hand's path, the stroke as it happens and whether the hand counts as open.

## Eyes (`--eyes`)

A mouse without hands: look where the cursor should go and click with your eyes.

| Action | How |
| --- | --- |
| Move the cursor | look at the spot: the cursor jumps there and stays |
| Fine-tune the cursor | turn the head towards the target, more than 1.5° (then 3% of the screen width per degree, up to 15%) |
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
  - A shift beyond the jump radius (~3× the jitter, shown in the log and on the gaze map)
    that was not a clear jump moves the cursor after 0.6 s of looking there.
  - **Smaller shifts never move it, on purpose.** Letting them move it was tried in
    simulation: with a calibration ~140 px off, the cursor ran away from the gaze in 20 of 20
    tries, and false moves rose from 0.8 to 2–4 per minute.
- **The last centimetres are done with the head.** After a jump, turn the head towards the
  target: past a 1.5° dead zone the cursor follows the turn. The next gaze jump resets this,
  so it never adds up. Without the dead zone a head that is never quite still pulled the
  cursor away from where you looked (+36% error in simulation). `EyeConfig.head_nudge = 0`
  turns it off.
- **Head movement is compensated.** Where you look depends on the eyes and on where the head
  points and sits. The model uses MediaPipe's 3D head pose, so turning, leaning or moving closer
  while looking at the same spot keeps the cursor there.

**The gaze map** in the preview's bottom-right corner is a small picture of the screen. It
shows what the pointer does and why:
- red dots: the raw gaze estimates of the last second;
- green ring: the fixation the cursor sits on and the jump radius;
- white cross: the cursor.

How to read it:
- red dots in a tight cloud off to one side of where you look = calibration error (recalibrate);
- a wide cloud = jitter (more light);
- red dots outside the ring while the cursor stays = a shift too small for a jump (use the head).

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
| **Hand mouse:** pinches do not register | keep another finger straight (OK sign); check "pinch I" in the HUD while touching, then `GestureConfig.pinch_enter` ↑ (e.g. 0.32) |
| **Hand mouse:** the button lets go while dragging | `pinch_exit` ↑ (e.g. 0.42), `pinch_release_frames` ↑, `drag_lost_grace` ↑ |
| **Hand mouse:** the button does not let go | open the fingers further, or `pinch_exit` ↓ (above `pinch_enter`) |
| **Hand mouse:** accidental clicks | `pinch_enter` ↓, `pinch_confirm_frames` ↑ |
| **Hand mouse:** the flick does not change the slide | bring the hand straight back; `flick_distance` ↓, `flick_speed` ↓ |
| **Hand mouse:** slides change while pointing or talking | `flick_distance` ↑, `flick_speed` ↑, or `flick_slides = False` |
| **Hand mouse:** cursor stops when the thumb rests near a finger | `approach_freeze` ↓ (0 disables freezing) |
| **Hand mouse:** scroll too fast / too slow | `scroll_gain`, `scroll_deadzone`, `scroll_max_rate` |
| **Hands:** the hand is not found from far away | come closer, or more light; `TrackerConfig.zoom_crop` (smaller = more magnified, narrower) |
| **Slides:** swipes are not recognised | does the HUD say "open hand"? Then `SlidesConfig.swipe_speed` ↓ (e.g. 3), `swipe_distance` ↓ (e.g. 0.8) |
| **Slides:** slides change while you talk | `swipe_speed` ↑ (e.g. 6), `swipe_distance` ↑ (e.g. 1.5), `min_fingers = 4` |
| **Slides:** the way back changes the slide back | `return_window` ↑ (e.g. 1.2) |
| **Slides / hand mouse:** zoom too fast / too slow | `SlidesConfig.zoom_travel` / `GestureConfig.zoom_travel` (travel per step) |
| **Eyes:** the cursor lands far from where you look | recalibrate (`c`) with light on your face; check the accuracy it reports |
| **Eyes:** the cursor jumps on its own | `EyeConfig.fixation_radius` = 0.12 (fixed; default: automatic, 3× the jitter) |
| **Eyes:** small gaze shifts are ignored | by design (see above): nudge with the head; better light lowers the jitter and with it the radius |
| **Eyes:** the head nudge is too strong / too weak | `head_nudge` (0.03 = 3% of the screen width per degree; 0 = off) |
| **Eyes:** the cursor moves when the head moves a little | `head_nudge_deadzone` ↑ (e.g. 2.5 degrees) |
| **Eyes:** blinks click by accident | `blink_min` ↑ (e.g. 0.45) |
| **Head:** arrows fire too easily | `HeadConfig.yaw_threshold` / `pitch_threshold` ↑ |
| **Head:** quick turns are missed | `gesture_time` ↑ (e.g. 0.8) or the thresholds ↓ |
| **Any mode:** low FPS in the HUD (`cam`) | more light; `--probe-camera` for a faster mode (a slow Full HD mode is replaced automatically) |
| **Any mode:** the preview is black or does not appear | `python main.py --probe-camera` shows what the camera delivers; `--width 1280 --height 720` keeps one size (the automatic size change is off then); `--camera-id 1` tries another camera; close other programs that use the camera |
| **Linux:** clicks and keys get lost, the cursor moves only over some windows | the session is Wayland (the log says so): log in with an X11 / Xorg session |

## Limitations

- **Linux:**
  - **X11 sessions only.** Under Wayland the mouse and keyboard can be controlled only in some
    windows (those running through XWayland), so clicks and keys get lost. The app warns at
    start. On the login screen pick an X11 session (Ubuntu: the gear icon → "Ubuntu on Xorg").
  - MediaPipe loads `libEGL.so.1`, so minimal systems need the `libegl1` package.
  - Some cameras deliver Full HD only uncompressed, at 5–10 FPS. When the measured rate is
    below 20 FPS, the app switches to 1280×720 or 640×480 by itself and says so
    (`--width` / `--height` keep a size).
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
- **Slides:** a wide, quick sideways gesture while talking changes the slide: no detector can
  tell it from a swipe of the same size and speed. Keep a fist while gesturing, or pause.
- **Pinch:** it needs an open hand (another finger straight). With every finger folded, a pinch
  and a fist give the same landmarks.
- **Eyes:** shifts smaller than the jump radius (~3× the jitter) need the head; see above.

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
- **Hands:** synthetic hands (`tests/handgen.py`), and landmarks of real hands
  (`tests/data/real_hands.json`, MediaPipe output for public MediaPipe sample photos).
  Synthetic hands are tidy; the problems users reported only showed on the real ones.
- **Eyes and head:** synthetic faces (`tests/facegen.py`) and a simulated laptop user
  (`tests/gazesim.py`). The simulated user has real geometry: a 15.6" screen with the camera
  above it, eyes that turn inside a moving head, and webcam noise that wanders from frame to
  frame. The gaze model and the fixation filter are tuned against it.

| File | Responsibility |
| --- | --- |
| `main.py` | command line, the main loops, guaranteed release of held buttons and keys |
| `config.py` | all parameters as dataclasses + consistency validation (e.g. `pinch_enter < pinch_exit`) |
| `src/camera.py` | a thread that keeps draining the camera and holds only the newest frame; falls back to a smaller size when the camera is too slow; `--probe-camera` |
| `src/hand_tracker.py` | MediaPipe `HandLandmarker` (VIDEO mode), the second search for far hands, landmarks in isotropic pixel space |
| `src/smoother.py` | 1€ filter for vectors (one shared cutoff frequency for x and y) |
| `src/pointer.py` | active region (ROI) matched to the screen's aspect ratio, absolute/relative mapping |
| `src/gesture_engine.py` | hand features, debounced posture classification, the pinch button (press / hold / release), right click, scroll, zoom, flick |
| `src/media_gestures.py` | media mode: swipes, volume jog dial, play/pause |
| `src/swipes.py` | swipe detection for `--slides` and the hand mouse's flick: strokes cut out of the hand's motion by speed |
| `src/slide_gestures.py` | `--slides`: swipes, pinch = next slide, pinch + vertical travel = zoom |
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
- **Pinch distance with half the depth.** MediaPipe's depth (z) is its least reliable output,
  and its noise is as large as a pinch is small. Measured on real OK-sign photos, tracked by
  real MediaPipe while moving with motion blur:
  - **Touching fingers** read 0.09–0.18 palm lengths in the image, but 0.13–0.26 with full
    depth. The first version engaged below 0.22.
  - **Before:** for two of four real hands the touching pinch never engaged, and over all
    four the button was down 57% of the time it was held.
  - **Now:** depth counts at half weight; the pinch engages below 0.27 and lets go above 0.36
    (a spread open hand reads ≥ 0.58, a relaxed one 0.41–0.46, so the button comes up there).
    All four hands engage and the button is down 99% of the time, also at 6 palm lengths per
    second. A held pinch reads below 0.32 in 99% of the frames even at 9 palm lengths per
    second, and a single wild frame does not let go (it takes three in a row).
- **The left button is a button.** It used to go down only after 0.25 s of holding, and a
  shorter pinch became a synthetic click. Now it is pressed when the fingers meet and released
  when they part. The cursor stays frozen for the first moment, so a quick pinch is still one
  clean click on one pixel.
- **What a pinch looks like, from 480 real photos.** The pinching finger reaches for the
  thumb: bent, not folded (extension ratio 0.98–1.04, against < 0.9 for a folded finger), and
  another finger is straight.
  - Without these two conditions, a thumb resting on a folded finger counted as a pinch on 30%
    of the fists and 36% of the V signs (as a thumb + ring double click).
  - With them: every real pinch, and none of the fists, V signs or open hands.
  - Thumb + ring double click is off by default for the same reason; two quick pinches do it.
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
  accumulator, and zoom steps follow the hand's travel. "N ticks per frame" would scroll twice
  as fast at 60 FPS as at 30 FPS.
- **Zoom by vertical travel, not by thumb spread.** The first version zoomed with the distance
  between the thumb and the straight index finger. A thumb cannot get much closer to a straight
  index than where it starts, so zooming out stopped after a step or two; bending the index to
  get there broke the pose. Now the thumb is free and the hand's height sets the zoom, in
  detents that move by whole steps, so noise never flickers in and out.
- **Swipes: strokes cut out by speed, measured at the hand's centroid** (`src/swipes.py`).
  - A first version required rest → quick stroke → rest. In simulated natural swipes it missed
    *every* swipe that came straight back, every swipe started while the hand was still moving,
    and every wrist flick: 0 of 20 in each case.
  - Now a stroke ends as soon as the hand slows down or turns back.
  - It is tracked at the centroid of all landmarks, because a flick swings the fingers around a
    wrist that barely moves.
  - **`--slides`** takes one-way strokes, also careful ones (18 cm taking 0.9 s): 20 of 20 for
    each kind of swipe in simulation, also with blur dropouts. Raising or lowering the hand,
    slow repositioning, bobbing, up/down beats and reaching to the face do nothing.
  - **Hand mouse:** the same detector, but the stroke has to be longer and faster and the hand
    has to come straight back. In simulation fast pointing never changed the slide, whether
    the hand stayed there or came back slowly; neither did a wide talking sweep.
  - Real MediaPipe keeps the hand under webcam motion blur: at most 1–2 frames lost even for
    very fast strokes, which the detector tolerates.
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
