# AI Virtual Mouse

Control the mouse and your music with hand gestures in front of a webcam: MediaPipe (hand
tracking) → gesture state machine → pynput (injects mouse/keyboard events into the OS). Cursor
movement, left/right/double click, drag and drop, vertical and horizontal scrolling, zoom
(Ctrl + wheel) and a clutch; in media mode previous/next track, volume and play/pause. Runs on
the CPU only.

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

The `hand_landmarker.task` model (~8 MB) is downloaded automatically into `models/` on the
first run.

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
instead (slides, photo viewers, seeking in YouTube).

## Options

| Flag | Meaning |
| --- | --- |
| `--camera-id N` | camera index (default 0) |
| `--no-hud` | no preview window (less CPU) |
| `--sensitivity X` | cursor gain; `1.5` = a third less hand travel for the same cursor travel |
| `--relative` | relative mode like a real mouse (fist = lift the mouse) |
| `--dry-run` | recognize gestures without touching the real mouse |
| `--width/--height/--fps` | requested camera mode (default 1280×720@60; the camera may deliver less) |
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
| `src/hand_tracker.py` | MediaPipe Tasks API (`HandLandmarker`, VIDEO mode), landmarks in isotropic pixel space |
| `src/smoother.py` | 1€ filter for vectors (one shared cutoff frequency for x and y) |
| `src/pointer.py` | active region (ROI) matched to the screen's aspect ratio, absolute/relative mapping |
| `src/gesture_engine.py` | hand features, debounced posture classification, hysteresis latches, clicks, scroll, zoom |
| `src/os_controller.py` | executes actions, tracks held buttons, `release_all()`, screen size |
| `src/hud.py` | preview: ROI in the state's color, pinch proximity lines, FPS, latency, metrics |

The gesture engine never calls the OS: it returns action objects (`Click`, `Press`, `Scroll`…).
That is why the whole state machine is tested on synthetic hands (`tests/handgen.py`), with no
camera and no display.

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
  size) and it runs on a single core. A lower resolution saves little time but adds jitter,
  so 1280×720 stays the default.
- The software takes roughly 15–20 ms per frame. A 30 FPS laptop webcam adds up to 33 ms of
  waiting per frame plus its exposure time, so the camera is the real latency bottleneck: a
  60 FPS camera and good lighting help more than any code change (many webcams lower their
  frame rate in dim light to expose longer).
- On a laptop, plug in or pick a balanced / best-performance power mode; power saving
  throttles the CPU.

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
- **Detection range:** MediaPipe's palm detector sees the whole frame scaled down to 192 px. A
  hand that is small in the frame (far from the camera, wide-angle lens) will not be detected;
  verified on a sample that was only detected after cropping the image to 80%.
- Keep one hand in the frame (`num_hands=1`).

## Tests

```bash
pip install -r requirements-dev.txt
```

```bash
pytest && ruff check . && ruff format --check . && mypy .
```

The tests need no camera and no display. The `HandTracker` test against real MediaPipe runs only
when the model is already in `models/`.
