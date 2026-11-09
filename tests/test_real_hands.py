"""The gesture engine on landmarks of real hands (tests/data/real_hands.json: MediaPipe output
for public MediaPipe sample photos; numbers only).

Synthetic hands are tidy. These are not: the depth is noisy, a pinching index finger is bent
rather than straight, and a thumb rests on folded fingers in fists and V signs. Both problems
reported by a real user ("the button is not held", "you have to practise") came from numbers
that only real hands show."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.gesture_engine import GestureEngine, HandFeatures, LeftState, Mode, pinch_distance
from src.hand_tracker import INDEX_TIP, THUMB_TIP, HandObservation
from src.os_controller import Action, Click, Press, Release
from tests.handgen import make_engine

HANDS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "data" / "real_hands.json").read_text(encoding="utf-8")
)


def observation(hand: dict[str, Any], t: float = 0.0) -> HandObservation:
    width, height = hand["frame"]
    return HandObservation.from_normalized(hand["norm"], (width, height), t)


def show(engine: GestureEngine, hand: dict[str, Any], t0: float, seconds: float) -> list[Action]:
    """Hold one real hand in front of the engine for a while (60 FPS)."""
    actions: list[Action] = []
    for i in range(round(seconds * 60)):
        t = t0 + i / 60
        actions += engine.update(observation(hand, t), t).actions
    return actions


def names(group: str) -> list[str]:
    return [hand["name"] for hand in HANDS[group]]


OPEN_HAND = HANDS["open"][0]


def test_real_touching_pinches_read_well_below_the_threshold() -> None:
    """The first version measured the full 3D distance and engaged below 0.22. On these four
    touching pinches that reads 0.13 - 0.24: two of them never engaged."""
    engine = make_engine()
    for hand in HANDS["pinch"]:
        obs = observation(hand)
        assert pinch_distance(obs, THUMB_TIP, INDEX_TIP, 0.5) < engine.cfg.pinch_enter - 0.07
    full_depth = [pinch_distance(observation(h), THUMB_TIP, INDEX_TIP, 1.0) for h in HANDS["pinch"]]
    assert sum(d > 0.22 for d in full_depth) == 2  # why "you have to practise"


@pytest.mark.parametrize("hand", HANDS["pinch"], ids=names("pinch"))
def test_a_real_pinch_presses_holds_and_releases(hand: dict[str, Any]) -> None:
    engine = make_engine(flick_slides=False)
    show(engine, OPEN_HAND, 0.0, 0.3)
    held = show(engine, hand, 0.3, 2.0)
    assert [a for a in held if isinstance(a, Press)] == [Press("left")]
    assert not [a for a in held if isinstance(a, Release)]
    assert engine.left_state is LeftState.DRAG  # button down for the whole 2 s
    let_go = show(engine, OPEN_HAND, 2.3, 0.3)
    assert [a for a in let_go if isinstance(a, Release)] == [Release("left")]


@pytest.mark.parametrize("group", ["fist", "v_sign", "open", "horns"])
def test_real_hands_that_are_not_pinching_never_click(group: str) -> None:
    """Fists and V signs have the thumb on a folded finger. With the first version's rule, 30%
    of real fists and 36% of real V signs counted as a pinch."""
    wrong: list[str] = []
    for hand in HANDS[group]:
        engine = make_engine(flick_slides=False)
        show(engine, OPEN_HAND, 0.0, 0.3)
        actions = show(engine, hand, 0.3, 0.6) + show(engine, OPEN_HAND, 0.9, 0.3)
        if any(isinstance(a, Press | Click) for a in actions):
            wrong.append(hand["name"])
    assert wrong == []


def test_real_fists_clutch_and_real_horns_zoom() -> None:
    def mode_for(hand: dict[str, Any]) -> Mode:
        engine = make_engine(flick_slides=False)
        show(engine, OPEN_HAND, 0.0, 0.3)
        show(engine, hand, 0.3, 0.5)
        return engine.mode

    fists = [mode_for(hand) for hand in HANDS["fist"]]
    # Seen head-on, a fist's fingers are foreshortened and a few read as "not folded": those
    # fists do not clutch, but (the test above) they never click either.
    assert fists.count(Mode.CLUTCH) >= 0.8 * len(fists)
    assert [mode_for(hand) for hand in HANDS["horns"]] == [Mode.ZOOM, Mode.ZOOM]


def test_real_finger_ratios_separate_folded_from_reaching_fingers() -> None:
    """The numbers behind pinch_min_extension (0.93): a pinching index is bent (ratio ~1.0),
    a folded finger is below it, and the other fingers of a pinching hand are straight."""
    cfg = make_engine().cfg
    pinching = [HandFeatures.from_observation(observation(h)).extension for h in HANDS["pinch"]]
    index = [next(v for k, v in e.items() if k.name == "INDEX") for e in pinching]
    assert all(cfg.pinch_min_extension + 0.04 < ratio < 1.1 for ratio in index)
    assert all(max(e.values()) > 1.25 for e in pinching)  # an open hand
    folded = []
    for hand in HANDS["fist"]:
        extension = HandFeatures.from_observation(observation(hand)).extension
        folded += [v for k, v in extension.items() if k.name in ("MIDDLE", "RING", "PINKY")]
    assert sorted(folded)[int(0.95 * len(folded))] < cfg.pinch_min_extension
