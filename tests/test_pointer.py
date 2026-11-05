from __future__ import annotations

import pytest

from config import FilterConfig
from src.os_controller import MoveBy, MoveTo
from src.pointer import CursorMapper, Roi

SCREEN = (1920, 1080)


@pytest.mark.parametrize("frame", [(640, 480), (1280, 720), (1920, 1080), (1080, 1920)])
def test_roi_has_screen_aspect_in_pixels(frame: tuple[int, int]) -> None:
    roi = Roi.for_screen(frame, SCREEN, width=0.5)
    width_px = (roi.x1 - roi.x0) * frame[0]
    height_px = (roi.y1 - roi.y0) * frame[1]
    assert width_px / height_px == pytest.approx(16 / 9)
    assert 0 <= roi.x0 < roi.x1 <= 1 and 0 <= roi.y0 < roi.y1 <= 1


def test_default_roi_on_720p_matches_spec_margins() -> None:
    roi = Roi.for_screen((1280, 720), SCREEN, width=0.6, center=(0.5, 0.5))
    assert (roi.x0, roi.y0, roi.x1, roi.y1) == pytest.approx((0.2, 0.2, 0.8, 0.8))


def test_sensitivity_shrinks_roi_around_its_center() -> None:
    base = Roi.for_screen((1280, 720), SCREEN, width=0.6, center=(0.5, 0.5))
    fast = Roi.for_screen((1280, 720), SCREEN, width=0.6, center=(0.5, 0.5), sensitivity=2.0)
    assert fast.x1 - fast.x0 == pytest.approx((base.x1 - base.x0) / 2)
    assert (fast.x0 + fast.x1) / 2 == pytest.approx(0.5)


def test_oversized_roi_is_kept_inside_frame() -> None:
    roi = Roi.for_screen((1280, 720), SCREEN, width=1.5, center=(0.9, 0.9))
    assert 0 <= roi.x0 < roi.x1 <= 1
    assert 0 <= roi.y0 < roi.y1 <= 1


def mapper(mode: str = "absolute", overshoot: float = 0.03) -> CursorMapper:
    roi = Roi(0.2, 0.2, 0.8, 0.8)
    return CursorMapper(roi, SCREEN, FilterConfig(), mode=mode, edge_overshoot=overshoot)


def frames_to_reach_left_edge(overshoot: float) -> int:
    m = mapper(overshoot=overshoot)
    t = 0.0
    for _ in range(60):  # rest at the centre
        m.update((0.5, 0.5), t)
        t += 1 / 60
    for frame in range(1, 600):
        move = m.update((0.2, 0.5), t)  # exactly on the ROI's left edge
        t += 1 / 60
        if isinstance(move, MoveTo) and move.x == 0:
            return frame
    return 10_000


def test_edge_overshoot_makes_screen_edges_reachable_quickly() -> None:
    with_overshoot = frames_to_reach_left_edge(0.03)
    without = frames_to_reach_left_edge(0.0)
    assert with_overshoot <= 15  # ~0.25 s
    assert without > 2 * with_overshoot  # the filter would only creep towards pixel 0


def test_absolute_mode_hits_corners_and_skips_duplicate_moves() -> None:
    m = mapper()
    moves = [m.update((0.85, 0.85), i / 60) for i in range(120)]  # beyond the bottom-right
    sent = [mv for mv in moves if mv is not None]
    assert sent[-1] == MoveTo(1919, 1079)
    assert moves[-1] is None  # position unchanged -> nothing sent


def test_frozen_absolute_keeps_filter_converged() -> None:
    m = mapper()
    t = 0.0
    for _ in range(60):
        m.update((0.5, 0.5), t)
        t += 1 / 60
    for _ in range(60):  # hand moves while frozen
        assert m.update((0.6, 0.5), t, frozen=True) is None
        t += 1 / 60
    move = m.update((0.6, 0.5), t)
    assert isinstance(move, MoveTo)
    expected_x = ((0.6 - 0.2) / 0.6 * 1.06 - 0.03) * 1919
    assert abs(move.x - expected_x) <= 2  # jumps straight to the hand, no slow glide


def test_relative_mode_accumulates_subpixel_motion() -> None:
    m = mapper(mode="relative")
    px = 0.6 / 1919  # one screen pixel in normalized frame units for this ROI
    total, t, x = 0, 0.0, 0.5
    for _ in range(200):  # 0.3 px per frame: every single frame rounds to zero
        x += 0.3 * px
        move = m.update((x, 0.5), t)
        total += move.dx if isinstance(move, MoveBy) else 0
        t += 1 / 60
    for _ in range(120):  # let the filter catch up
        move = m.update((x, 0.5), t)
        total += move.dx if isinstance(move, MoveBy) else 0
        t += 1 / 60
    assert abs(total - 60) <= 1


def test_relative_mode_discards_motion_while_frozen() -> None:
    m = mapper(mode="relative")
    t = 0.0
    for _ in range(60):
        m.update((0.5, 0.5), t)
        t += 1 / 60
    moved = 0
    for i in range(60):
        move = m.update((0.5 - 0.2 * i / 60, 0.5), t, frozen=True)
        moved += abs(move.dx) if isinstance(move, MoveBy) else 0
        t += 1 / 60
    for _ in range(30):
        move = m.update((0.3, 0.5), t)
        moved += abs(move.dx) if isinstance(move, MoveBy) else 0
        t += 1 / 60
    assert moved <= 3
