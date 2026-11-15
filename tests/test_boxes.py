"""Unit tests for the box-selection rules (Steps 5, 6 and 8)."""

import pytest

from requilt.boxes import greedy_match, iou, select_best_box, select_panels


def test_iou_identical_and_disjoint():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # overlap 5x10 = 50, union = 100 + 100 - 50 = 150
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_single_panel_prefers_confidence_times_area():
    small_confident = ([0, 0, 10, 10], 0.99, 100)     # score  99
    large_less_conf = ([0, 0, 100, 100], 0.60, 10000)  # score 6000
    assert select_best_box([small_confident, large_less_conf]) == [large_less_conf]


def test_multi_panel_keeps_separate_panels_and_drops_overlaps():
    left = ([0, 0, 100, 100], 0.9, 10000)
    right = ([120, 0, 220, 100], 0.9, 10000)
    left_dup = ([5, 5, 100, 100], 0.6, 9025)          # IoU with `left` > 0.5
    kept = select_panels([left_dup, left, right], iou_threshold=0.5)
    assert left in kept and right in kept and left_dup not in kept


def test_greedy_match_counts():
    gt = [[0, 0, 10, 10], [50, 50, 60, 60]]
    pred = [[0, 0, 10, 10], [200, 200, 210, 210]]
    ious, missed, extra = greedy_match(gt, pred)
    # the zero-IoU pair is still a match, as in the original evaluation script
    assert sorted(ious) == [0.0, 1.0] and missed == 0 and extra == 0

    ious, missed, extra = greedy_match(gt, [[0, 0, 10, 10]])
    assert ious == [1.0] and missed == 1 and extra == 0
