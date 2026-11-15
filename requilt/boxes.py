"""Box geometry and the box-selection rules of the paper (Steps 5, 6 and 8).

A *candidate* is a tuple ``(xyxy, confidence, area)`` where ``xyxy`` is
``[x1, y1, x2, y2]`` in pixels.
"""


def box_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a, box_b):
    """Intersection over union of two ``[x1, y1, x2, y2]`` boxes."""
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def score(candidate):
    """Ranking score used throughout the pipeline: confidence x area."""
    _, conf, area = candidate
    return conf * area


# --------------------------------------------------------------------------- #
# Step 5 - single-panel images
# --------------------------------------------------------------------------- #
def select_best_box(candidates):
    """Keep the single candidate with the largest confidence x area."""
    return [max(candidates, key=score)] if candidates else []


# --------------------------------------------------------------------------- #
# Step 6 - multi-panel images
# --------------------------------------------------------------------------- #
def select_panels(candidates, iou_threshold=0.5):
    """Keep every non-overlapping candidate (one per panel).

    Candidates are visited in descending confidence x area. A candidate is
    kept unless its IoU with an already-kept box exceeds ``iou_threshold``,
    which would mean both boxes cover the same panel.
    """
    kept = []
    for cand in sorted(candidates, key=score, reverse=True):
        if all(iou(cand[0], k[0]) <= iou_threshold for k in kept):
            kept.append(cand)
    return kept


# --------------------------------------------------------------------------- #
# Step 8 - localization validation
# --------------------------------------------------------------------------- #
def greedy_match(gt_boxes, pred_boxes):
    """Greedily pair ground-truth and predicted boxes in descending IoU.

    Each box is used at most once. Every pair is kept, as in the paper.

    Returns
    -------
    matched_ious : list of float   IoU of each matched pair
    n_missed     : int             ground-truth boxes left unmatched
    n_extra      : int             predicted boxes left unmatched
    """
    pairs = sorted(
        ((iou(g, p), gi, pi) for gi, g in enumerate(gt_boxes) for pi, p in enumerate(pred_boxes)),
        reverse=True,
    )
    used_gt, used_pred, matched_ious = set(), set(), []
    for value, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched_ious.append(value)
    return matched_ious, len(gt_boxes) - len(used_gt), len(pred_boxes) - len(used_pred)
