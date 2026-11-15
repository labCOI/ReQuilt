"""Tests for Steps 5-7 using a fake detector result (no model or GPU needed)."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from requilt.recovery import RecoveryStats, recover_image

TISSUE = 3
RULES = {"min_confidence": 0.5, "multipanel_iou": 0.5, "min_crop_size": 224}


def fake_result(path, boxes):
    """boxes = [(cls, conf, [x1, y1, x2, y2]), ...] -> object shaped like an Ultralytics result."""
    cls = np.array([b[0] for b in boxes], dtype=float)
    conf = np.array([b[1] for b in boxes], dtype=float)
    xyxy = np.array([b[2] for b in boxes], dtype=float).reshape(-1, 4)
    return SimpleNamespace(path=str(path), boxes=_Boxes(cls, conf, xyxy))


class _Boxes:
    """Minimal stand-in for ``ultralytics.engine.results.Boxes``."""

    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = cls, conf, xyxy

    def __len__(self):
        return len(self.cls)


def run(tmp_path, boxes, multipanel):
    img = tmp_path / "src.jpg"
    Image.new("RGB", (1000, 600), "white").save(img)
    stats = RecoveryStats()
    rows = recover_image(fake_result(img, boxes), class_index=TISSUE, is_multipanel=multipanel,
                         caption="a caption", out_dir=tmp_path, cfg_recovery=RULES, stats=stats)
    return rows, stats


def test_single_panel_keeps_one_crop_and_drops_low_confidence(tmp_path):
    boxes = [
        (TISSUE, 0.90, [0, 0, 300, 300]),      # score  81,000
        (TISSUE, 0.70, [300, 0, 900, 500]),    # score 210,000  <- best
        (TISSUE, 0.40, [0, 0, 1000, 600]),     # below 0.5 -> ignored
        (1, 0.99, [0, 0, 1000, 600]),          # not tissue -> ignored
    ]
    rows, stats = run(tmp_path, boxes, multipanel=False)
    assert len(rows) == 1 and rows[0]["box_x1"] == 300
    assert rows[0]["crop_filename"] == "src_crop.png" and rows[0]["caption"] == "a caption"
    assert stats.n_multi_panel == 0 and stats.n_with_candidate == 1


def test_multi_panel_split_overlap_and_size_filter(tmp_path):
    boxes = [
        (TISSUE, 0.9, [0, 0, 400, 300]),       # panel 1
        (TISSUE, 0.9, [500, 0, 900, 300]),     # panel 2
        (TISSUE, 0.6, [10, 10, 400, 300]),     # duplicate of panel 1 -> suppressed
        (TISSUE, 0.9, [0, 350, 200, 590]),     # 200 px wide -> too small
    ]
    rows, stats = run(tmp_path, boxes, multipanel=True)
    assert len(rows) == 2
    assert stats.n_overlaps_suppressed == 1          # duplicate of panel 1 removed
    assert all("_panel" in r["crop_filename"] for r in rows)


def test_no_detection(tmp_path):
    rows, stats = run(tmp_path, [], multipanel=False)
    assert rows == [] and stats.n_with_candidate == 0 and stats.n_crops_saved == 0
