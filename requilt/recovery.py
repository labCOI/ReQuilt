"""Steps 5-7 - turn YOLOv8 detections into clean tissue crops.

For every impure image:

1. keep the *Pathology Tissue* boxes with confidence >= ``min_confidence``      (Step 5)
2. single-panel image -> keep the one box with the largest confidence x area    (Step 5)
   multi-panel image  -> keep every box that does not overlap a kept box        (Step 6)
3. crop each kept box and drop crops narrower or shorter than ``min_crop_size`` (Step 7)
4. every surviving crop inherits the caption of its source image                (Step 7)
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .boxes import box_area, select_best_box, select_panels


# columns of the per-crop metadata table
METADATA_COLUMNS = [
    "crop_filename", "source_filename", "caption", "confidence", "is_multipanel",
    "panel_index", "box_x1", "box_y1", "box_x2", "box_y2",
]


def tissue_candidates(result, class_index, min_confidence=0.5):
    """Extract ``(xyxy, confidence, area)`` tuples for the target class from an Ultralytics result."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    keep = (boxes.cls == class_index) & (boxes.conf >= min_confidence)
    return [
        (xyxy, float(conf), box_area(xyxy))
        for xyxy, conf in zip(boxes.xyxy[keep].tolist(), boxes.conf[keep].tolist())
    ]


def choose_boxes(candidates, is_multipanel, iou_threshold=0.5):
    """Apply the single-panel (Step 5) or multi-panel (Step 6) selection rule."""
    if is_multipanel:
        return select_panels(candidates, iou_threshold)
    return select_best_box(candidates)


@dataclass
class RecoveryStats:
    """Counters reported at the end of a recovery run."""

    n_images: int = 0
    n_with_candidate: int = 0        # images with >= 1 tissue box at confidence >= threshold
    n_multi_panel: int = 0           # multi-panel images processed
    n_overlaps_suppressed: int = 0   # boxes removed by the Step 6 IoU rule
    n_crops_saved: int = 0

    def summary(self):
        n = max(self.n_images, 1)
        return "\n".join([
            "===== recovery summary =====",
            f"  images processed                      : {self.n_images}",
            f"  images with >= 1 candidate region     : {self.n_with_candidate} "
            f"({self.n_with_candidate / n:.1%})",
            f"  multi-panel images processed          : {self.n_multi_panel}",
            f"  overlapping boxes suppressed (Step 6) : {self.n_overlaps_suppressed}",
            f"  tissue crops saved                    : {self.n_crops_saved}",
        ])


def recover_image(result, *, class_index, is_multipanel, caption, out_dir, cfg_recovery, stats):
    """Run Steps 5-7 on one detector result. Returns metadata rows for the saved crops."""
    image_path = Path(result.path)
    stem = image_path.stem
    stats.n_images += 1

    candidates = tissue_candidates(result, class_index, cfg_recovery["min_confidence"])
    if not candidates:
        return []
    stats.n_with_candidate += 1

    kept = choose_boxes(candidates, is_multipanel, cfg_recovery["multipanel_iou"])
    if is_multipanel:
        stats.n_multi_panel += 1
        stats.n_overlaps_suppressed += len(candidates) - len(kept)

    rows, min_size = [], cfg_recovery["min_crop_size"]
    with Image.open(image_path) as img:
        for panel_idx, (xyxy, conf, _) in enumerate(kept, start=1):
            x1, y1, x2, y2 = map(int, xyxy)
            crop = img.crop((x1, y1, x2, y2))

            # Step 7 - size filter
            if crop.width < min_size or crop.height < min_size:
                continue

            suffix = f"_panel{panel_idx}" if is_multipanel and len(kept) > 1 else ""
            crop_name = f"{stem}{suffix}_crop.png"
            crop.save(Path(out_dir) / crop_name)
            stats.n_crops_saved += 1

            rows.append({
                "crop_filename": crop_name,
                "source_filename": image_path.name,
                "caption": caption,              # caption inherited from the source image
                "confidence": round(conf, 4),
                "is_multipanel": bool(is_multipanel),
                "panel_index": panel_idx,
                "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
            })
    return rows
