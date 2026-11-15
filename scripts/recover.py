"""Run the ReQuilt recovery stage (Steps 5-7) on any folder of images.

Example
-------
python scripts/recover.py \
    --weights weights/requilt_yolov8n.pt \
    --images  path/to/impure_images \
    --out     outputs/recovered \
    --impurity-csv outputs/quilt_impurity_scores.csv \
    --captions path/to/quilt_1M_lookup.csv

``--impurity-csv`` supplies the QuiltCleaner "Multi-Panel Image" flag; without
it every image is processed as single-panel. ``--captions`` attaches the
original Quilt-1M caption to each crop.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from requilt import load_config  # noqa: E402
from requilt.data import MULTI_PANEL, load_caption_map  # noqa: E402
from requilt.recovery import METADATA_COLUMNS, RecoveryStats, recover_image  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="fine-tuned YOLOv8 weights (.pt)")
    p.add_argument("--images", required=True, help="folder of images to clean")
    p.add_argument("--out", required=True, help="output folder for crops + metadata.csv")
    p.add_argument("--impurity-csv", help="table with 'Filename' and 'Multi-Panel Image' columns")
    p.add_argument("--captions", help="Quilt-1M caption lookup CSV")
    p.add_argument("--config", default=None, help="config file (default: configs/config.yaml)")
    return p.parse_args()


def main():
    from ultralytics import YOLO

    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    rec, thr = cfg.recovery, cfg.impurity.threshold

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    multipanel = {}
    if args.impurity_csv:
        df = pd.read_csv(args.impurity_csv, usecols=["Filename", MULTI_PANEL])
        multipanel = dict(zip(df["Filename"], df[MULTI_PANEL] >= thr))
    captions = load_caption_map(args.captions) if args.captions else {}

    model = YOLO(args.weights)
    class_index = {v: k for k, v in model.names.items()}[rec.target_class]

    stats, rows = RecoveryStats(), []
    for result in model.predict(source=args.images, imgsz=cfg.train.img_size,
                                 conf=rec.predict_conf, stream=True, verbose=False):
        name = Path(result.path).name
        rows += recover_image(
            result,
            class_index=class_index,
            is_multipanel=multipanel.get(name, False),
            caption=captions.get(name),
            out_dir=out_dir,
            cfg_recovery=rec,
            stats=stats,
        )

    pd.DataFrame(rows, columns=METADATA_COLUMNS).to_csv(out_dir / "metadata.csv", index=False)
    print(stats.summary())
    print(f"\ncrops and metadata.csv written to {out_dir}")


if __name__ == "__main__":
    main()
