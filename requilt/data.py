"""Readers for the Quilt-1M tables and the YOLO annotation files."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# The eight impurity categories defined by QuiltCleaner (Aubreville et al., 2024).
IMPURITY_COLUMNS = [
    "Persons/Photos",
    "Desktop/Windows/SlideViewer",
    "Text/Logo in Image",
    "Arrow/Annotation",
    "Image Perspective/Quality",
    "Additional (On-Slide) Overview",
    "Additional Control Elements",
    "Multi-Panel Image",
]
ANY_IMPAIRMENT = "Any impairment"
MULTI_PANEL = "Multi-Panel Image"


# --------------------------------------------------------------------------- #
# Quilt-1M tables
# --------------------------------------------------------------------------- #
def load_impurity_table(csv_path):
    """Load the per-image impurity table built in notebook 00 (``quilt_impurity_scores.csv``)."""
    return pd.read_csv(csv_path)


def load_caption_map(caption_lookup_csv):
    """Return ``{filename: caption}`` from the Quilt-1M caption lookup table.

    The lookup has one row per (image, caption) pair; when an image has
    several rows the last caption is kept, exactly as in the original run.
    """
    captions = pd.read_csv(caption_lookup_csv, usecols=["caption", "image_path"])
    filenames = captions["image_path"].map(os.path.basename)
    return dict(zip(filenames, captions["caption"]))


def add_dominant_impurity(df):
    """Add a ``dominant_impurity`` column: the category with the highest score."""
    df = df.copy()
    df["dominant_impurity"] = df[IMPURITY_COLUMNS].idxmax(axis=1)
    return df


def stratified_allocation(categories, n_total):
    """Split ``n_total`` samples across categories in proportion to their frequency.

    Uses the largest-remainder method so that the counts add up to exactly ``n_total``.
    """
    proportions = categories.value_counts(normalize=True)
    raw = proportions * n_total
    alloc = np.floor(raw).astype(int)
    n_left = n_total - alloc.sum()
    alloc[(raw - alloc).sort_values(ascending=False).index[:n_left]] += 1
    return alloc


def stratified_sample(df, n_total, seed, by="dominant_impurity"):
    """Draw ``n_total`` rows from ``df`` stratified by the column ``by``."""
    alloc = stratified_allocation(df[by], n_total)
    parts = [
        df[df[by] == cat].sample(n=min(n, (df[by] == cat).sum()), random_state=seed)
        for cat, n in alloc.items()
        if n > 0
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# YOLO annotation files (CVAT "YOLO 1.1" export)
# --------------------------------------------------------------------------- #
def read_class_names(obj_names_path):
    """Class names in the exact order used by the label indices."""
    with open(obj_names_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_file_list(txt_path):
    """Read a list of image paths (one per line) and keep only the file names."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return [os.path.basename(line.strip()) for line in f if line.strip()]


def label_path_for(image_name, labels_dir):
    """``abc.jpg`` -> ``<labels_dir>/abc.txt``."""
    return Path(labels_dir) / (Path(image_name).stem + ".txt")


def read_yolo_labels(label_path):
    """Return a list of ``(class_id, xc, yc, w, h)`` tuples (normalised coordinates)."""
    rows = []
    if not Path(label_path).exists():
        return rows
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """Convert a normalised YOLO box to pixel ``[x1, y1, x2, y2]``."""
    return [
        (xc - w / 2) * img_w,
        (yc - h / 2) * img_h,
        (xc + w / 2) * img_w,
        (yc + h / 2) * img_h,
    ]


def build_label_matrix(image_names, labels_dir, n_classes):
    """Binary matrix ``[n_images, n_classes]``: 1 if the class appears in the image."""
    matrix = np.zeros((len(image_names), n_classes), dtype=int)
    for i, name in enumerate(image_names):
        for cls_id, *_ in read_yolo_labels(label_path_for(name, labels_dir)):
            matrix[i, cls_id] = 1
    return matrix
