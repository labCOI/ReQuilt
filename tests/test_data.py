"""Checks on the shipped annotation files and the stratified split (Step 3)."""

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from requilt import REPO_ROOT
from requilt.data import build_label_matrix, read_class_names, read_file_list, stratified_allocation

ANN = REPO_ROOT / "data" / "annotations"
SPLITS = REPO_ROOT / "data" / "splits"


def test_annotation_files():
    names = read_class_names(ANN / "obj.names")
    assert names == ["Persons/Photos", "Desktop/Windows/SlideViewer",
                     "Text/Logo in Image", "Pathology Tissue"]
    images = read_file_list(ANN / "train.txt")
    assert len(images) == 1000
    assert all((ANN / "obj_train_data" / (img.rsplit(".", 1)[0] + ".txt")).exists() for img in images)


def test_split_is_reproducible():
    images = read_file_list(ANN / "train.txt")
    matrix = build_label_matrix(images, ANN / "obj_train_data", 4)
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=17)
    train_idx, val_idx = next(splitter.split(np.array(images), matrix))
    assert set(images[i] for i in val_idx) == set(read_file_list(SPLITS / "val.txt"))
    assert set(images[i] for i in train_idx) == set(read_file_list(SPLITS / "train.txt"))
    # images per class in the validation split (Table I)
    assert matrix[val_idx].sum(axis=0).tolist() == [37, 97, 56, 143]


def test_stratified_allocation_sums_exactly():
    import pandas as pd
    cats = pd.Series(["a"] * 7 + ["b"] * 2 + ["c"])
    alloc = stratified_allocation(cats, 6)
    assert alloc.sum() == 6 and alloc["a"] >= alloc["b"] >= alloc["c"]
