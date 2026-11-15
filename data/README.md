# Data shipped with ReQuilt

Quilt-1M images and captions are **not** redistributed here. Get them from the
[Quilt-1M release](https://github.com/wisdomikezogwo/quilt1m) under its own license.
This folder holds only what this work produced.

```
data/
├── annotations/            # manual annotations: 1,000 impure Quilt-1M images (CVAT "YOLO 1.1" export)
│   ├── obj.names           #   class names, in label-index order
│   ├── obj.data            #   CVAT export metadata
│   ├── train.txt           #   the 1,000 annotated image names
│   └── obj_train_data/     #   one YOLO label file per image: <class> <xc> <yc> <w> <h> (normalised)
├── splits/                 # multi-label stratified split used in the paper (Step 3)
│   ├── train.txt           #   850 images
│   └── val.txt             #   150 images
└── evaluation/             # held-out evaluation samples used in the paper
    ├── test_dirty_sample.csv   # 1,000 impure images + QuiltCleaner scores
    └── test_clean_sample.csv   #   200 clean images merged into the final dataset
```

## Classes

| id | Class |
|:-:|---|
| 0 | Persons/Photos |
| 1 | Desktop/Windows/SlideViewer |
| 2 | Text/Logo in Image |
| 3 | Pathology Tissue |

Annotator: a single annotator using [CVAT](https://www.cvat.ai/). Tissue boxes exclude
software-interface elements along the tissue's edges and corners.
