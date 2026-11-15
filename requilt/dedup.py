"""Step 1 - near-duplicate detection with perceptual hashing and a BK-tree.

Video frames captured during slow panning look almost identical. We compute
a perceptual hash (pHash) for every image and index the hashes in a BK-tree,
which finds all hashes within a given Hamming distance without comparing
every pair of images.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def compute_phash(image_path, hash_size=8):
    """Return the pHash of one image as an int, or ``None`` if it cannot be read."""
    try:
        with Image.open(image_path) as img:
            return int(str(imagehash.phash(img, hash_size=hash_size)), 16)
    except Exception:
        return None


def _hash_one(args):
    path, hash_size = args
    return Path(path).name, compute_phash(path, hash_size)


def compute_phashes(image_paths, hash_size=8, num_workers=8):
    """Hash many images in parallel. Returns ``{filename: hash_int}``."""
    jobs = [(str(p), hash_size) for p in image_paths]
    hashes = {}
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for name, h in tqdm(pool.map(_hash_one, jobs, chunksize=256), total=len(jobs), desc="pHash"):
            if h is not None:
                hashes[name] = h
    return hashes


def hamming(a, b):
    """Number of differing bits between two integer hashes."""
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# BK-tree
# --------------------------------------------------------------------------- #
class BKTree:
    """Metric tree for fast "all items within distance r" queries.

    Each node stores one hash; its children are keyed by their distance to
    that node. By the triangle inequality, a query with radius ``r`` only
    needs to visit children whose key lies in ``[d - r, d + r]``.
    """

    def __init__(self):
        self.root = None  # node = [hash, name, {distance: child_node}]

    def add(self, h, name):
        if self.root is None:
            self.root = [h, name, {}]
            return
        node = self.root
        while True:
            d = hamming(h, node[0])
            child = node[2].get(d)
            if child is None:
                node[2][d] = [h, name, {}]
                return
            node = child

    def query(self, h, radius):
        """Return ``[(distance, name), ...]`` for all stored hashes within ``radius``."""
        if self.root is None:
            return []
        found, stack = [], [self.root]
        while stack:
            node = stack.pop()
            d = hamming(h, node[0])
            if d <= radius:
                found.append((d, node[1]))
            for key, child in node[2].items():
                if d - radius <= key <= d + radius:
                    stack.append(child)
        return found


def find_near_duplicates(hashes, max_distance=5):
    """Greedy near-duplicate removal.

    Images are visited in sorted filename order (deterministic). An image
    whose hash is within ``max_distance`` of an already-kept image is marked
    as a near-duplicate of it; otherwise it is kept and added to the tree.

    Returns
    -------
    list of dict with keys ``Filename``, ``duplicate_of``, ``hamming_distance``
    (one row per removed image).
    """
    tree, removed = BKTree(), []
    for name in tqdm(sorted(hashes), desc="BK-tree"):
        h = hashes[name]
        matches = tree.query(h, max_distance)
        if matches:
            dist, kept_name = min(matches)
            removed.append({"Filename": name, "duplicate_of": kept_name, "hamming_distance": dist})
        else:
            tree.add(h, name)
    return removed
