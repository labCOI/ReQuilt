"""Unit tests for the pHash + BK-tree near-duplicate search (Step 1)."""

import random

from requilt.dedup import BKTree, find_near_duplicates, hamming


def test_bktree_matches_brute_force():
    rng = random.Random(0)
    hashes = [rng.getrandbits(64) for _ in range(500)]
    hashes += [h ^ (1 << rng.randrange(64)) for h in hashes[:50]]   # 1-bit near copies

    tree = BKTree()
    for i, h in enumerate(hashes):
        tree.add(h, i)

    for q in hashes[:40]:
        expected = sorted(i for i, h in enumerate(hashes) if hamming(q, h) <= 5)
        assert sorted(name for _, name in tree.query(q, 5)) == expected


def test_find_near_duplicates_keeps_first_in_sorted_order():
    hashes = {"a.jpg": 0b0, "b.jpg": 0b111, "c.jpg": (1 << 63) - 1}
    removed = find_near_duplicates(hashes, max_distance=5)
    assert removed == [{"Filename": "b.jpg", "duplicate_of": "a.jpg", "hamming_distance": 3}]
