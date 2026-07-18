"""Regression tests for the packed-bundle backbone builder. This module
exists specifically because an earlier version silently failed to fold: a
naive single-chain dihedral construction left a 'helix-hairpin' with its two
helices ~40 A apart (an extended rod, not a fold) — caught by exactly the
kind of geometric check below. These tests exist so that regression can
never happen silently again."""
import numpy as np
import pytest

from sentinel.design.topology_builder import HELIX_PACKING_DISTANCE_A, build_packed_bundle


def test_hairpin_helices_are_actually_antiparallel_and_packed():
    full = build_packed_bundle([("H", 28), ("H", 28)], loop_length=4, seed=42)
    ca = full["CA"]
    h1, h2 = ca[:28], ca[32:60]
    axis1 = (h1[-1] - h1[0]); axis1 /= np.linalg.norm(axis1)
    axis2 = (h2[-1] - h2[0]); axis2 /= np.linalg.norm(axis2)
    angle_deg = np.degrees(np.arccos(np.clip(np.dot(axis1, axis2), -1, 1)))
    assert angle_deg > 150, f"helices should be ~antiparallel, got {angle_deg} degrees"

    midpoint_dist = np.linalg.norm(h1[len(h1) // 2] - h2[len(h2) // 2])
    assert 7.0 < midpoint_dist < 15.0, (
        f"packed helices should sit ~{HELIX_PACKING_DISTANCE_A} A apart, got {midpoint_dist} A "
        f"(a value near 40 A would mean this is an unfolded extended rod, not a real hairpin)"
    )


def test_three_helix_bundle_is_compact_not_a_flat_row():
    full = build_packed_bundle([("H", 22), ("H", 22), ("H", 22)], loop_length=4, seed=42)
    ca = full["CA"]
    h1, h2, h3 = ca[:22], ca[26:48], ca[52:74]
    pairs = [(h1, h2), (h2, h3), (h1, h3)]
    for a, b in pairs:
        d = np.linalg.norm(a[len(a) // 2] - b[len(b) // 2])
        assert 7.0 < d < 20.0, f"every pair of helices in a 3-helix bundle should be reasonably close, got {d}"


def test_no_backbone_clashes_or_absurd_stretching():
    full = build_packed_bundle([("H", 24), ("H", 24), ("H", 24)], loop_length=3, seed=7)
    ca = full["CA"]
    steps = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    assert steps.min() > 1.5, "no near-overlapping consecutive CA atoms"
    assert steps.max() < 10.0, "no absurdly stretched consecutive CA-CA distance"


def test_mixed_helix_strand_helix_topology_builds():
    full = build_packed_bundle([("H", 20), ("E", 8), ("H", 20)], loop_length=3, seed=1)
    assert full["CA"].shape[0] == 20 + 8 + 20 + 3 + 3
    for atom in ["N", "CA", "C", "O"]:
        assert full[atom].shape[0] == full["CA"].shape[0]


def test_deterministic_given_seed():
    a = build_packed_bundle([("H", 20), ("H", 20)], loop_length=3, seed=99)
    b = build_packed_bundle([("H", 20), ("H", 20)], loop_length=3, seed=99)
    assert np.allclose(a["CA"], b["CA"])


def test_different_seeds_give_different_roll_but_same_packing():
    a = build_packed_bundle([("H", 20), ("H", 20)], loop_length=3, seed=1)
    b = build_packed_bundle([("H", 20), ("H", 20)], loop_length=3, seed=2)
    assert not np.allclose(a["CA"], b["CA"])  # different random roll angles
    # but both should still be genuinely packed, not both degenerate the same way
    for full in (a, b):
        ca = full["CA"]
        h1, h2 = ca[:20], ca[23:43]
        d = np.linalg.norm(h1[10] - h2[10])
        assert d < 20.0
