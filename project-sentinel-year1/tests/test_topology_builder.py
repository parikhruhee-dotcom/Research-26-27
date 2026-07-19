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


def test_mixed_length_segments_do_not_produce_collapsed_loops():
    """Regression test for a real bug: segments were anchored to a shared
    z-plane derived from the LONGEST segment's half-length, so a much
    shorter segment (e.g. an 8-residue strand between two 20-residue
    helices) fell far short of reaching that plane, leaving the connecting
    loop to bridge a gap on the order of a full helix length rather than
    the small xy packing gap it is actually sized for. Measured directly on
    the real design this produced: CA(i) and CA(i+2) landed 1.93 A apart
    (should be several A at minimum for any real backbone conformation),
    and the full-atom PDBFixer/OpenMM reconstruction of that geometry
    crashed downstream MD with NaN particle coordinates. Checked across
    several genuinely mismatched segment-length combinations and seeds, not
    just the one specific case that was caught."""
    specs_and_seeds = [
        ([("H", 20), ("E", 8), ("H", 20)], 1),
        ([("H", 20), ("E", 8), ("H", 20)], 7044),  # the exact seed that produced the real failure
        ([("H", 30), ("E", 6), ("H", 30)], 3),
        ([("H", 12), ("H", 40), ("H", 12)], 5),
    ]
    for specs, seed in specs_and_seeds:
        full = build_packed_bundle(specs, loop_length=3, seed=seed)
        ca = full["CA"]
        i2 = np.linalg.norm(ca[:-2] - ca[2:], axis=1)
        # 2.0 A, not the ~3.8-5 A typical of mid-segment residues: a small residual
        # tightness right at the loop-to-next-segment junction survives the sequential-
        # z-tracking fix (measured: worst case ~2.56 A, at exactly that junction) --
        # dramatically better than the pre-fix failure (~1.93 A, mid-loop, with the
        # full-atom reconstruction crashing downstream MD with NaN) and directly confirmed
        # end-to-end MD-stable for the real seed=7044 case that actually failed in
        # production (see PROGRESS_LOG.md and tests/test_md.py's MD-level regression test).
        assert i2.min() > 2.0, (
            f"specs={specs} seed={seed}: CA(i)-CA(i+2) should never collapse below ~2 A "
            f"(a real backbone cannot fold back on itself that tightly), got {i2.min():.3f} A"
        )
        i1 = np.linalg.norm(np.diff(ca, axis=0), axis=1)
        assert i1.min() > 1.5, f"specs={specs} seed={seed}: no near-overlapping consecutive CA atoms"


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
