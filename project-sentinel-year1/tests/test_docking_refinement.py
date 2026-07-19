"""Tests for the local rigid-body docking refinement (backbone_gen.refine_
dock_pose) — a real local-search pose optimization added because a single
random placement was leaving real, avoidable clashes/poor packing on the
table (the same class of local perturbation real docking tools like
RosettaDock use, simplified to a dependency-free hill-climb)."""
import numpy as np

from sentinel.design.backbone_gen import (
    _pose_score, build_topology_backbone, dock_onto_target, refine_dock_pose,
)


def _make_flat_target(n_atoms_per_type=30, spacing=4.0):
    """A synthetic flat grid of target atoms in the xy-plane at z=0 — a
    simple, real, reproducible surface to dock against for a pure
    unit test (not the real tau tip, so this test has no external file
    dependency)."""
    side = int(np.sqrt(n_atoms_per_type))
    xs, ys = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
    grid = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    return {a: grid.copy() for a in ["N", "CA", "C", "O"]}


def test_refine_dock_pose_never_gets_worse_than_starting_pose():
    coords = build_topology_backbone("helix_hairpin", topology_seed=1)
    target = _make_flat_target()
    target_centroid = np.array([10.0, 10.0, 0.0])
    approach = np.array([0.0, 0.0, 1.0])

    rng_start = np.random.default_rng(5)
    starting_pose = dock_onto_target(coords, target_centroid, approach, 10.0, rng_start)
    starting_score = _pose_score(starting_pose, target)

    rng_refine = np.random.default_rng(5)
    refined_pose = refine_dock_pose(coords, target_centroid, approach, target, 10.0,
                                       rng_refine, n_iterations=40)
    refined_score = _pose_score(refined_pose, target)

    assert refined_score >= starting_score - 1e-9, (
        "refinement must be monotonically non-worsening (hill-climb only accepts improving moves)"
    )


def test_refine_dock_pose_is_deterministic_given_seed():
    coords = build_topology_backbone("three_helix_bundle", topology_seed=2)
    target = _make_flat_target()
    target_centroid = np.array([8.0, 8.0, 0.0])
    approach = np.array([0.0, 0.0, 1.0])

    pose_a = refine_dock_pose(coords, target_centroid, approach, target, 10.0,
                                np.random.default_rng(99), n_iterations=20)
    pose_b = refine_dock_pose(coords, target_centroid, approach, target, 10.0,
                                np.random.default_rng(99), n_iterations=20)
    assert np.allclose(pose_a["CA"], pose_b["CA"])


def test_refine_dock_pose_multirestart_never_worse_than_single_restart():
    """Regression test for a real bug: a single hill-climb basin can get
    stuck near a bad initial placement (the cooling schedule shrinks step
    size over iterations, limiting how far a late escape can travel).
    Multi-restart must never do worse than n_restarts=1 for the same rng
    stream, since it strictly explores a superset of what a single restart
    would find."""
    coords = build_topology_backbone("helix_hairpin", topology_seed=1)
    target = _make_flat_target()
    target_centroid = np.array([10.0, 10.0, 0.0])
    approach = np.array([0.0, 0.0, 1.0])

    single = refine_dock_pose(coords, target_centroid, approach, target, 10.0,
                                np.random.default_rng(5), n_iterations=40, n_restarts=1)
    multi = refine_dock_pose(coords, target_centroid, approach, target, 10.0,
                               np.random.default_rng(5), n_iterations=40, n_restarts=3)

    assert _pose_score(multi, target) >= _pose_score(single, target) - 1e-9


def test_refine_dock_pose_preserves_backbone_geometry():
    """Refinement is rigid-body only — internal bond lengths must be
    completely unchanged by the pose search."""
    coords = build_topology_backbone("long_helix_capper", topology_seed=3)
    target = _make_flat_target()
    target_centroid = np.array([5.0, 5.0, 0.0])
    approach = np.array([0.0, 0.0, 1.0])
    refined = refine_dock_pose(coords, target_centroid, approach, target, 10.0,
                                 np.random.default_rng(7), n_iterations=20)

    orig_n_ca = np.linalg.norm(coords["N"] - coords["CA"], axis=1)
    refined_n_ca = np.linalg.norm(refined["N"] - refined["CA"], axis=1)
    assert np.allclose(orig_n_ca, refined_n_ca, atol=1e-6)
