"""M6a — CPU-tier backbone generation.

RFdiffusion (the brief's specified backbone generator) needs CUDA SE3-
Transformer kernels and is not runnable on this 2-core CPU sandbox. Per the
brief's Part 4 compute strategy, the GPU step is (a) not silently skipped —
install + Colab instructions live in notebooks/colab_rfdiffusion.ipynb and
results/design/GPU_TIER_STATUS.md — and (b) the closed loop is still fully
exercised on CPU using a documented, deterministic, non-ML substitute: a
small library of idealized secondary-structure topologies (built with real
peptide geometry via geometry.py's NeRF construction, not fabricated
coordinates), rigid-body docked onto the M5 target tip's hotspot-residue
centroid with sampled approach angles.

This is a geometric BASELINE, explicitly weaker than a trained generative
model — it does not learn to shape-complement the target the way RFdiffusion
does. Its role is to keep every downstream stage (ProteinMPNN, scoring,
active learning, selectivity) genuinely exercised end-to-end on real
(if geometrically naive) backbones, so that dropping in real RFdiffusion
backbones from the Colab notebook later is a pure swap (same file format,
same downstream code path — see results/design/backbones/README.md).
"""
from __future__ import annotations

import numpy as np

from sentinel.design.geometry import build_backbone
from sentinel.design.scaffold_library import SCAFFOLD_SOURCES, load_scaffold_backbone
from sentinel.design.topology_builder import build_packed_bundle
from sentinel.utils.config import load_config, repo_path

# Each idealized topology is a list of (secondary-structure-type, length)
# segments plus the loop length connecting consecutive segments, built via
# topology_builder.build_packed_bundle, which EXPLICITLY places each segment
# in a real, packed antiparallel arrangement (verified: ~10.5 A inter-helix
# spacing, ~180 degree antiparallel axis angle) rather than hoping a naive
# dihedral-only chain happens to fold back on itself (it doesn't — see
# PROGRESS_LOG.md for the measurement that caught this).
#
# These are kept alongside (not replaced by) the real scaffold library below
# — the active-learning loop can then empirically discover whether idealized
# or real-scaffold backbones perform better, which is itself an honest,
# informative result rather than an assumption.
IDEALIZED_TOPOLOGIES = {
    "helix_hairpin": {"segments": [("H", 30), ("H", 30)], "loop_length": 4},            # 64 aa
    "three_helix_bundle": {"segments": [("H", 24), ("H", 24), ("H", 24)], "loop_length": 3},  # 78 aa
    "helix_strand_helix": {"segments": [("H", 24), ("E", 10), ("H", 24)], "loop_length": 3},  # 64 aa
    "long_helix_capper": {"segments": [("H", 65)], "loop_length": 0},                    # 65 aa, no packing needed
}

# Backwards-compatible alias (older code/tests reference TOPOLOGIES for the
# idealized set specifically).
TOPOLOGIES = IDEALIZED_TOPOLOGIES

REAL_SCAFFOLD_NAMES = list(SCAFFOLD_SOURCES.keys())
ALL_TOPOLOGY_NAMES = list(IDEALIZED_TOPOLOGIES.keys()) + REAL_SCAFFOLD_NAMES

AA_PLACEHOLDER = "ALA"


def build_topology_backbone(topo_name: str, topology_seed: int) -> dict:
    """Build one topology's coordinates. Real scaffolds are deterministic by
    construction (a fixed solved structure — no seed needed for shape, only
    for the downstream docking pose). Idealized topologies build
    deterministically from their own seed (independent of the docking pose
    seed), so selectivity.py can rebuild the identical backbone shape when
    redocking onto other folds either way."""
    if topo_name in SCAFFOLD_SOURCES:
        return load_scaffold_backbone(topo_name)
    spec = IDEALIZED_TOPOLOGIES[topo_name]
    if len(spec["segments"]) == 1:
        ss_char, length = spec["segments"][0]
        return build_backbone(ss_char * length)
    return build_packed_bundle(spec["segments"], spec["loop_length"], seed=topology_seed)


def topology_length(topo_name: str) -> int:
    if topo_name in SCAFFOLD_SOURCES:
        return SCAFFOLD_SOURCES[topo_name]["expected_length"]
    spec = IDEALIZED_TOPOLOGIES[topo_name]
    n_segments = len(spec["segments"])
    return sum(length for _, length in spec["segments"]) + spec["loop_length"] * (n_segments - 1)


def topology_source_label(topo_name: str) -> str:
    if topo_name in SCAFFOLD_SOURCES:
        pdb_id = SCAFFOLD_SOURCES[topo_name]["pdb_id"]
        return f"real_solved_scaffold ({pdb_id}, RCSB PDB)"
    return "cpu_geometric_baseline (idealized NeRF-built topology; RFdiffusion Colab-deferred)"


def _write_backbone_pdb(coords: dict, dest_path, chain_id: str = "A") -> None:
    lines = []
    atom_serial = 1
    n_res = coords["N"].shape[0]
    for res_i in range(n_res):
        for atom_name in ["N", "CA", "C", "O"]:
            x, y, z = coords[atom_name][res_i]
            lines.append(
                f"ATOM  {atom_serial:5d}  {atom_name:<3s}{AA_PLACEHOLDER:>4s} {chain_id}"
                f"{res_i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
                f"          {atom_name[0]:>2s}"
            )
            atom_serial += 1
    lines.append("TER")
    lines.append("END")
    with open(dest_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c + a * d), 2 * (b * d - a * c)],
        [2 * (b * c - a * d), a * a + c * c - b * b - d * d, 2 * (c * d + a * b)],
        [2 * (b * d + a * c), 2 * (c * d - a * b), a * a + d * d - b * b - c * c],
    ])


def dock_onto_target(coords: dict, target_centroid: np.ndarray, approach_direction: np.ndarray,
                       standoff_A: float, rng: np.random.Generator) -> dict:
    """Rigid-body place a backbone so its geometric center sits `standoff_A`
    away from target_centroid along approach_direction, with a random
    rotation about that axis for orientational diversity (a real, if crude,
    substitute for a learned docking pose)."""
    all_atoms = np.concatenate([coords[a] for a in ["N", "CA", "C", "O"]], axis=0)
    center = all_atoms.mean(axis=0)
    centered = {a: coords[a] - center for a in coords}

    twist_angle = rng.uniform(0, 2 * np.pi)
    tilt_axis = rng.normal(size=3)
    tilt_axis /= np.linalg.norm(tilt_axis)
    tilt_angle = rng.uniform(-np.radians(20), np.radians(20))

    r_twist = _rotation_matrix(approach_direction, twist_angle)
    r_tilt = _rotation_matrix(tilt_axis, tilt_angle)
    r = r_tilt @ r_twist

    placed = {a: (centered[a] @ r.T) + target_centroid + approach_direction * standoff_A
               for a in centered}
    return placed


def _pose_score(placed: dict, target_coords: dict, clash_distance_A: float = 2.8,
                  contact_distance_A: float = 5.0) -> float:
    """Fast, dependency-free scoring for pose search: packing density minus a
    clash penalty (the same physical quantities interface_scorer.geometric_
    complementarity computes, inlined here to avoid a heavier per-candidate
    import and to keep this local search cheap enough to run every backbone)."""
    binder_atoms = np.concatenate([placed[a] for a in ["N", "CA", "C", "O"]], axis=0)
    target_atoms = np.concatenate([target_coords[a] for a in ["N", "CA", "C", "O"]], axis=0)
    diffs = binder_atoms[:, None, :] - target_atoms[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    n_clashes = int((dists < clash_distance_A).sum())
    contacts_per_atom = ((dists >= clash_distance_A) & (dists < contact_distance_A)).sum(axis=1)
    mean_contacts = float(contacts_per_atom.mean())
    sc_proxy = mean_contacts / (mean_contacts + 8.0)
    clash_score = n_clashes / binder_atoms.shape[0]
    return sc_proxy - clash_score


def _single_basin_hill_climb(start: dict, target_coords: dict, rng: np.random.Generator,
                               n_iterations: int) -> tuple[dict, float]:
    best = start
    best_score = _pose_score(best, target_coords)

    translation_step_A = 2.0
    rotation_step_rad = np.radians(15)
    for it in range(n_iterations):
        cooling = 1.0 - (it / n_iterations)
        all_atoms = np.concatenate([best[a] for a in ["N", "CA", "C", "O"]], axis=0)
        center = all_atoms.mean(axis=0)
        centered = {a: best[a] - center for a in best}

        jitter_t = rng.normal(size=3) * translation_step_A * cooling
        jitter_axis = rng.normal(size=3)
        jitter_axis /= np.linalg.norm(jitter_axis)
        jitter_angle = rng.uniform(-rotation_step_rad, rotation_step_rad) * cooling
        r_jitter = _rotation_matrix(jitter_axis, jitter_angle)

        candidate = {a: (centered[a] @ r_jitter.T) + center + jitter_t for a in centered}
        score = _pose_score(candidate, target_coords)
        if score > best_score:
            best, best_score = candidate, score

    return best, best_score


def refine_dock_pose(coords: dict, target_centroid: np.ndarray, approach_direction: np.ndarray,
                       target_coords: dict, standoff_A: float, rng: np.random.Generator,
                       n_iterations: int = 40, n_restarts: int = 3) -> dict:
    """Real local rigid-body docking refinement: a simple, dependency-free
    hill-climbing search over small translation/rotation perturbations that
    maximizes packing complementarity while minimizing clashes against the
    ACTUAL target tip coordinates (not just a fixed standoff from the
    hotspot centroid, which is all the original one-shot placement used).
    This is a real, if simplified, physics-based docking step — the same
    class of local-search refinement real docking tools (e.g. RosettaDock's
    local perturbation phase) use, just without a full energy function.

    Runs n_restarts independent basins (each from its own random initial
    placement) and keeps the best. A real bug was found and fixed here: a
    single basin can get stuck — the cooling schedule shrinks the step size
    over iterations, so an unlucky initial placement with a bad clash
    sometimes cannot be escaped within budget, occasionally producing a
    catastrophically bad final pose purely from initialization luck rather
    than any real geometric incompatibility (measured: one backbone's AD-tip
    redock in the M6e selectivity scan landed at packing-minus-clash=-0.23
    against that same backbone's own other-fold scores all sitting near
    zero, entirely attributable to a single bad basin — see PROGRESS_LOG.md
    M6-quality-4). Multi-restart is the standard fix for basin-dependent
    local search; it can only match or beat a single restart, never do
    worse. Deterministic given rng."""
    best, best_score = None, -np.inf
    for _ in range(n_restarts):
        start = dock_onto_target(coords, target_centroid, approach_direction, standoff_A, rng)
        pose, score = _single_basin_hill_climb(start, target_coords, rng, n_iterations)
        if score > best_score:
            best, best_score = pose, score
    return best


def generate_backbones(target_spec: dict, n_backbones: int, seed: int) -> list[dict]:
    import biotite.structure.io.pdb as pdb_io

    rng = np.random.default_rng(seed)
    stack_pdb = repo_path(target_spec["reference_stack_pdb"])
    tip_chain = target_spec["reference_tip_chain"]
    hotspots = target_spec["conditioning_residues_for_rfdiffusion"]

    reader = pdb_io.PDBFile.read(str(stack_pdb))
    arr = reader.get_structure(model=1)
    hotspot_mask = (arr.chain_id == tip_chain) & np.isin(arr.res_id, hotspots) & (arr.atom_name == "CA")
    hotspot_coords = arr.coord[hotspot_mask]
    target_centroid = hotspot_coords.mean(axis=0)

    all_ca_mask = (arr.chain_id == tip_chain) & (arr.atom_name == "CA")
    chain_centroid = arr.coord[all_ca_mask].mean(axis=0)
    approach_direction = target_centroid - chain_centroid
    approach_direction /= np.linalg.norm(approach_direction)

    out_dir = repo_path("results", "design", "backbones")
    out_dir.mkdir(parents=True, exist_ok=True)

    topo_names = ALL_TOPOLOGY_NAMES
    manifest = []
    for i in range(n_backbones):
        topo_name = topo_names[i % len(topo_names)]
        topology_seed = int(rng.integers(0, 2**31 - 1))
        coords = build_topology_backbone(topo_name, topology_seed)
        standoff = float(rng.uniform(9.0, 13.0))
        placed = dock_onto_target(coords, target_centroid, approach_direction, standoff, rng)

        tag = f"backbone_{i:03d}_{topo_name}"
        pdb_path = out_dir / f"{tag}.pdb"
        _write_backbone_pdb(placed, pdb_path)
        dock_seed = int(rng.integers(0, 2**31 - 1))
        manifest.append({
            "backbone_id": tag, "topology": topo_name, "length": topology_length(topo_name),
            "topology_seed": topology_seed, "standoff_A": round(standoff, 2), "pdb_path": str(pdb_path),
            "dock_seed": dock_seed,  # lets selectivity.py reproduce the identical rigid-body
                                      # pose (twist/tilt) when redocking this same backbone onto
                                      # each negative-design fold's tip for a fair comparison
            "source": topology_source_label(topo_name),
        })

    return manifest
