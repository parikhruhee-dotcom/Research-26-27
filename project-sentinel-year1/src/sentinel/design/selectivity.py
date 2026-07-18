"""M6e — Selectivity / negative design.

For every surviving design, redock the SAME backbone shape (identical
topology + rigid-body sampling seed, only the target tip differs) onto the
AD tip and onto every negative-design-panel fold's tip, then compare the
geometric-complementarity score. A design that fits AD's templating tip
better than the others' (by >= config.design.selectivity.min_selectivity_margin)
is AD-selective — not just "a sticky binder that fits everything."

Also implements the developability filter (M6e second half): the binder's
own aggregation propensity (reusing M3 on the binder sequence — a good
binder must not itself nucleate aggregation), plus a free-cysteine check.
"""
from __future__ import annotations

import numpy as np

from sentinel.aggregation.scorer import compute_profile as compute_aggregation_profile
from sentinel.design.backbone_gen import TOPOLOGIES, build_topology_backbone, dock_onto_target
from sentinel.design.interface_scorer import geometric_complementarity
from sentinel.utils.config import load_config, repo_path


def _fold_target_geometry(prepared_entry: dict, hotspot_resids: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    import biotite.structure.io.pdb as pdb_io
    stack_pdb = repo_path(prepared_entry["stack_pdb"])
    tip_chain = prepared_entry["stack_chains_used"][-1]
    reader = pdb_io.PDBFile.read(str(stack_pdb))
    arr = reader.get_structure(model=1)

    all_ca_mask = (arr.chain_id == tip_chain) & (arr.atom_name == "CA")
    present_resids = set(arr.res_id[all_ca_mask].tolist())
    usable_hotspots = [r for r in hotspot_resids if r in present_resids]
    if len(usable_hotspots) < 3:
        usable_hotspots = sorted(present_resids)  # fold's core doesn't cover AD hotspots; use whole tip

    hotspot_mask = (arr.chain_id == tip_chain) & np.isin(arr.res_id, usable_hotspots) & (arr.atom_name == "CA")
    target_centroid = arr.coord[hotspot_mask].mean(axis=0)
    chain_centroid = arr.coord[all_ca_mask].mean(axis=0)
    approach = target_centroid - chain_centroid
    approach /= np.linalg.norm(approach)

    tip_coords = {}
    for atom_name in ["N", "CA", "C", "O"]:
        mask = (arr.chain_id == tip_chain) & (arr.atom_name == atom_name)
        tip_coords[atom_name] = arr.coord[mask]
    return target_centroid, approach, tip_coords


def score_backbone_against_fold(backbone_record: dict, prepared_entry: dict, hotspot_resids: list[int]) -> dict:
    coords = build_topology_backbone(backbone_record["topology"], backbone_record["topology_seed"])
    target_centroid, approach, tip_coords = _fold_target_geometry(prepared_entry, hotspot_resids)

    rng = np.random.default_rng(backbone_record["dock_seed"])
    placed = dock_onto_target(coords, target_centroid, approach, backbone_record["standoff_A"], rng)
    geom = geometric_complementarity(placed, tip_coords)
    return geom


def build_selectivity_matrix(backbones: list[dict], target_spec: dict, prepared_by_id: dict) -> dict:
    cfg = load_config()
    hotspot_resids = target_spec["hotspot_residues"]

    ref_entry = {"stack_pdb": target_spec["reference_stack_pdb"],
                 "stack_chains_used": [target_spec["reference_tip_chain"]]}
    # reference fold uses the real prepared entry (has the true multi-layer chain list)
    ref_prepared = prepared_by_id[target_spec["reference_pdb_id"]]

    fold_targets = {target_spec["reference_strain"]: ref_prepared}
    for neg in target_spec["negative_design_panel"]:
        fold_targets[neg["strain"]] = prepared_by_id[neg["pdb_id"]]

    matrix = {}
    for bb in backbones:
        row = {}
        for strain, prepared_entry in fold_targets.items():
            geom = score_backbone_against_fold(bb, prepared_entry, hotspot_resids)
            row[strain] = geom["packing_density_sc_proxy"] - geom["clash_score"]
        matrix[bb["backbone_id"]] = row

    margin = cfg["design"]["selectivity"]["min_selectivity_margin"]
    selective = {}
    for bb_id, row in matrix.items():
        ref_strain = target_spec["reference_strain"]
        others = [v for s, v in row.items() if s != ref_strain]
        mean_other = float(np.mean(others)) if others else 0.0
        selective[bb_id] = {
            "ad_score": row[ref_strain], "mean_other_score": round(mean_other, 4),
            "margin": round(row[ref_strain] - mean_other, 4),
            "is_selective": (row[ref_strain] - mean_other) >= margin,
        }
    return {"matrix": matrix, "selectivity_calls": selective}


def developability_filter(sequence: str, full_tau_agg_profile_scores: list[float],
                            tau_normalization_bounds: dict) -> dict:
    """tau_normalization_bounds: from sentinel.aggregation.scorer.get_normalization_bounds()
    on tau's own profile. The binder must be scored on TAU'S scale, not its own —
    min-max normalizing a short binder sequence against only its own windows
    would make its max score trivially always 1.0 regardless of how
    amyloidogenic it actually is."""
    cfg = load_config()
    profile = compute_aggregation_profile(sequence, normalization_bounds=tau_normalization_bounds)
    binder_scores = [r["combined_score"] for r in profile["records"]]
    max_binder_score = max(binder_scores) if binder_scores else 0.0

    percentile = float(np.mean(np.array(full_tau_agg_profile_scores) <= max_binder_score) * 100)
    max_allowed = cfg["design"]["developability"]["max_binder_aggregation_percentile"]

    n_cys = sequence.count("C")
    disallow_cys = cfg["design"]["developability"]["disallow_free_cysteines"]
    cys_ok = (n_cys == 0) if disallow_cys else True

    passes = (percentile <= max_allowed) and cys_ok
    return {
        "max_own_aggregation_score": round(max_binder_score, 4),
        "percentile_vs_tau_windows": round(percentile, 2),
        "max_allowed_percentile": max_allowed,
        "n_cysteines": n_cys, "cysteine_check_passed": cys_ok,
        "developability_passed": passes,
    }
