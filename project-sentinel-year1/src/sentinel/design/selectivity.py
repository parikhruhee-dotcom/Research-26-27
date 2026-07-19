"""M6e — Selectivity / negative design.

For every surviving design, redock the SAME backbone shape (identical
topology + rigid-body sampling seed, only the target tip differs) onto the
AD tip and onto every negative-design-panel fold's tip, then compare a
combined geometric + chemical complementarity score. A design that fits AD's
templating tip better than the others' (by
>= config.design.selectivity.min_selectivity_margin) is AD-selective — not
just "a sticky binder that fits everything."

Also implements the developability filter (M6e second half): the binder's
own aggregation propensity (reusing M3 on the binder sequence — a good
binder must not itself nucleate aggregation), plus a free-cysteine check.

A real bug was found and fixed here: selectivity originally scored by
BACKBONE SHAPE alone (build_selectivity_matrix took a list of backbones, not
designs, and never looked at the designed sequence at all) using only
geometric_complementarity, which is itself sequence-blind (see
interface_scorer.geometric_complementarity/chemical_complementarity
docstrings). That meant every design sharing a backbone got the IDENTICAL
selectivity call regardless of its actual sequence, and the score itself
could never reflect whether that specific sequence's chemistry preferred AD
over another fold — only whether the generic rigid shape happened to fit.
Measured directly: without a chemistry term, AD-tip scores averaged BELOW
the mean other-fold score across the top-10 backbone pool (real tauopathy
fibril tips can share broadly similar concave amyloid-groove geometry even
when their fold, and surface chemistry, genuinely differs). Fixed by scoring
per-DESIGN (backbone shape + actual sequence) and combining geometric
complementarity with interface_scorer.chemical_complementarity, using the
same composite weights as design-time scoring for methodological
consistency.
"""
from __future__ import annotations

import numpy as np

from sentinel.aggregation.scorer import compute_profile as compute_aggregation_profile
from sentinel.design.backbone_gen import build_topology_backbone, refine_dock_pose
from sentinel.design.interface_scorer import (
    chemical_complementarity, geometric_complementarity, scale_chemical_complementarity,
)
from sentinel.utils.config import load_config, repo_path


def _fold_target_geometry(prepared_entry: dict, hotspot_resids: list[int]) -> tuple[np.ndarray, np.ndarray, dict, list]:
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
    # real target residue identities aligned with tip_coords["CA"], for chemical_complementarity
    tip_res_names = arr.res_name[all_ca_mask].tolist()
    return target_centroid, approach, tip_coords, tip_res_names


def score_design_against_fold(design: dict, prepared_entry: dict, hotspot_resids: list[int],
                                weights: dict, chemical_weight: float) -> dict:
    """Redocks the SAME backbone SHAPE (identical topology, identical seed —
    so the local-search's random perturbation sequence is identical too) onto
    a different fold's tip, with its own local docking refinement run fresh
    against that fold's surface, THEN scores the design's ACTUAL sequence's
    chemical complementarity against that fold's real surface residues. This
    is the physically honest comparison: a real binder would also locally
    settle against whatever surface it actually encounters.

    chemical_weight is deliberately much smaller here than the design-time
    composite weight — see config design.selectivity.chemical_complementarity_weight's
    comment for why: all 9 fold targets are the SAME tau sequence in
    different conformations, so bulk chemical composition is close to fold-
    invariant and would otherwise dilute the one signal (geometric/spatial
    register) that genuinely differs by fold."""
    coords = build_topology_backbone(design["topology"], design["topology_seed"])
    target_centroid, approach, tip_coords, tip_res_names = _fold_target_geometry(prepared_entry, hotspot_resids)

    rng = np.random.default_rng(design["dock_seed"])
    placed = refine_dock_pose(coords, target_centroid, approach, tip_coords,
                                design["standoff_A"], rng, n_iterations=40)
    geom = geometric_complementarity(placed, tip_coords)
    chem = chemical_complementarity(placed["CA"], design["sequence"], tip_coords["CA"], tip_res_names)
    chem_scaled = scale_chemical_complementarity(chem)

    combined = (weights["packing_density_sc_proxy"] * geom["packing_density_sc_proxy"]
                 - weights["clash_penalty"] * geom["clash_score"]
                 + chemical_weight * chem_scaled)
    return {"geometric": geom, "chemical": chem, "combined_score": round(float(combined), 4)}


def build_selectivity_matrix(designs: list[dict], target_spec: dict, prepared_by_id: dict) -> dict:
    """designs: list of {design_id, topology, topology_seed, dock_seed,
    standoff_A, sequence} — a real backbone SHAPE plus the ACTUAL designed
    sequence docked on it (see module docstring for the per-backbone-only
    bug this fixes)."""
    cfg = load_config()
    hotspot_resids = target_spec["hotspot_residues"]
    weights = cfg["design"]["scoring"]["composite_weights"]
    chemical_weight = cfg["design"]["selectivity"]["chemical_complementarity_weight"]

    ref_prepared = prepared_by_id[target_spec["reference_pdb_id"]]
    fold_targets = {target_spec["reference_strain"]: ref_prepared}
    for neg in target_spec["negative_design_panel"]:
        fold_targets[neg["strain"]] = prepared_by_id[neg["pdb_id"]]

    matrix = {}
    for design in designs:
        row = {}
        for strain, prepared_entry in fold_targets.items():
            scored = score_design_against_fold(design, prepared_entry, hotspot_resids, weights, chemical_weight)
            row[strain] = scored["combined_score"]
        matrix[design["design_id"]] = row

    margin = cfg["design"]["selectivity"]["min_selectivity_margin"]
    selective = {}
    for design_id, row in matrix.items():
        ref_strain = target_spec["reference_strain"]
        others = [v for s, v in row.items() if s != ref_strain]
        mean_other = float(np.mean(others)) if others else 0.0
        selective[design_id] = {
            "ad_score": row[ref_strain], "mean_other_score": round(mean_other, 4),
            "margin": round(row[ref_strain] - mean_other, 4),
            "is_selective": (row[ref_strain] - mean_other) >= margin,
        }
    return {"matrix": matrix, "selectivity_calls": selective}


def developability_filter(sequence: str, reference_population_scores: list[float],
                            tau_normalization_bounds: dict,
                            per_residue_rel_sasa: dict[int, float] | None = None) -> dict:
    """tau_normalization_bounds: from sentinel.aggregation.scorer.get_normalization_bounds()
    on tau's own profile. The binder must be scored on TAU'S scale, not its own —
    min-max normalizing a short binder sequence against only its own windows
    would make its max score trivially always 1.0 regardless of how
    amyloidogenic it actually is. (This is purely a scale/normalization
    choice — it does not imply tau is also the right population to percentile
    against; see reference_population_scores below.)

    reference_population_scores: the population of window scores the binder's
    own worst window is percentiled against. A real bug was found and fixed
    here — tau's own window-score distribution was originally (mis)used for
    this. Tau is intrinsically disordered and overwhelmingly low-aggregation-
    propensity by amino-acid composition, so almost any ordinarily-folded
    protein looks like an outlier by comparison: measured directly, this
    build's own real, hyperstable, industrially-used scaffold proteins
    (native Protein A B-domain, DARPin, engrailed homeodomain) scored at the
    73rd-98th percentile against tau's windows even restricted to solvent-
    exposed patches only — despite having no known aggregation liability.
    That is decisive evidence tau is the wrong reference population. Callers
    should pass developability_reference.build_reference_distribution()'s
    pooled real-scaffold-surface population instead (see that module's
    docstring for the full rationale and the measured evidence). Tests in
    this repo may still pass tau's windows directly to exercise the filter's
    mechanics in isolation — that is fine for testing the windowing/exposure
    logic, but is not the production reference population.

    per_residue_rel_sasa (optional, {1-indexed res_id: relative SASA} on the
    binder's OWN folded structure): a second, separate real bug was found and
    fixed here — the original version flagged a design as a liability
    whenever ANY window scored high on the M3 aggregation scale, but high
    beta-propensity/hydrophobicity is exactly what a real, properly packed
    hydrophobic CORE is supposed to look like (that's what makes it a core).
    The real developability risk is specifically an EXPOSED aggregation-prone
    patch (buried ones are just... a normal hydrophobic core). When
    per-residue SASA is available, only EXPOSED windows (mean relative SASA
    >= the config-documented exposure threshold) count toward the liability
    check; without structural context (per_residue_rel_sasa=None), falls back
    to the original sequence-only behavior, now clearly labeled as an
    approximation rather than the primary check."""
    cfg = load_config()
    profile = compute_aggregation_profile(sequence, normalization_bounds=tau_normalization_bounds)
    max_allowed = cfg["design"]["developability"]["max_binder_aggregation_percentile"]
    exposure_threshold = cfg["atlas"]["exposed_rel_sasa_threshold"]

    if per_residue_rel_sasa:
        window_size = profile["window_size"]
        relevant_scores = []
        for r in profile["records"]:
            window_sasas = [per_residue_rel_sasa[i] for i in range(r["window_start"], r["window_end"] + 1)
                              if i in per_residue_rel_sasa]
            if window_sasas and (sum(window_sasas) / len(window_sasas)) >= exposure_threshold:
                relevant_scores.append(r["combined_score"])
        scoring_basis = "exposed_windows_only"
    else:
        relevant_scores = [r["combined_score"] for r in profile["records"]]
        scoring_basis = "whole_sequence_no_structural_context"

    max_binder_score = max(relevant_scores) if relevant_scores else 0.0
    percentile = float(np.mean(np.array(reference_population_scores) <= max_binder_score) * 100)

    n_cys = sequence.count("C")
    disallow_cys = cfg["design"]["developability"]["disallow_free_cysteines"]
    cys_ok = (n_cys == 0) if disallow_cys else True

    passes = (percentile <= max_allowed) and cys_ok
    return {
        "max_own_aggregation_score": round(max_binder_score, 4),
        "percentile_vs_reference_windows": round(percentile, 2),
        "max_allowed_percentile": max_allowed,
        "scoring_basis": scoring_basis,
        "n_cysteines": n_cys, "cysteine_check_passed": cys_ok,
        "developability_passed": passes,
    }
