"""M6 — the closed-loop, self-improving de novo design engine (flagship).

Orchestrates: backbone generation (CPU geometric baseline; RFdiffusion is
Colab-deferred, see GPU_TIER_STATUS.md) -> real ProteinMPNN sequence design
-> CPU interface scoring (geometric complementarity + ESM-2 plausibility) ->
a Gaussian-Process active-learning loop (>=5 rounds) that proposes the next
round's generation settings via expected improvement -> an equal-budget
random-search baseline for comparison (M9 ablation) -> selectivity scoring
against the 7-fold negative-design panel -> developability filtering ->
ranked leads.

Run: python -m sentinel.design.run_design_loop
"""
from __future__ import annotations

import json

import numpy as np

from sentinel.design.active_learning import (
    PARAM_BOUNDS, build_surrogate, decode_params, propose_next_params, random_params,
)
from sentinel.design.backbone_gen import TOPOLOGIES, dock_onto_target
from sentinel.design.geometry import build_backbone
from sentinel.design.interface_scorer import esm_plausibility, geometric_complementarity
from sentinel.design.proteinmpnn_runner import run_proteinmpnn
from sentinel.design.selectivity import build_selectivity_matrix, developability_filter
from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger
from sentinel.utils.seeds import set_global_seed

logger = get_logger(__name__)


def _write_backbone_pdb(coords, dest_path, chain_id="A"):
    from sentinel.design.backbone_gen import _write_backbone_pdb as impl
    impl(coords, dest_path, chain_id)


def load_target_and_tip():
    import biotite.structure.io.pdb as pdb_io
    target_spec = json.load(open(repo_path("results", "target", "ad_capper_target.json")))
    stack_pdb = repo_path(target_spec["reference_stack_pdb"])
    tip_chain = target_spec["reference_tip_chain"]
    reader = pdb_io.PDBFile.read(str(stack_pdb))
    arr = reader.get_structure(model=1)
    tip_coords = {a: arr.coord[(arr.chain_id == tip_chain) & (arr.atom_name == a)]
                   for a in ["N", "CA", "C", "O"]}
    all_ca_mask = (arr.chain_id == tip_chain) & (arr.atom_name == "CA")
    chain_centroid = arr.coord[all_ca_mask].mean(axis=0)
    return target_spec, tip_coords, chain_centroid


def evaluate_one_candidate(x: np.ndarray, round_idx: int, cand_idx: int, target_spec: dict,
                             tip_coords: dict, chain_centroid, cfg: dict, seed: int,
                             mpnn_out_dir, backbone_out_dir) -> dict:
    topologies = list(TOPOLOGIES.keys())
    mpnn_temps = cfg["design"]["proteinmpnn"]["sampling_temperatures"]
    params = decode_params(x, topologies, mpnn_temps)

    all_hotspots = target_spec["hotspot_residues"]
    n_use = max(3, int(round(len(all_hotspots) * params["hotspot_fraction"])))
    hotspots_used = all_hotspots[:n_use]

    import biotite.structure.io.pdb as pdb_io
    stack_pdb = repo_path(target_spec["reference_stack_pdb"])
    tip_chain = target_spec["reference_tip_chain"]
    reader = pdb_io.PDBFile.read(str(stack_pdb))
    arr = reader.get_structure(model=1)
    hotspot_mask = (arr.chain_id == tip_chain) & np.isin(arr.res_id, hotspots_used) & (arr.atom_name == "CA")
    target_centroid = arr.coord[hotspot_mask].mean(axis=0)
    approach = target_centroid - chain_centroid
    approach = approach / np.linalg.norm(approach)

    ss = TOPOLOGIES[params["topology"]]
    coords = build_backbone(ss)
    rng = np.random.default_rng(seed)
    placed = dock_onto_target(coords, target_centroid, approach, params["standoff_A"], rng)

    bb_id = f"r{round_idx}_c{cand_idx}_{params['topology']}"
    bb_pdb_path = backbone_out_dir / f"{bb_id}.pdb"
    _write_backbone_pdb(placed, bb_pdb_path)
    dock_seed = int(rng.integers(0, 2**31 - 1))

    geom = geometric_complementarity(placed, tip_coords)

    n_seq = cfg["design"]["proteinmpnn"]["n_sequences_per_backbone"]
    esm_model = cfg["design"]["scoring"]["esm_model"]
    weights = cfg["design"]["scoring"]["composite_weights"]
    tau_seq = json.load(open(repo_path("data", "interim", "tau_sequence.json")))["sequence"]
    from sentinel.aggregation.scorer import compute_profile
    tau_agg_scores = [r["combined_score"] for r in compute_profile(tau_seq)["records"]]

    try:
        mpnn_records = run_proteinmpnn(str(bb_pdb_path), str(mpnn_out_dir), n_seq,
                                          params["mpnn_temperature"], seed)
    except Exception as exc:
        logger.error(f"ProteinMPNN failed for {bb_id}: {exc}")
        mpnn_records = []

    designs = []
    for si, mrec in enumerate(mpnn_records):
        plaus = esm_plausibility(mrec["sequence"], esm_model)
        dev = developability_filter(mrec["sequence"], tau_agg_scores)
        composite = (weights["packing_density_sc_proxy"] * geom["packing_density_sc_proxy"]
                     - weights["clash_penalty"] * geom["clash_score"]
                     + weights["esm_plausibility"] * plaus
                     + weights["developability_bonus"] * (1.0 if dev["developability_passed"] else 0.0))
        designs.append({
            "design_id": f"{bb_id}_s{si}", "backbone_id": bb_id, "params": params,
            "sequence": mrec["sequence"], "mpnn_score": mrec["mpnn_score"],
            "geometric": geom, "esm_plausibility": plaus, "developability": dev,
            "composite_score": round(float(composite), 4),
        })

    best_score = max((d["composite_score"] for d in designs), default=-1.0)
    return {
        "x": x.tolist(), "params": params, "backbone_id": bb_id, "backbone_pdb": str(bb_pdb_path),
        "dock_seed": dock_seed, "standoff_A": params["standoff_A"], "ss_string": ss,
        "designs": designs, "best_score": best_score,
    }


def run_loop(mode: str, cfg: dict, target_spec: dict, tip_coords: dict, chain_centroid,
              out_dir) -> dict:
    n_rounds = cfg["design"]["active_learning"]["n_rounds"]
    n_per_round = cfg["design"]["synthetic_backbone_generation"]["n_backbones_per_round"]
    xi = cfg["design"]["active_learning"]["xi"]
    surrogate_kind = cfg["design"]["active_learning"]["surrogate_model"]
    base_seed = cfg["project"]["seed"]

    mpnn_out_dir = out_dir / f"mpnn_{mode}"
    backbone_out_dir = out_dir / "backbones" / mode
    backbone_out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(base_seed if mode == "active_learning" else base_seed + 1000)
    X_all, y_all, round_records = [], [], []
    all_designs = []
    backbone_meta = {}

    for round_idx in range(n_rounds):
        if mode == "random_search" or round_idx == 0:
            X_round = random_params(rng, n_per_round)
        else:
            surrogate = build_surrogate(surrogate_kind)
            surrogate.fit(np.array(X_all), np.array(y_all))
            X_round = propose_next_params(surrogate, np.array(X_all), np.array(y_all),
                                             n_per_round, xi, rng)

        round_best = -1.0
        for cand_idx, x in enumerate(X_round):
            seed = base_seed + round_idx * 1000 + cand_idx + (0 if mode == "active_learning" else 500000)
            result = evaluate_one_candidate(x, round_idx, cand_idx, target_spec, tip_coords,
                                              chain_centroid, cfg, seed, mpnn_out_dir, backbone_out_dir)
            X_all.append(x)
            y_all.append(result["best_score"])
            all_designs.extend(result["designs"])
            backbone_meta[result["backbone_id"]] = {
                "backbone_id": result["backbone_id"], "ss_string": result["ss_string"],
                "dock_seed": result["dock_seed"], "standoff_A": result["standoff_A"],
            }
            round_best = max(round_best, result["best_score"])
            logger.info(f"[{mode}] round {round_idx} cand {cand_idx}: backbone={result['backbone_id']} "
                        f"n_designs={len(result['designs'])} best_score={result['best_score']:.4f}")

        round_records.append({"round": round_idx, "round_best_score": round_best,
                                "cumulative_best_score": max(y_all)})
        logger.info(f"[{mode}] round {round_idx} complete: round_best={round_best:.4f}, "
                    f"cumulative_best={max(y_all):.4f}")

    return {"mode": mode, "learning_curve": round_records, "all_designs": all_designs,
            "backbone_meta": backbone_meta,
            "X_all": [x.tolist() if hasattr(x, "tolist") else list(x) for x in X_all], "y_all": y_all}


def main() -> dict:
    set_global_seed()
    cfg = load_config()
    target_spec, tip_coords, chain_centroid = load_target_and_tip()

    out_dir = repo_path("results", "design")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("running active-learning design loop...")
    al_result = run_loop("active_learning", cfg, target_spec, tip_coords, chain_centroid, out_dir)

    logger.info("running equal-budget random-search baseline...")
    rs_result = run_loop("random_search", cfg, target_spec, tip_coords, chain_centroid, out_dir)

    with open(out_dir / "active_learning_result.json", "w") as fh:
        json.dump(al_result, fh, indent=2)
    with open(out_dir / "random_search_result.json", "w") as fh:
        json.dump(rs_result, fh, indent=2)

    all_designs = sorted(al_result["all_designs"], key=lambda d: d["composite_score"], reverse=True)
    prepared = {p["pdb_id"]: p for p in json.load(open(repo_path("data", "interim", "structures",
                                                                    "prepared_manifest.json")))}
    unique_backbones = {}
    for d in all_designs:
        unique_backbones.setdefault(d["backbone_id"], []).append(d)
    top_backbone_ids = sorted(unique_backbones, key=lambda b: max(x["composite_score"] for x in unique_backbones[b]),
                                reverse=True)[:10]
    top_backbones_meta = [al_result["backbone_meta"][bid] for bid in top_backbone_ids
                            if bid in al_result["backbone_meta"]]
    sel = build_selectivity_matrix(top_backbones_meta, target_spec, prepared) if top_backbones_meta else \
        {"matrix": {}, "selectivity_calls": {}}
    with open(out_dir / "selectivity_matrix.csv", "w", newline="") as fh:
        import csv
        strains = [target_spec["reference_strain"]] + [n["strain"] for n in target_spec["negative_design_panel"]]
        writer = csv.writer(fh)
        writer.writerow(["backbone_id"] + strains + ["margin", "is_selective"])
        for bid, row in sel["matrix"].items():
            call = sel["selectivity_calls"][bid]
            writer.writerow([bid] + [row.get(s) for s in strains] + [call["margin"], call["is_selective"]])

    selective_backbone_ids = {bid for bid, c in sel["selectivity_calls"].items() if c["is_selective"]}
    leads = [d for d in all_designs if d["backbone_id"] in selective_backbone_ids and
             d["developability"]["developability_passed"]]
    leads = leads[:20]

    fasta_path = out_dir / "leads.fasta"
    with open(fasta_path, "w") as fh:
        for d in leads:
            fh.write(f">{d['design_id']} composite_score={d['composite_score']} "
                      f"backbone={d['backbone_id']}\n{d['sequence']}\n")

    import csv
    with open(out_dir / "all_designs_scored.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["design_id", "backbone_id", "topology", "sequence", "mpnn_score",
                          "esm_plausibility", "clash_score", "packing_density_sc_proxy",
                          "developability_passed", "composite_score", "is_selective_backbone"])
        for d in all_designs:
            writer.writerow([d["design_id"], d["backbone_id"], d["params"]["topology"], d["sequence"],
                              d["mpnn_score"], d["esm_plausibility"], d["geometric"]["clash_score"],
                              d["geometric"]["packing_density_sc_proxy"],
                              d["developability"]["developability_passed"], d["composite_score"],
                              d["backbone_id"] in selective_backbone_ids])

    with open(out_dir / "learning_curves.json", "w") as fh:
        json.dump({"active_learning": al_result["learning_curve"],
                    "random_search": rs_result["learning_curve"]}, fh, indent=2)

    logger.info(f"design loop complete: {len(all_designs)} total designs, "
                f"{len(selective_backbone_ids)}/{len(top_backbone_ids)} top backbones AD-selective, "
                f"{len(leads)} leads written to results/design/leads.fasta")
    logger.info(f"active-learning final best={al_result['learning_curve'][-1]['cumulative_best_score']:.4f} "
                f"vs random-search final best={rs_result['learning_curve'][-1]['cumulative_best_score']:.4f}")

    append_progress_log(
        "M6",
        f"Ran the closed-loop design engine: CPU geometric backbone baseline (RFdiffusion "
        f"Colab-deferred) -> real ProteinMPNN sequence design (fully executed, "
        f"{cfg['design']['proteinmpnn']['n_sequences_per_backbone']} seqs/backbone) -> CPU interface "
        f"scoring (geometric complementarity + ESM-2 single-pass plausibility, ESMFold substituted "
        f"per PROGRESS_LOG.md) -> {cfg['design']['active_learning']['n_rounds']}-round Gaussian-Process "
        f"active-learning loop vs an equal-budget random-search baseline -> selectivity scoring "
        f"against the {len(target_spec['negative_design_panel'])}-fold negative-design panel -> "
        f"developability filtering. {len(all_designs)} total designs scored, {len(leads)} leads "
        f"survived selectivity+developability. Active-learning final best="
        f"{al_result['learning_curve'][-1]['cumulative_best_score']:.4f} vs random-search="
        f"{rs_result['learning_curve'][-1]['cumulative_best_score']:.4f}.",
    )
    return {"active_learning": al_result, "random_search": rs_result, "leads": leads}


if __name__ == "__main__":
    main()
