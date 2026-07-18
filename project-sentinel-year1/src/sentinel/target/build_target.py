"""M5 — Design-target definition. Fuses the M2 strain fingerprint (which
residues are AD-selectively exposed) with the M4 fibril-tip RMSF (which
residues are rigid "anchors" a binder can reliably engage) into a single
machine-readable design-target spec that M6 consumes.

Run: python -m sentinel.target.build_target
"""
from __future__ import annotations

import json

from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger

logger = get_logger(__name__)


def load_rigid_anchor_residues(md_results: dict, tag: str = "fibril_tip",
                                 rmsf_percentile: float = 40.0) -> list[int]:
    """Residues in the bottom `rmsf_percentile` of per-residue RMSF (i.e. the
    most rigid, most reliable anchor points) at the fibril tip."""
    import numpy as np
    rmsf = md_results[tag]["analysis"]["rmsf"]
    res_ids = rmsf["res_id"]
    rmsf_vals = rmsf["rmsf_nm"]
    threshold = float(np.percentile(rmsf_vals, rmsf_percentile))
    rigid = sorted({res_ids[i] for i in range(len(res_ids)) if rmsf_vals[i] <= threshold})
    return rigid


def main() -> dict:
    cfg = load_config()
    atlas_dir = repo_path("results", "atlas")
    md_dir = repo_path("results", "md")
    target_dir = repo_path("results", "target")
    target_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = json.load(open(atlas_dir / "ad_strain_fingerprint.json"))
    md_results = json.load(open(md_dir / "md_results_full.json"))

    top_n = cfg["atlas"]["fingerprint_top_n_hotspots"]
    hotspot_resids = [h["res_id"] for h in fingerprint["top_hotspots"][:top_n]]

    rigid_anchors = load_rigid_anchor_residues(md_results)
    anchor_hotspots = sorted(set(hotspot_resids) & set(rigid_anchors))

    ref_entry = next(e for e in cfg["data"]["strain_panel"] if e["strain"] == cfg["data"]["reference_strain"])
    prepared = json.load(open(repo_path("data", "interim", "structures", "prepared_manifest.json")))
    ref_prepared = next(p for p in prepared if p["pdb_id"] == ref_entry["id"])

    negative_panel = []
    for strain_name in cfg["target"]["negative_design_panel"]:
        entry = next(e for e in cfg["data"]["strain_panel"] if e["strain"] == strain_name)
        p = next(pp for pp in prepared if pp["pdb_id"] == entry["id"])
        negative_panel.append({
            "strain": strain_name, "pdb_id": entry["id"],
            "stack_pdb": p["stack_pdb"], "tip_chain": p["stack_chains_used"][-1],
            "core_start": entry["core_start"], "core_end": entry["core_end"],
        })

    target_spec = {
        "target_name": "AD_tau_fibril_templating_tip_capper",
        "reference_strain": cfg["data"]["reference_strain"],
        "reference_pdb_id": ref_entry["id"],
        "reference_stack_pdb": ref_prepared["stack_pdb"],
        "reference_tip_chain": ref_prepared["stack_chains_used"][-1],
        "core_residue_range": [ref_entry["core_start"], ref_entry["core_end"]],
        "hotspot_residues": hotspot_resids,
        "rigid_anchor_residues": rigid_anchors,
        "hotspot_and_anchor_residues": anchor_hotspots,
        "conditioning_residues_for_rfdiffusion": anchor_hotspots if len(anchor_hotspots) >= 5 else hotspot_resids,
        "templating_tip_geometry": {
            "description": "the terminal cross-beta layer of the AD PHF stack (see results/atlas/"
                            "per_strain_characterization.json, strain AD_PHF) — the surface a new "
                            "tau monomer would dock onto to extend the fibril; a capper must "
                            "complement its exposed backbone H-bond donors/acceptors and hotspot "
                            "side chains without occluding residues shared with the other 7 folds.",
            "n_beta_strands_per_layer": None,  # filled below from atlas
        },
        "binder_length_range": [cfg["target"]["binder_length_min"], cfg["target"]["binder_length_max"]],
        "negative_design_panel": negative_panel,
        "provenance": {
            "hotspots_source": "results/atlas/ad_strain_fingerprint.json (top_hotspots, boundary-excluded)",
            "anchors_source": f"results/md/md_results_full.json (fibril_tip RMSF, bottom "
                               f"{100 - 40}th... percentile <=40 = most rigid)",
        },
    }

    per_strain = json.load(open(atlas_dir / "per_strain_characterization.json"))
    target_spec["templating_tip_geometry"]["n_beta_strands_per_layer"] = \
        per_strain[cfg["data"]["reference_strain"]]["topology"]["n_beta_strands_per_layer"]

    out_path = target_dir / "ad_capper_target.json"
    with open(out_path, "w") as fh:
        json.dump(target_spec, fh, indent=2)

    logger.info(f"design target spec: {len(hotspot_resids)} hotspots, {len(rigid_anchors)} rigid anchors, "
                f"{len(anchor_hotspots)} overlap (hotspot AND rigid) -> "
                f"{len(target_spec['conditioning_residues_for_rfdiffusion'])} conditioning residues, "
                f"{len(negative_panel)}-fold negative-design panel.")
    append_progress_log(
        "M5",
        f"Built results/target/ad_capper_target.json by fusing the M2 AD strain fingerprint "
        f"({len(hotspot_resids)} top hotspots) with M4 fibril-tip RMSF ({len(rigid_anchors)} rigid "
        f"anchor residues at <=40th percentile RMSF); {len(anchor_hotspots)} residues satisfy both "
        f"criteria and are used as the RFdiffusion conditioning set "
        f"({'anchor-hotspot overlap' if len(anchor_hotspots) >= 5 else 'fell back to hotspots alone: too few anchor/hotspot overlaps'}). "
        f"Negative-design panel: {', '.join(n['strain'] for n in negative_panel)}.",
    )
    return target_spec


if __name__ == "__main__":
    main()
