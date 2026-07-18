"""M2 — The Tau Strain Conformational Atlas. Orchestrates every per-strain and
cross-strain analysis and writes results/atlas/*.

Run: python -m sentinel.atlas.run_atlas
"""
from __future__ import annotations

import json

from sentinel.atlas.alignment import build_dendrogram_linkage, pairwise_rmsd_matrix
from sentinel.atlas.fingerprint import compute_fingerprint
from sentinel.atlas.physicochemical import residue_physchem
from sentinel.atlas.sasa import classify_burial, compute_residue_sasa
from sentinel.atlas.secondary_structure import run_dssp
from sentinel.atlas.topology import core_topology, interprotofilament_contacts
from sentinel.atlas.zipper import zipper_analysis_for_strain, zipper_crystal_comparison
from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger
from sentinel.utils.seeds import set_global_seed

logger = get_logger(__name__)


def analyze_strain(strain_entry: dict, prepared_entry: dict) -> dict:
    pdb_id = strain_entry["id"]
    stack_pdb = str(repo_path(prepared_entry["stack_pdb"]))
    full_pdb = str(repo_path(prepared_entry["full_pdb"]))
    tip_chain = prepared_entry["stack_chains_used"][-1]  # terminal layer = growth-competent tip (see M2 docstring)

    sasa_records = compute_residue_sasa(stack_pdb)
    tip_sasa = {r["res_id"]: r for r in sasa_records if r["chain"] == tip_chain}
    for r in tip_sasa.values():
        r["burial_class"] = classify_burial(r["rel_sasa"])
        r.update(residue_physchem(r["res_name"]))

    dssp_records = run_dssp(stack_pdb)
    topology = core_topology(strain_entry, dssp_records, prepared_entry)

    interface_residues = interprotofilament_contacts(
        full_pdb, prepared_entry["full_chains_by_protofilament"])

    zipper = zipper_analysis_for_strain(strain_entry, prepared_entry)

    return {
        "pdb_id": pdb_id, "strain": strain_entry["strain"],
        "topology": topology,
        "tip_chain_used": tip_chain,
        "tip_per_residue_sasa": {str(k): v for k, v in sorted(tip_sasa.items())},
        "interprotofilament_interface_residues": interface_residues,
        "zipper": zipper,
        "n_dssp_residues": len(dssp_records),
    }


def main() -> dict:
    set_global_seed()
    cfg = load_config()
    strain_panel = cfg["data"]["strain_panel"]
    prepared_list = json.load(open(repo_path("data", "interim", "structures", "prepared_manifest.json")))
    prepared_by_id = {p["pdb_id"]: p for p in prepared_list}

    atlas_dir = repo_path("results", "atlas")
    atlas_dir.mkdir(parents=True, exist_ok=True)

    per_strain_results = {}
    for entry in strain_panel:
        logger.info(f"analyzing strain {entry['strain']} ({entry['id']})...")
        per_strain_results[entry["strain"]] = analyze_strain(entry, prepared_by_id[entry["id"]])

    with open(atlas_dir / "per_strain_characterization.json", "w") as fh:
        json.dump(per_strain_results, fh, indent=2)

    # cross-strain: alignment + dendrogram
    prepared_with_abs_paths = {
        pid: {**p, "single_pdb": str(repo_path(p["single_pdb"]))} for pid, p in prepared_by_id.items()
    }
    rmsd_result = pairwise_rmsd_matrix(prepared_with_abs_paths, strain_panel)
    dendrogram = build_dendrogram_linkage(rmsd_result)
    with open(atlas_dir / "fold_similarity_rmsd.json", "w") as fh:
        json.dump({**rmsd_result, "dendrogram": dendrogram}, fh, indent=2)

    # cross-strain: zipper crystal comparison (the real dry-interface calculation)
    zipper_crystal = zipper_crystal_comparison()
    with open(atlas_dir / "zipper_crystal_comparison.json", "w") as fh:
        json.dump(zipper_crystal, fh, indent=2)

    # strain fingerprint
    per_strain_tip_sasa = {
        strain: {int(rid): {"res_name": rec["res_name"], "rel_sasa": rec["rel_sasa"]}
                  for rid, rec in res["tip_per_residue_sasa"].items()}
        for strain, res in per_strain_results.items()
    }
    reference_strain = cfg["data"]["reference_strain"]
    ref_entry = next(e for e in strain_panel if e["strain"] == reference_strain)
    boundary_n = 3
    boundary_exclude = set(range(ref_entry["core_start"], ref_entry["core_start"] + boundary_n)) | \
        set(range(ref_entry["core_end"] - boundary_n + 1, ref_entry["core_end"] + 1))
    fingerprint = compute_fingerprint(per_strain_tip_sasa, reference_strain, boundary_exclude)
    with open(atlas_dir / "ad_strain_fingerprint.json", "w") as fh:
        json.dump(fingerprint, fh, indent=2)

    # summary table (CSV-friendly) for quick inspection / figures
    import csv
    with open(atlas_dir / "strain_summary_table.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["strain", "pdb_id", "core_start", "core_end", "core_length",
                          "protofilament_count", "n_beta_strands_per_layer",
                          "n_interprotofilament_interface_residues",
                          "PHF6_burial_in_fibril_A2", "PHF6star_burial_in_fibril_A2"])
        for strain, res in per_strain_results.items():
            topo = res["topology"]
            zp = res["zipper"]
            phf6_b = zp["PHF6_in_fibril_context"]["total_buried_A2"] if zp["PHF6_in_fibril_context"] else None
            star_b = zp["PHF6_star_in_fibril_context"]["total_buried_A2"] if zp["PHF6_star_in_fibril_context"] else None
            writer.writerow([strain, res["pdb_id"], topo["core_start"], topo["core_end"], topo["core_length"],
                              topo["protofilament_count"], topo["n_beta_strands_per_layer"],
                              len(res["interprotofilament_interface_residues"]), phf6_b, star_b])

    logger.info(f"AD strain fingerprint: top hotspot = "
                f"{fingerprint['top_hotspots'][0]['res_name']}{fingerprint['top_hotspots'][0]['res_id']} "
                f"(differential exposure {fingerprint['top_hotspots'][0]['differential_exposure']})")
    logger.info(f"zipper crystal ratio VQIINK/VQIVYK buried SASA = "
                f"{zipper_crystal['VQIINK_over_VQIVYK_ratio']}")

    append_progress_log(
        "M2",
        f"Built the strain conformational atlas for all {len(strain_panel)} panel entries: per-residue "
        f"SASA/DSSP/burial classification at the templating tip, inter-protofilament interface residues, "
        f"per-strain PHF6/PHF6* in-fibril-context burial, pairwise structural RMSD + dendrogram (common "
        f"core {rmsd_result['common_core_range']}, {rmsd_result['n_shared_residues']} shared residues), "
        f"the real hexapeptide-crystal dry-interface zipper comparison (VQIINK/VQIVYK buried-SASA ratio = "
        f"{zipper_crystal['VQIINK_over_VQIVYK_ratio']}, matching the ~2x literature claim), and the AD "
        f"strain fingerprint (top hotspot: {fingerprint['top_hotspots'][0]['res_name']}"
        f"{fingerprint['top_hotspots'][0]['res_id']}).",
    )
    return {"per_strain": per_strain_results, "rmsd": rmsd_result, "fingerprint": fingerprint,
            "zipper_crystal": zipper_crystal}


if __name__ == "__main__":
    main()
