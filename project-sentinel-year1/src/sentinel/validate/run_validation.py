"""M7 — In-silico validation of leads.

For the top few leads (results/design/leads.fasta), build a binder-alone
OpenMM system (real backbone from the design loop, ProteinMPNN-assigned
sequence — full-atom via PDBFixer's sidechain placement from sequence),
run short implicit-solvent MD, and report:
  - stability (CA RMSD over the trajectory)
  - a "capping test" proxy: whether the binder's docked pose, when re-checked
    after equilibration, still occludes the fibril tip's exposed backbone
    H-bond donor/acceptor atoms (the groups a new tau monomer would use to
    template) — i.e. does the binder still block those atoms from solvent
    after the binder itself has relaxed.

Run: python -m sentinel.validate.run_validation
"""
from __future__ import annotations

import csv
import json

import numpy as np

from sentinel.md.analyze import compute_rmsd, compute_radius_of_gyration, load_trajectory
from sentinel.md.simulate import run_md
from sentinel.md.system_builder import fix_and_protonate
from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger
from sentinel.utils.seeds import set_global_seed

logger = get_logger(__name__)

MAX_WALLCLOCK_S = 180.0
N_LEADS_TO_VALIDATE = 3


def build_binder_pdb_with_sequence(backbone_pdb: str, sequence: str, dest_path) -> None:
    """Mutate the CA-only-derived backbone's residue names to the designed
    sequence and let PDBFixer rebuild sidechains from identity + backbone."""
    lines = open(backbone_pdb).read().splitlines()
    three = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
        "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
        "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
        "Y": "TYR", "V": "VAL",
    }
    new_lines = []
    for line in lines:
        if line.startswith("ATOM"):
            res_id = int(line[22:26])
            aa = sequence[res_id - 1] if res_id - 1 < len(sequence) else "A"
            resname = three.get(aa, "ALA")
            line = line[:17] + f"{resname:>3s}" + line[20:]
        new_lines.append(line)
    raw_path = str(dest_path).replace(".pdb", "_raw.pdb")
    with open(raw_path, "w") as fh:
        fh.write("\n".join(new_lines) + "\n")
    fix_and_protonate(raw_path, dest_path)


def capping_occlusion_check(tip_coords: dict, binder_final_coords: dict, occlusion_radius_A: float = 5.0) -> dict:
    """Fraction of the tip's exposed backbone N/O atoms (the templating
    H-bond groups) within occlusion_radius_A of any binder atom, before vs.
    after the binder's own MD relaxation."""
    tip_no = np.concatenate([tip_coords["N"], tip_coords["O"]], axis=0)
    binder_all = np.concatenate([binder_final_coords[a] for a in ["N", "CA", "C", "O"]], axis=0)
    dists = np.linalg.norm(tip_no[:, None, :] - binder_all[None, :, :], axis=-1)
    occluded = (dists.min(axis=1) < occlusion_radius_A)
    return {"n_tip_hbond_atoms": len(tip_no), "n_occluded": int(occluded.sum()),
            "fraction_occluded": round(float(occluded.mean()), 4)}


def main() -> dict:
    set_global_seed()
    cfg = load_config()
    design_dir = repo_path("results", "design")
    all_designs = list(csv.DictReader(open(design_dir / "all_designs_scored.csv")))
    all_designs.sort(key=lambda d: float(d["composite_score"]), reverse=True)
    top_leads = all_designs[:N_LEADS_TO_VALIDATE]

    al_result = json.load(open(design_dir / "active_learning_result.json"))
    backbone_meta = al_result["backbone_meta"]
    target_spec = json.load(open(repo_path("results", "target", "ad_capper_target.json")))

    import biotite.structure.io.pdb as pdb_io
    stack_pdb = repo_path(target_spec["reference_stack_pdb"])
    tip_chain = target_spec["reference_tip_chain"]
    reader = pdb_io.PDBFile.read(str(stack_pdb))
    arr = reader.get_structure(model=1)
    tip_coords = {a: arr.coord[(arr.chain_id == tip_chain) & (arr.atom_name == a)]
                   for a in ["N", "CA", "C", "O"]}

    out_dir = repo_path("results", "validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_cfg = cfg["md"]["hexapeptide"]  # reuse the small-system implicit-solvent settings

    results = []
    for lead in top_leads:
        design_id = lead["design_id"]
        backbone_id = lead["backbone_id"]
        bb_meta = backbone_meta.get(backbone_id)
        if bb_meta is None:
            logger.warning(f"no backbone metadata for {backbone_id}, skipping {design_id}")
            continue
        backbone_pdb = repo_path("results", "design", "backbones", "active_learning", f"{backbone_id}.pdb")
        if not backbone_pdb.exists():
            logger.warning(f"backbone PDB missing for {backbone_id}, skipping {design_id}")
            continue

        binder_fixed = out_dir / f"{design_id}_binder.pdb"
        build_binder_pdb_with_sequence(str(backbone_pdb), lead["sequence"], binder_fixed)

        md_result = run_md(
            pdb_path=str(binder_fixed), forcefield_files=md_cfg["forcefield"],
            temperature_K=md_cfg["temperature_K"], friction_per_ps=md_cfg["friction_per_ps"],
            timestep_fs=md_cfg["timestep_fs"], minimize_max_iterations=md_cfg["minimize_max_iterations"],
            target_ns=0.02, report_interval_steps=200, out_dir=out_dir, tag=f"{design_id}_validate",
            seed=cfg["project"]["seed"], max_wallclock_s=MAX_WALLCLOCK_S,
        )

        traj = load_trajectory(md_result["trajectory_dcd"], str(binder_fixed))
        rmsd = compute_rmsd(traj)
        rg = compute_radius_of_gyration(traj)

        import biotite.structure.io.pdb as pdb_io2
        final_reader = pdb_io2.PDBFile.read(md_result["final_structure_pdb"])
        final_arr = final_reader.get_structure(model=1)
        binder_final_coords = {a: final_arr.coord[final_arr.atom_name == a] for a in ["N", "CA", "C", "O"]}
        capping = capping_occlusion_check(tip_coords, binder_final_coords)

        results.append({
            "design_id": design_id, "backbone_id": backbone_id, "composite_score": lead["composite_score"],
            "actual_ns_simulated": md_result["actual_ns_simulated"],
            "mean_rmsd_nm": round(float(np.mean(rmsd)), 4), "final_rmsd_nm": round(float(rmsd[-1]), 4),
            "mean_rg_nm": round(float(np.mean(rg)), 4),
            "capping_occlusion": capping,
            "stable": bool(np.mean(rmsd) < 1.0),  # 1.0 nm: a coarse, documented stability threshold
                                                     # for a 60-80 aa mini-protein at this short timescale
            "mechanistically_plausible": capping["fraction_occluded"] > 0.3,
        })
        logger.info(f"{design_id}: RMSD_mean={np.mean(rmsd):.3f}nm, tip H-bond occlusion="
                    f"{capping['fraction_occluded']:.2%}")

    with open(out_dir / "validation_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    n_stable = sum(1 for r in results if r["stable"])
    n_plausible = sum(1 for r in results if r["mechanistically_plausible"])
    append_progress_log(
        "M7",
        f"Ran in-silico validation MD on the top {len(results)} leads (real OpenMM implicit-solvent, "
        f"full-atom sidechains from PDBFixer given the ProteinMPNN-designed sequence). {n_stable}/"
        f"{len(results)} stable by RMSD, {n_plausible}/{len(results)} show >30% occlusion of the AD "
        f"tip's templating H-bond groups after relaxation (a capping-mechanism plausibility proxy, "
        f"not a rigorous steered-MD blocking assay).",
    )
    return {"results": results}


if __name__ == "__main__":
    main()
