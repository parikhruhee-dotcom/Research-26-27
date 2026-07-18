"""M4 — Molecular dynamics orchestrator.

System 1: PHF6 and PHF6* hexapeptides, implicit-solvent MD, starting from
real crystal coordinates (2ON9/5V5C). Analyzed for RMSD/RMSF/Rg/secondary
structure to characterize how ordered/floppy the isolated nucleating
segments are.

System 2: the AD fibril growing tip (a truncated 2-layer stack — see
PROGRESS_LOG.md M4 for why 3 layers as prepared in M1d was too large for this
2-core CPU machine even with a nonbonded cutoff). Analyzed for per-residue
RMSF to identify rigid "anchor" residues at the templating surface.

Every run's ACTUAL simulated ns (measured wall-clock throughput on this
machine, never the config target) is what gets logged and reported, per the
brief's honesty rule.

Run: python -m sentinel.md.run_md
"""
from __future__ import annotations

import json

from sentinel.md.analyze import full_analysis
from sentinel.md.simulate import run_md
from sentinel.md.system_builder import build_fibril_tip_system, build_hexapeptide_system
from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger
from sentinel.utils.seeds import set_global_seed

logger = get_logger(__name__)

MAX_WALLCLOCK_HEXAPEPTIDE_S = 420.0
MAX_WALLCLOCK_FIBRIL_TIP_S = 300.0


def run_hexapeptide_system(motif: str, cfg: dict, out_dir) -> dict:
    pdb_path = build_hexapeptide_system(motif)
    md_cfg = cfg["md"]["hexapeptide"]
    result = run_md(
        pdb_path=pdb_path, forcefield_files=md_cfg["forcefield"],
        temperature_K=md_cfg["temperature_K"], friction_per_ps=md_cfg["friction_per_ps"],
        timestep_fs=md_cfg["timestep_fs"], minimize_max_iterations=md_cfg["minimize_max_iterations"],
        target_ns=md_cfg["target_ns_cpu"], report_interval_steps=md_cfg["report_interval_steps"],
        out_dir=out_dir, tag=motif, seed=cfg["project"]["seed"],
        max_wallclock_s=MAX_WALLCLOCK_HEXAPEPTIDE_S,
    )
    analysis = full_analysis(result["trajectory_dcd"], pdb_path)
    return {**result, "analysis": analysis}


def run_fibril_tip_system(cfg: dict, out_dir, strain_id: str = "5O3L") -> dict:
    stack_pdb = repo_path("data", "interim", "structures", f"{strain_id}_stack.pdb")
    import biotite.structure.io.pdb as pdb_io
    reader = pdb_io.PDBFile.read(str(stack_pdb))
    arr = reader.get_structure(model=1)
    chains = sorted(set(arr.chain_id))
    two_layers = chains[:2]  # see module docstring: truncated from 3 to 2 layers for CPU tractability
    reduced_dir = repo_path("data", "interim", "md")
    reduced_dir.mkdir(parents=True, exist_ok=True)
    raw_path = reduced_dir / f"{strain_id}_2layer_raw.pdb"
    sub = arr[(arr.chain_id == two_layers[0]) | (arr.chain_id == two_layers[1])]
    writer = pdb_io.PDBFile()
    writer.set_structure(sub)
    writer.write(str(raw_path))

    pdb_path = build_fibril_tip_system(str(raw_path), f"{strain_id}_2layer")
    md_cfg = cfg["md"]["fibril_tip"]
    result = run_md(
        pdb_path=pdb_path, forcefield_files=md_cfg["forcefield"],
        temperature_K=md_cfg["temperature_K"], friction_per_ps=md_cfg["friction_per_ps"],
        timestep_fs=md_cfg["timestep_fs"], minimize_max_iterations=md_cfg["minimize_max_iterations"],
        target_ns=md_cfg["target_ns_cpu"], report_interval_steps=md_cfg["report_interval_steps"],
        out_dir=out_dir, tag=f"{strain_id}_fibril_tip", seed=cfg["project"]["seed"],
        max_wallclock_s=MAX_WALLCLOCK_FIBRIL_TIP_S,
    )
    analysis = full_analysis(result["trajectory_dcd"], pdb_path)
    return {**result, "analysis": analysis, "layers_used": two_layers,
            "note": "truncated to 2 stacked layers (from the 3-layer M1d model) for CPU tractability"}


def main() -> dict:
    set_global_seed()
    cfg = load_config()
    out_dir = repo_path("results", "md")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for motif in ["PHF6", "PHF6_star"]:
        logger.info(f"running hexapeptide MD for {motif}...")
        results[motif] = run_hexapeptide_system(motif, cfg, out_dir)

    logger.info("running fibril growing-tip MD...")
    results["fibril_tip"] = run_fibril_tip_system(cfg, out_dir)

    summary = {}
    for tag, res in results.items():
        summary[tag] = {
            "actual_ns_simulated": res["actual_ns_simulated"],
            "target_ns": res["target_ns"],
            "capped_by_wallclock_budget": res["capped_by_wallclock_budget"],
            "mean_rmsd_nm": sum(res["analysis"]["rmsd_nm"]) / len(res["analysis"]["rmsd_nm"]),
            "mean_rg_nm": sum(res["analysis"]["radius_of_gyration_nm"]) / len(res["analysis"]["radius_of_gyration_nm"]),
            "mean_beta_fraction": sum(res["analysis"]["secondary_structure"]["beta_fraction_per_frame"]) /
                                   len(res["analysis"]["secondary_structure"]["beta_fraction_per_frame"]),
        }

    with open(out_dir / "md_results_full.json", "w") as fh:
        json.dump(results, fh, indent=2)
    with open(out_dir / "md_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    scaleup_notes = out_dir / "md_scaleup.md"
    scaleup_notes.write_text(_scaleup_markdown(cfg, summary))

    for tag, s in summary.items():
        logger.info(f"{tag}: {s['actual_ns_simulated']} ns simulated (target {s['target_ns']} ns), "
                    f"mean RMSD={s['mean_rmsd_nm']:.3f} nm, mean Rg={s['mean_rg_nm']:.3f} nm, "
                    f"mean beta-fraction={s['mean_beta_fraction']:.3f}")

    append_progress_log(
        "M4",
        "Ran real OpenMM implicit-solvent (amber14 + GBn2) MD for PHF6, PHF6*, and a truncated "
        "2-layer AD fibril growing tip. Actual ns simulated is measured wall-clock throughput on "
        "this 2-core CPU sandbox, not the config target (a 3420-atom 3-layer fibril-tip system with "
        "all-pairs NoCutoff GBSA did not finish 100 minimization iterations in >4 minutes; switched to "
        "a documented CutoffNonPeriodic (1.5 nm) nonbonded scheme and truncated to 2 layers to make the "
        "system CPU-tractable). Summary: " +
        "; ".join(f"{tag}={s['actual_ns_simulated']}ns" for tag, s in summary.items()) +
        f". Full-scale (longer ns, all layers) reproduction commands are in results/md/md_scaleup.md.",
    )
    return results


def _scaleup_markdown(cfg: dict, summary: dict) -> str:
    lines = [
        "# MD scale-up: reproducing at full length / full GPU speed",
        "",
        "This machine is a 2-core CPU sandbox with no GPU. Every MD run in this",
        "module ran for real, but was capped to a wall-clock budget and (for the",
        "fibril-tip system) truncated in size so it would finish. The exact",
        "achieved simulation length for each system:",
        "",
    ]
    for tag, s in summary.items():
        lines.append(f"- **{tag}**: {s['actual_ns_simulated']} ns "
                      f"(target was {s['target_ns']} ns; capped_by_wallclock_budget="
                      f"{s['capped_by_wallclock_budget']})")
    lines += [
        "",
        "## To reproduce at full scale on a GPU workstation",
        "",
        "```bash",
        "python -c \"",
        "from sentinel.md.run_md import run_hexapeptide_system, run_fibril_tip_system",
        "from sentinel.utils.config import load_config, repo_path",
        "cfg = load_config()",
        "# 1. Edit config.yaml: md.hexapeptide.target_ns_cpu / md.fibril_tip.target_ns_cpu to the",
        "#    desired full length (e.g. 100-500 ns), and set a CUDA platform in simulate.build_simulation",
        "#    (Platform.getPlatformByName('CUDA') instead of 'CPU').",
        "# 2. In run_md.run_fibril_tip_system, use all layers in the M1d *_stack.pdb (not a 2-layer",
        "#    truncation) and raise MAX_WALLCLOCK_FIBRIL_TIP_S / drop the wall-clock cap entirely.",
        "run_hexapeptide_system('PHF6', cfg, repo_path('results','md'))",
        "run_fibril_tip_system(cfg, repo_path('results','md'))",
        "\"",
        "```",
        "",
        "No GPU-specific model weights are needed for this module (OpenMM MD runs",
        "identically on CPU/GPU, only faster) — there is no Colab notebook for M4;",
        "the code above is the full scale-up path.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
