"""Writes results/design/GPU_TIER_STATUS.md — the single, honest place that
states exactly what ran locally at full/reduced scale vs. what is prepared
and Colab-deferred (brief Part 4 / rule 0.2's compute-tier honesty rule).

Run: python -m sentinel.design.gpu_tier_status
"""
from __future__ import annotations

from sentinel.utils.config import load_config, repo_path


def main() -> None:
    cfg = load_config()
    import json
    compute_profile = json.load(open(repo_path("results", "compute_profile.json")))

    content = f"""# GPU-tier status — what ran locally vs. what is Colab-deferred

Detected compute tier on this build: **{compute_profile['tier']}**
({compute_profile['cpu_cores']} CPU cores, {compute_profile['ram_gb']} GB RAM,
CUDA available: {compute_profile['cuda_available']}). Full profile:
`results/compute_profile.json`.

## M6a — Backbone generation

- **Specified tool:** RFdiffusion (RosettaCommons/RFdiffusion). Requires a
  CUDA-capable GPU (its SE(3)-Transformer denoiser has no practical CPU
  inference path).
- **What ran on this machine:** NOT RFdiffusion. A documented, deterministic,
  non-ML **CPU geometric baseline** (`src/sentinel/design/backbone_gen.py`):
  a 4-topology idealized secondary-structure scaffold library (helix-hairpin,
  three-helix-bundle, helix-strand-helix, long single helix), built with real
  peptide geometry (NeRF construction from standard bond lengths/angles/ideal
  Ramachandran dihedrals — `src/sentinel/design/geometry.py`, unit-tested
  against known ideal-alpha-helix rise), then rigid-body docked onto the AD
  templating tip's hotspot-residue centroid with sampled approach angles.
  This is explicitly weaker than a trained generative model — it does not
  learn shape complementarity the way RFdiffusion does — but it kept every
  downstream stage (ProteinMPNN, scoring, active learning, selectivity)
  genuinely exercised end-to-end on real (if geometrically naive) backbones.
- **Full-scale GPU path:** `notebooks/colab_rfdiffusion.ipynb` — one-click,
  pre-filled with this build's real target spec (hotspot residues, core
  range, PDB ID). Drop resulting backbones into
  `results/design/backbones/rfdiffusion/` and re-run `make design`; the
  active-learning loop reads backbones from disk with no code changes needed
  (same PDB format as the CPU baseline's output).
- **Labeling:** every design record in `results/design/all_designs_scored.csv`
  and the backbone manifest states its `source` field explicitly
  (`cpu_geometric_baseline` — never silently presented as RFdiffusion output).

## M6c — Complex folding / interface scoring

- **Specified tool:** AlphaFold2-multimer (via localcolabfold) or Boltz.
  Needs a GPU for practical runtime (large transformer, structure module).
- **What ran on this machine:** NOT AlphaFold2-multimer. The brief's own
  CPU-runnable substitute (`src/sentinel/design/interface_scorer.py`):
  geometric/energetic complementarity (buried SASA via freesasa, a backbone
  clash score, an H-bond-geometry proxy, a packing-density Sc proxy) plus
  ESM-2 (`{cfg['design']['scoring']['esm_model']}`, ~30 MB) single-pass
  sequence-plausibility scoring — substituted for full ESMFold (~15 GB,
  infeasible on this sandbox's disk) and for per-position masked
  pseudo-perplexity (would need one forward pass per residue per sequence —
  too slow at this design loop's candidate volume). Both substitutions are
  logged in `results/PROVENANCE.json` and PROGRESS_LOG.md M6.
- **Full-scale GPU path:** `notebooks/colab_af2_multimer.ipynb` — one-click,
  folds every lead in `results/design/leads.fasta` against the real tau
  target and applies this project's own thresholds (interface pAE <
  {cfg['design']['scoring']['interface_pae_max']}, pLDDT >
  {cfg['design']['scoring']['plddt_min']}, ipTM >
  {cfg['design']['scoring']['iptm_min']}).

## Fully executed on CPU (no substitution, no deferral)

Per the brief's Part 4 compute strategy, everything else in Year 1 ran for
real on this CPU sandbox: the entire strain atlas (M2), the aggregation
engine (M3, validated on first attempt), molecular dynamics (M4, real OpenMM
trajectories, actual ns logged — not the config target), the design target
spec (M5), **real ProteinMPNN** sequence design (M6b — this is the one
GPU-associated-sounding tool that DOES run natively on CPU, and is fully
executed, not substituted), the {cfg['design']['active_learning']['n_rounds']}
-round active-learning loop and its random-search baseline (M6d), selectivity
scoring against the full negative-design panel (M6e), in-silico lead
validation (M7), the biosensor concept (M8), and all benchmarks (M9).
"""
    out_path = repo_path("results", "design", "GPU_TIER_STATUS.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)


if __name__ == "__main__":
    main()
