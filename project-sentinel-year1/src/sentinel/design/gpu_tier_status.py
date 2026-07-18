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
- **Verified, not assumed:** the real RFdiffusion repository was cloned to
  this machine and its actual install path was attempted and inspected
  (not just asserted infeasible from prior knowledge):
  - `env/SE3nv.yml` (RFdiffusion's own specified conda environment) pins
    `cudatoolkit=11.1` and `dgl-cuda11.1` — packages that require a matching
    NVIDIA driver/CUDA runtime to install correctly, not merely to run fast.
  - `pip install -e .` on this machine fails at dependency resolution:
    `ERROR: Could not find a version that satisfies the requirement
    se3-transformer (from rfdiffusion) (from versions: none)` — the
    `se3-transformer` package is not published on PyPI at all; it only
    exists as the CUDA-dependent bundled subpackage in
    `env/SE3Transformer/`, whose own `requirements.txt` pulls in `pynvml`
    (the NVIDIA Management Library Python bindings).
  - `env/SE3Transformer/se3_transformer/model/layers/convolution.py`
    imports `torch.cuda.nvtx` unconditionally at module load time.
  This is a genuine, source-level-verified hard CUDA dependency, not a
  "would be slow on CPU" situation — the package cannot even be installed,
  let alone run, without a CUDA-capable GPU present.
- **What ran on this machine instead:** a documented, deterministic,
  non-ML **CPU geometric baseline** (`src/sentinel/design/backbone_gen.py`,
  `topology_builder.py`): a 4-topology idealized secondary-structure scaffold
  library (helix-hairpin, three-helix-bundle, helix-strand-helix, long single
  helix), each segment built with real peptide geometry (NeRF construction
  from standard bond lengths/angles/ideal Ramachandran dihedrals —
  `geometry.py`, unit-tested against the known ideal-alpha-helix rise), then
  EXPLICITLY rigid-body packed — multi-helix topologies place each segment
  antiparallel at a real ~10.5 A inter-helix spacing around a shared bundle
  axis (verified numerically: ~180 degree axis angle, ~9.5-14 A pairwise
  spacing — a real compact fold, not an extended rod; an earlier version of
  this builder that grew a single continuous dihedral chain through the loop
  region left the two helices ~40 A apart with no reversal at all — caught
  by exactly this numeric check and fixed, see PROGRESS_LOG.md) — then the
  whole assembled backbone is rigid-body docked onto the AD templating tip's
  hotspot-residue centroid with sampled approach angles. This is still
  explicitly weaker than a trained generative model — it does not learn
  target-specific shape complementarity the way RFdiffusion does, and the
  resulting ProteinMPNN sequences are more polar/charged than a real
  RFdiffusion backbone would elicit — but it now produces genuinely packed,
  non-degenerate 3D scaffolds, keeping every downstream stage (ProteinMPNN,
  scoring, active learning, selectivity) exercised end-to-end on real,
  structurally sound (if not target-optimized) backbones.
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
