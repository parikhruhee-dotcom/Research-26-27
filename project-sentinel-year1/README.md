# Project SENTINEL — Year 1

**The Conformational Atlas + Design Engine.**

An open-source, tested, reproducible computational pipeline that (1) builds a
quantitative structural atlas of the eight known cryo-EM tau fibril "strains"
(Alzheimer's, CTE×2, Pick's, CBD, PSP, AGD, GGT, GPT), (2) derives an
Alzheimer's-selective "strain fingerprint," (3) runs a validated aggregation-
propensity engine that re-identifies the known PHF6/PHF6* nucleating motifs,
and (4) drives a closed-loop, self-improving de novo design engine that
proposes strain-selective mini-protein binders against the Alzheimer's fold,
with explicit negative design against the other seven folds.

This is Year 1 of the 6-year **Project SENTINEL** platform (see
[reports/YEAR1_SCIENTIFIC_REPORT.md](reports/YEAR1_SCIENTIFIC_REPORT.md) §Future Work
for the roadmap). Year 1 is entirely computational.

## Start here

- **New to the field / non-specialist?** Read
  [REPRODUCIBILITY_ARTIFACT.md](REPRODUCIBILITY_ARTIFACT.md) — it explains
  everything from scratch and gives copy-paste reproduction instructions.
- **Scientist / reviewer?** Read
  [reports/YEAR1_SCIENTIFIC_REPORT.md](reports/YEAR1_SCIENTIFIC_REPORT.md).
- **Just want to run it?**

```bash
conda env create -f environment.yml && conda activate sentinel
make setup
make all        # data -> atlas -> aggregation -> md -> target -> design -> validate -> biosensor -> bench -> figures -> test -> report
```

or with pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make setup
make all
```

`make all` is idempotent: it detects your compute tier (CPU vs GPU) in
`results/compute_profile.json` and routes GPU-only steps (RFdiffusion
backbone generation, AlphaFold2-multimer complex folding) to auto-generated
one-click Colab notebooks in `notebooks/`, while running everything else for
real on CPU. See `results/design/GPU_TIER_STATUS.md` after running `make
design` for exactly what ran locally vs. what is Colab-deferred.

## Repository map

See [PROGRESS_LOG.md](PROGRESS_LOG.md) for a running build log and
`results/PROVENANCE.json` for every data source, checksum, and tool
substitution used to produce this repository's outputs.
