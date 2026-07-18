# PROGRESS LOG — Project SENTINEL Year 1

Append-only build log. Entries below the marker are auto-appended by
`sentinel.utils.logging.append_progress_log`; the header entries here were
written manually at each module boundary during the autonomous build.

---

## M0 — Environment, scaffolding, config, utilities
- Repository scaffolded under `project-sentinel-year1/` inside the existing
  git repo (no nested `git init` — the enclosing repo is already version
  controlled).
- Environment: CPU-only sandbox, 2 cores, 7.8 GB RAM, ~20 GB free disk,
  outbound internet to `rest.uniprot.org` and `files.rcsb.org` confirmed
  reachable (HTTP 200). No CUDA GPU (`torch.cuda.is_available()` will be
  False; `nvidia-smi` absent). This is exactly the "CPU tier" the brief
  anticipates in Part 4 — GPU-only steps (RFdiffusion backbone generation,
  AlphaFold2-multimer) will be prepared, unit-tested where possible, and
  Colab-ified rather than run locally.
- Installed via pip: numpy, scipy, pandas, matplotlib, seaborn, biopython,
  biotite, mdtraj, openmm (8.5.x CPU build), freesasa, scikit-learn,
  requests, pyyaml, tqdm, pytest, pdbfixer — all imported successfully.
- Installed `mkdssp` 4.6.1 via `conda install -c conda-forge dssp` (the
  `dssp` package is not in the default apt sources on this image; conda-forge
  worked). This is the real DSSP binary specified in the brief — no
  substitution needed.
- Wrote `config/config.yaml` as the single source of truth for every
  threshold, seed, PDB ID, weight, and budget used downstream.
- Wrote `src/sentinel/utils/{config,seeds,logging,provenance,compute}.py`:
  global seed = 42; structured logging to `results/run.log` +
  `PROGRESS_LOG.md`; `results/PROVENANCE.json` ledger (downloads, tool
  versions, documented substitutions, seeds); compute-tier detector writing
  `results/compute_profile.json`.
- **2026-07-18T12:15:53Z** `[M1a]` Fetched tau P10636 (758 aa) from UniProt REST API; PHF6/PHF6* and repeat-region landmarks FAILED VERIFICATION against the downloaded sequence.
- **2026-07-18T12:16:41Z** `[M1a]` Fetched tau P10636 (441 aa) from UniProt REST API; PHF6/PHF6* and repeat-region landmarks verified against the downloaded sequence.
- **2026-07-18T12:17:34Z** `[M1b]` Fetched 10/10 strain-panel structures (AD_PHF, AD_SF, CTE_I, CTE_II, PiD, CBD, PSP, AGD, GGT, GPT) from RCSB, each verified via the Data API before coordinate download. Bonus RCSB search found 1 VQIVYK-sequence hits and 1 VQIINK-sequence hits.
- **2026-07-18T12:18:54Z** `[M1c]` Curated 4-entry known-tau-inhibitors table from a literature search (Seidler 2018 Nature Chemistry confirmed via WebSearch: VQIINK is the dominant nucleator and VQIINK-targeted structure-based inhibitors block full-length-tau seeding more effectively than VQIVYK-targeted ones). Exact designed-peptide sequences from the primary reference's supplement were deliberately NOT transcribed (avoids mis-transcription risk); the table records target segment + mechanism, which is what M9's segment-recovery benchmark checks. Broad-spectrum compounds (methylene blue, EGCG) included as negative controls for segment-level specificity.
- **2026-07-18T12:21:18Z** `[M1d]` Prepared single-protofilament and >= 3-layer stacked models for 10/10 strain-panel structures (PDBFixer: missing atoms/residues filled, hydrogens added at pH 7.0, heterogens/waters stripped). 0 structures had fewer than 3 layers available in their largest deposited protofilament and used all available layers instead (documented per entry in data/interim/structures/prepared_manifest.json, field 'layers_below_target').
- **2026-07-18T12:21:53Z** `[M1a]` Fetched tau P10636 (441 aa) from UniProt REST API; PHF6/PHF6* and repeat-region landmarks verified against the downloaded sequence.
