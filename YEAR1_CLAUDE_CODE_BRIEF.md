# PROJECT SENTINEL — YEAR 1 AUTONOMOUS BUILD BRIEF
## For Claude Code: build the entire Year-1 computational project, end to end, with zero human input

> **You are Claude Code.** This document is your complete work order. Read all of it before starting. Then execute the entire Year-1 project autonomously — acquire data, write code, run real computations, generate figures, run tests, write documentation, and finally produce a plain-language reproducibility artifact for the human (who knows nothing about this field). You have full authority to make every scientific, engineering, and design decision. Do not stop, do not ask questions, do not wait. When you hit a fork, pick the best option, document why, and keep going. **Boil the ocean. Do the whole thing. Ship the finished project, not a plan.**

---

# PART 0 — YOUR OPERATING MANDATE (read first)

## 0.1 What "done" means
Year 1 is **done** when the repository contains: a fully working, tested, documented software platform; **real** computed results and figures from real downloaded data; auto-generated Colab notebooks for any GPU-only step; a scientific report; and a plain-language reproducibility artifact. Nothing is left as a stub. Every claimed result is backed by a file you actually produced by running code on real data.

## 0.2 Absolute rules (never violate)
1. **NEVER fabricate, invent, mock, or hand-wave a scientific result.** Every number, plot, and table must come from real code run on real data you downloaded/computed. If a step cannot run in your environment (e.g., no GPU), you (a) build and unit-test the code anyway, (b) run a *miniaturized but real* demonstration that proves the code works, (c) generate a one-click Colab notebook that runs the full-scale version, and (d) clearly label in all outputs that the full-scale run is pending GPU. Placeholder/simulated data is only ever allowed inside unit tests and must be explicitly named `*_synthetic_for_tests`.
2. **NEVER require human input.** No prompts, no "please provide," no waiting. Make the decision, log it, continue.
3. **Determinism & provenance.** Set and record random seeds. Record every data source (URL, PDB ID, access date, checksum). Record tool versions. A reader must be able to reproduce byte-for-byte where possible.
4. **Honesty in outputs.** In the report and artifact, plainly separate "computed locally at full scale," "computed locally at reduced scale (demonstration)," and "prepared, awaiting GPU/Colab." Judges reward this; misrepresentation destroys the project.
5. **Test before you trust.** Ship a passing `pytest` suite, including *scientific validation tests* (e.g., your aggregation predictor must rank the known PHF6/PHF6* segments near the top — if it doesn't, your method is wrong; fix it).
6. **Search when unsure.** You have web access. Verify PDB IDs, tool install commands, API endpoints, and thresholds against current sources before relying on them. This brief's facts were correct at authoring time but verify anything that could have changed.

## 0.3 How to work
- Work in phases (Part 2 modules M0→M13, roughly in order, but parallelize where sensible).
- Commit to git after every module with a clear message. Keep a running `PROGRESS_LOG.md` you append to as you go (what you did, decisions, problems, resolutions).
- Prefer well-maintained open-source tools. If a tool won't install, find an alternative that achieves the same scientific goal, and document the substitution.
- Budget your effort toward **completeness and correctness**, not speed. It is fine for the full build to take many hours of compute.

## 0.4 The one honest constraint you must engineer around: compute tiers
Some steps need a GPU and large model weights. You will not assume a GPU exists. Detect it (Part 4) and route work accordingly. **The good news: the scientific core of Year 1 — the strain atlas, the aggregation engine, real molecular-dynamics runs, and the entire closed-loop design-engine *logic* (sequence design, surrogate model, active-learning loop, selectivity scoring) — runs on CPU and must be fully executed.** Only the generative backbone step (RFdiffusion) and top-accuracy complex folding (AlphaFold2-multimer) truly need a GPU; for those, prepare + demonstrate + Colab-ify.

---

# PART 1 — MISSION CONTEXT (why this project exists)

## 1.1 The disease, in one paragraph
Alzheimer's disease and ~25 related "tauopathies" are caused by the protein **tau** misfolding and stacking into **fibrils** (tangles) that spread neuron-to-neuron like a prion. Cryo-EM revealed that **each tauopathy is a distinct misfolded tau "strain"** — a unique 3-D fold (Alzheimer's, CTE, Pick's, CBD, PSP, AGD, GGT, GPT), each carrying its own post-translational-modification code. Tau's aggregation is nucleated by two tiny segments, **PHF6\* (²⁷⁵VQIINK²⁸⁰)** and **PHF6 (³⁰⁶VQIVYK³¹¹)**. Because each disease has a different fold, a *precision* medicine can, in principle, recognize one strain and spare the others.

## 1.2 The 6-year platform ("Project SENTINEL")
Design, from scratch, **strain-selective, self-clearing tau degraders** — small designed proteins that grab the Alzheimer's tau fold specifically and drag it into the cell's autophagy (garbage-truck) clearance system — then validate them from computer to structure to human neurons to fly models, deliver them across a human blood-brain-barrier model, and generalize the platform to other aggregation diseases. **Honest endpoint: a validated, first-in-class *preclinical platform*, not an approved cure.**

**Approved scope decisions already made (reflect these in all forward-looking text):**
- **In-vivo model = *Drosophila* (fruit flies), not mice.** Flies are invertebrates (no ISEF vertebrate-animal restriction) *and* have a real glial blood-brain barrier, which supports the delivery story. C. elegans is a fallback.
- **Delivery is validated in a human *in-vitro* BBB model** (transwell / microfluidic BBB-on-chip) plus transgenic expression in flies — NOT via AAV into a mammal.
- **Framing is neuroprotection / functional recovery, NOT neuron regeneration** (astrocyte-to-neuron reprogramming is scientifically contested; do not build on it).
- **ISEF categories:** Y1–Y2 → Computational Biology & Bioinformatics (CBIO); Y3 → Biochemistry (BCHM); Y4 → Cellular & Molecular Biology (CELL); Y5 → Biomedical & Health Sciences (BMED); Y6 → Translational Medical Science (TMED).

## 1.3 What Year 1 specifically delivers (your job)
Year 1 = **"The Conformational Atlas + Design Engine."** Three defensible, novel contributions, all computational:
1. **A quantitative Tau Strain Conformational Atlas** — characterize all 8 disease folds, extract each fold's growth-competent "templating tip," and compute a reproducible **strain-fingerprint** that identifies what is unique/exposed on the Alzheimer's fold vs. the others (the selectivity handle).
2. **A closed-loop, self-improving de novo design engine** that outputs candidate mini-protein binders selective for the Alzheimer's fold, with explicit **negative design** against the other folds, validated in silico.
3. **A validated aggregation-propensity engine** (correctly re-identifying PHF6/PHF6* as the nucleating cores) plus a **conformation-sensitive biosensor design concept** (a diagnostic spin-off).

Everything is wrapped as a reproducible, tested, documented open-source repo, plus a scientific report, plus a plain-language reproducibility artifact for the human.

---

# PART 2 — THE YEAR-1 TECHNICAL SPECIFICATION (build these modules)

> File/dir names below are mandatory targets so the reproducibility artifact can reference them. Use Python 3.11+. Root package name: `sentinel`.

## Repository layout to create
```
project-sentinel-year1/
├── README.md
├── PROGRESS_LOG.md
├── environment.yml               # conda env (primary)
├── requirements.txt              # pip fallback
├── pyproject.toml
├── Makefile                      # `make all` runs the whole pipeline
├── config/
│   └── config.yaml               # all parameters, thresholds, seeds, PDB IDs
├── data/
│   ├── raw/                      # downloaded structures/sequences (never edited)
│   ├── interim/                  # cleaned/processed
│   └── external/                 # reference datasets, known binders
├── src/sentinel/
│   ├── __init__.py
│   ├── io/                       # M1 data acquisition
│   ├── atlas/                    # M2 strain atlas
│   ├── aggregation/              # M3 aggregation engine
│   ├── md/                       # M4 molecular dynamics
│   ├── target/                   # M5 design-target definition
│   ├── design/                   # M6 design engine (RFdiffusion/MPNN/fold/loop)
│   ├── validate/                 # M7 in-silico validation
│   ├── biosensor/                # M8 biosensor concept
│   ├── bench/                    # M9 benchmarking
│   ├── figures/                  # M10 plotting
│   └── utils/                    # seeds, logging, provenance, compute-tier
├── notebooks/
│   ├── colab_rfdiffusion.ipynb   # auto-generated, one-click GPU
│   └── colab_af2_multimer.ipynb  # auto-generated, one-click GPU
├── results/
│   ├── atlas/                    # tables, per-fold characterization
│   ├── aggregation/
│   ├── md/
│   ├── design/                   # designed sequences, scores, selectivity matrix
│   ├── validation/
│   └── benchmarks/
├── figures/                      # all publication-quality figures (.png + .svg + .pdf)
├── tests/                        # pytest
├── reports/
│   └── YEAR1_SCIENTIFIC_REPORT.md
└── REPRODUCIBILITY_ARTIFACT.md   # THE plain-language deliverable for the human
```

---

## M0 — Environment, scaffolding, config, utilities
- Create the repo, git init, the directory tree, `pyproject.toml`, `Makefile` (`make setup`, `make data`, `make atlas`, `make aggregation`, `make md`, `make design`, `make bench`, `make figures`, `make test`, `make report`, `make all`).
- Build `environment.yml` pinning: python, numpy, scipy, pandas, matplotlib, seaborn, biopython, biotite, mdtraj, openmm, freesasa, scikit-learn, py3dmol, nglview, requests, pyyaml, tqdm, pytest, jupyter, pdbfixer; conda-forge `pymol-open-source`; `dssp`/`mkdssp`. Provide a pip `requirements.txt` fallback. For deep-learning parts: torch (CPU wheel by default), and clone instructions for ProteinMPNN, RFdiffusion, ESM/ESMFold, localcolabfold.
- `src/sentinel/utils/`:
  - `seeds.py` (global seed = 42, seed numpy/torch/random; record in provenance).
  - `provenance.py` (log every download URL, PDB ID, checksum sha256, timestamp, tool version to `results/PROVENANCE.json`).
  - `compute.py` (detect CPU cores, RAM, CUDA GPU via torch; expose `TIER` ∈ {CPU, GPU_LOCAL}; used to route Part 4 decisions).
  - `logging.py` (structured logging to console + `results/run.log`).
- `config/config.yaml`: put ALL tunables here (seeds, PDB ID list, tau residue definitions, thresholds, design budget, MD length, etc.). Code reads from config, never hardcodes.

## M1 — Data acquisition (`src/sentinel/io/`)
All downloads go to `data/raw/` with checksums logged.

**1a. Tau sequence.** Fetch canonical human tau **2N4R (441 aa)** from UniProt **P10636** via `https://rest.uniprot.org/uniprotkb/P10636.fasta`. Record isoform numbering. Define landmark residues in config (2N4R numbering):
- PHF6\* = 275–280 `VQIINK` ; PHF6 = 306–311 `VQIVYK`
- Microtubule-binding repeats (approx): R1 244–274, R2 275–305, R3 306–336, R4 337–368
- Common fibril core region ≈ 306–378
Also generate the standard aggregation constructs in silico: **K18** (4R MTBR, 244–372) and **K19** (3R, lacks R2).

**1b. The 8-fold tau strain cryo-EM panel.** Download these PDB structures (verify each via the RCSB Data API `https://data.rcsb.org/rest/v1/core/entry/{ID}` and download coordinates from `https://files.rcsb.org/download/{ID}.cif` and `.pdb`). **Seed list (verified correct at authoring; re-verify):**

| Strain | PDB | Core residues |
|---|---|---|
| Alzheimer's PHF | **5O3L** | 306–378 |
| Alzheimer's SF | **5O3T** | 306–378 |
| CTE (Type I) | **6NWP** | 305–379 |
| CTE (Type II) | **6NWQ** | 305–379 |
| Pick's (PiD) | **6GX5** | 254–274 + 306–378 |
| CBD | **6TJO** | 274–380 |
| PSP | **7P65** | 272–381 |
| AGD | **7P6D** | 273–387 |
| GGT | **7P66** | 272–379 |
| GPT | **7P6A** | 272–379 |

Additionally query the RCSB **Search API** (`https://search.rcsb.org/rcsbsearch/v2/query`, POST JSON) for any newer AD/other tau ex-vivo filament structures and for the **hexapeptide steric-zipper microcrystal structures** by sequence search on `VQIVYK` and `VQIINK` (these are short amyloid-segment crystal structures from the Eisenberg lab lineage). Add what you find; log provenance. Cite Fitzpatrick 2017 (AD), Falcon 2018/2019 (PiD/CTE), Zhang/Arakhamia 2020 (CBD), Shi 2021 (PSP/AGD/GGT/GPT).

**1c. Known tau aggregation inhibitors / binders for benchmarking (`data/external/`).** Assemble a small curated table (from literature you search): known tau-aggregation-inhibiting peptides/segments (e.g., Eisenberg-lab structure-based peptide inhibitors targeting VQIINK/VQIVYK; D-peptides; known small molecules like methylene blue/LMTX context). You will use these as positive controls for the benchmark (M9) — your pipeline should recognize the segments they target.

**1d. Clean/prepare structures** (`data/interim/`). For each structure: strip waters/ligands, select the ordered core chains, add hydrogens and fix missing atoms with **PDBFixer**, and save prepared `.pdb`. Keep a single-protofilament and a stacked-layers version (fibrils are stacks of identical chains; you need ≥3 stacked layers to define a realistic "growing tip").

## M2 — The Tau Strain Conformational Atlas (`src/sentinel/atlas/`) — CONTRIBUTION #1
For **every** fold in the panel, compute and tabulate (save to `results/atlas/`):
- **Core definition & topology:** ordered residue span, number of β-strands, cross-β geometry, protofilament count, inter-protofilament interface residues.
- **Per-residue solvent accessibility (SASA)** via `freesasa` on the stacked-layer model → classify each core residue buried vs. exposed on the fibril *surface* vs. exposed at the *growing tip*.
- **Secondary structure** via DSSP.
- **The growth-competent templating tip:** take the terminal layer of the stack; the surface a new monomer docks onto. Characterize its shape, exposed backbone H-bond donors/acceptors (the cross-β templating groups), and side-chain pattern. This is the surface a "capper" must complement — define it precisely per fold.
- **Physicochemical surface maps:** hydrophobicity (Kyte–Doolittle), electrostatic character (charged-residue map; optionally APBS if installable, else a Coulombic proxy), aromatic residues.
- **The steric-zipper interfaces** (PHF6/PHF6* dry interfaces): shape complementarity (Sc) and buried surface area; note VQIINK buries ~2× VQIVYK (verify against your computation and literature).
- **Structural alignment across folds:** pairwise align all folds (e.g., `biotite`/`mdtraj` superposition on shared core residues); compute pairwise RMSD; build a **fold-similarity dendrogram**.
- **THE STRAIN FINGERPRINT (the selectivity handle):** for the AD fold, compute a per-residue "differential exposure/geometry" vector = (what is exposed & geometrically distinctive on the AD templating tip) MINUS (what is shared with CBD/PSP/GGT/AGD/GPT/PiD/CTE). Output a ranked list of **AD-selective hotspot residues + local geometry** — the residues a binder should touch to be AD-specific. Save as a machine-readable `results/atlas/ad_strain_fingerprint.json`.

## M3 — Aggregation-propensity engine (`src/sentinel/aggregation/`) — CONTRIBUTION #3 (validated method)
Goal: an independent, transparent, *validated* predictor that re-derives PHF6/PHF6* as top nucleators (this both validates your method and demonstrates the target choice).
- Implement a **documented consensus β-aggregation score** over a sliding 6-residue window across full-length tau, combining published per-residue scales: β-sheet propensity (Chou–Fasman), hydrophobicity (Kyte–Doolittle), net-charge penalty, aromatic bonus, and a "steric-zipper compatibility" term (based on the known zipper-forming rules / threading against the hexapeptide zipper geometry from M1c). Normalize and combine with documented weights (put weights in config; justify each).
- **Positive-control validation (this is a required test):** PHF6 (306–311) and PHF6\* (275–280) must appear among the highest-scoring windows in tau. If they don't, your predictor is wrong — iterate until it correctly ranks them, then lock it. Also confirm it flags the R2/R3 repeat regions.
- Optionally, if reachable without auth, cross-check a subset against public predictors (TANGO/Waltz/AGGRESCAN/Camsol web or open reimplementations). Do not depend on web servers; treat any as a bonus cross-check.
- Output: `results/aggregation/tau_aggregation_profile.csv` (per-residue and per-window scores), and a call-set of predicted aggregation-nucleating segments.

## M4 — Molecular dynamics (`src/sentinel/md/`) — real dynamics, tiered
Use **OpenMM**. Run **real** simulations at a scale that completes in your environment; provide scale-up configs.
- **System 1 — hexapeptides:** build PHF6 and PHF6\* peptides; solvate (implicit solvent GBSA for CPU speed, or explicit if GPU); energy-minimize; run MD (target ≥ 20 ns if GPU; ≥ 2–5 ns implicit-solvent on CPU is acceptable as a real demonstration — pick the largest that completes, log actual ns). Analyze with **mdtraj**: RMSD, RMSF (per-residue flexibility), radius of gyration, secondary-structure timeline, β-content. Question answered: how floppy/ordered are the nucleating segments in isolation?
- **System 2 — fibril growing tip:** take the AD stacked-layer model; simulate the terminal layer(s) + a docking monomer to characterize templating-surface stability and the exposed cross-β H-bonding groups over time. Compute per-residue RMSF on the tip → identify the rigid "anchor" residues a capper should engage.
- Save trajectories (compressed), analysis CSVs, and plots to `results/md/`. Provide a `md_scaleup.md` describing exact commands to reproduce at full ns on GPU, and generate matching Colab cells if useful.

## M5 — Design-target definition (`src/sentinel/target/`) — CONTRIBUTION #1 output
Fuse M2 (fingerprint) + M4 (rigid anchors) into a single **machine-readable design target spec** `results/target/ad_capper_target.json` containing: the AD-fold PDB + chain/residue selection to design against, the **hotspot residues** (for RFdiffusion conditioning), the templating-tip geometry, the desired binder length range (start 60–120 aa), and the **negative-design panel** (the other folds' equivalent surfaces to avoid). This file is the contract the design engine consumes.

## M6 — The closed-loop de novo design engine (`src/sentinel/design/`) — CONTRIBUTION #2 (the flagship)
This is the flagship deliverable: a **self-improving loop**, not a one-shot run. Architecture:

```
[target spec] → generate backbones → design sequences → fold & score complex
      ↑                                                        │
      └──── Bayesian active-learning proposes next round ←── surrogate ML model
```

**6a. Backbone generation (GPU-tier — prepare + demonstrate + Colab).**
- Integrate **RFdiffusion** (RosettaCommons/RFdiffusion): condition on the M5 hotspots to generate binder backbones against the AD templating tip. Also integrate **BindCraft** as an alternative one-shot pipeline if it installs.
- If GPU present: run a real (modest) campaign — e.g., tens to low-hundreds of backbones. If no GPU: install/set up fully, run the smallest real job that proves the code path (even 1–2 backbones if that's all CPU allows, or the tool's test example), AND auto-generate `notebooks/colab_rfdiffusion.ipynb` that runs the full campaign one-click. Log clearly which happened.

**6b. Sequence design (CPU — fully run).**
- Run **ProteinMPNN** (dauparas/ProteinMPNN, CPU-OK) on every backbone to assign multiple sequences (temperatures in config, e.g., 0.1–0.2; N sequences per backbone). This step you fully execute.

**6c. Fold & score the binder–tau complex (tiered).**
- Primary (GPU): **AlphaFold2-multimer** via localcolabfold, or **Boltz** — refold each designed binder + tau target and compute interface metrics. Filter with literature thresholds (in config): **interface pAE < 7.5, pLDDT > 85, ipTM > 0.7**.
- CPU-runnable substitute you MUST also implement so the loop closes without GPU: an interface scorer combining (i) **ESMFold** single-chain plausibility of the binder (facebookresearch/esm — runs on CPU for short seqs, slowly), (ii) a geometric/energetic complementarity score of the docked binder against the target tip (shape complementarity Sc, buried SASA, clash score, H-bond count, hydrophobic packing — compute with freesasa + your own geometry code), and (iii) predicted binder developability (see 6e). Calibrate this CPU scorer against any GPU-folded examples you do get.
- Output per design: a feature vector + a composite score.

**6d. The active-learning loop (CPU — fully run, this is the novelty).**
- Train a **surrogate model** (scikit-learn `GaussianProcessRegressor` or `RandomForestRegressor`) mapping design features → composite score, updated each round.
- Use **Bayesian optimization** (expected-improvement acquisition) over the design space (e.g., over ProteinMPNN sampling params, hotspot subsets, backbone hyperparameters, length) to propose the next round's design settings. Run **≥ 5 rounds**; show the **learning curve** (best-score-per-round improving). This closed loop is the defensible "self-driving in-silico lab" contribution — implement it end-to-end even where backbone generation is Colab-deferred (drive the loop over the parts you can run, and structure it so plugging in Colab-generated backbones is trivial).

**6e. Selectivity / negative design (CPU — fully run, the precision claim).**
- For every surviving design, score predicted binding against the **AD tip** vs. each **other-fold tip** (from M5 panel) using the same CPU scorer. Build a **selectivity matrix** (`results/design/selectivity_matrix.csv`) and keep only designs predicted AD-selective. Also check against a functional-amyloid / unrelated-zipper control so you can argue "not a generic amyloid binder."
- Developability filter: predict the binder's own aggregation propensity (reuse M3 on the binder sequence — a good binder must not itself aggregate), stability heuristics, and cysteine/glycosylation liabilities.

**6f. Outputs.** `results/design/`: ranked designed sequences (FASTA), per-design feature/score tables, the selectivity matrix, the learning curves, and the top-N "lead" designs carried to M7. Save the trained surrogate and the loop state so Year 2 can resume.

## M7 — In-silico validation of leads (`src/sentinel/validate/`)
- Run short **OpenMM MD** on the top few binder–tau complex models (or binder-alone if only that's foldable on CPU): stability (RMSD), interface persistence (contact maps over time), and a **capping test** — evidence the binder occupies the templating tip such that a new tau monomer can't add (e.g., steric occlusion of the tip H-bonding groups; optionally a steered-MD or simple docking check that monomer addition is blocked).
- Report which leads are stable and mechanistically plausible. Save to `results/validation/`.

## M8 — Conformation-sensitive biosensor concept (`src/sentinel/biosensor/`)
- Design (computationally propose) a **split-reporter biosensor**: take a validated AD-selective binder and propose a split-luciferase or FRET architecture that produces signal only when AD-fold tau is present (e.g., two binder copies bridging the fibril, or binder-induced conformational change). Output a design spec + rationale + a schematic figure. This is a diagnostic spin-off; it need not be experimentally validated (that's later years) but must be a concrete, buildable proposal.

## M9 — Benchmarking & controls (`src/sentinel/bench/`) — makes it defensible
- **Aggregation predictor benchmark:** ROC/PR curve for recovering known aggregation-nucleating segments (PHF6/PHF6* and literature set) vs. non-aggregating tau regions. Report AUC.
- **Design engine sanity/retrospective:** show the loop's scores improve over rounds (learning curve) and that AD-selective designs score better on the AD tip than on other tips (paired stats). Where possible, check whether your pipeline's target site overlaps the sites known Eisenberg-lab inhibitors engage (a form of retrospective validation).
- **Ablations:** show the active-learning loop beats random search (run a random-search baseline for equal budget and compare best-score curves) — this quantifies that your engine actually learns. Report effect size + a simple significance test.
- **Negative controls:** confirm scrambled/known-nonbinder sequences score poorly.
- Save all to `results/benchmarks/` with figures.

## M10 — Figures (`src/sentinel/figures/`) — publication quality, every result gets a figure
Produce, at minimum (save each as `.png` 300dpi + `.svg` + `.pdf` in `figures/`, with captions in a `figures/CAPTIONS.md`):
1. The 8 tau strain folds rendered side by side (PyMOL/py3Dmol), core colored by repeat.
2. Fold-similarity dendrogram + pairwise-RMSD heatmap.
3. Per-residue SASA / buried-vs-exposed maps for the AD tip.
4. **The AD strain-fingerprint** (bar/heatmap of AD-selective hotspot residues).
5. Tau full-length aggregation-propensity profile with PHF6/PHF6* highlighted; benchmark ROC/PR.
6. MD analyses: RMSF plots for hexapeptides and the fibril tip; secondary-structure timelines; Rg.
7. Design-engine **learning curves** (active-learning vs random-search baseline).
8. Score distributions of designs; top-lead complex renders.
9. **Selectivity matrix heatmap** (designs × folds).
10. Biosensor schematic.
11. A one-page "graphical abstract" summarizing the Year-1 story.
Use a clean, consistent style (colorblind-safe palette, readable fonts, labeled axes, units).

## M11 — Tests (`tests/`, pytest)
- Unit tests for every module (I/O parsing, SASA calc, alignment, aggregation scoring, feature extraction, surrogate model, selectivity scoring, figure generation runs without error).
- **Scientific validation tests (required, must pass):**
  - `test_phf6_ranked_top`: aggregation engine ranks PHF6 & PHF6* in the top windows of tau.
  - `test_vqiink_buries_more_than_vqivyk`: zipper analysis reproduces the known relationship.
  - `test_ad_selective_designs_prefer_ad_tip`: surviving designs score higher on AD tip than mean of other tips (paired).
  - `test_active_learning_beats_random`: best-score curve of the loop dominates random search over equal budget.
  - `test_determinism`: fixed seed reproduces key numbers.
- A `make test` target runs the suite; the report states pass/fail counts honestly.

## M12 — Scientific report (`reports/YEAR1_SCIENTIFIC_REPORT.md`)
Write a manuscript-style report (Abstract, Background, Methods, Results, Discussion, Limitations, Future Work, References, Reproducibility). Every claim references a results file/figure. Include an explicit **"What ran locally vs. what awaits GPU/Colab"** section. Include a limitations section that is honest (in-silico only; design success rates; CPU-scorer approximations). Include the real numbers you computed. This is the ISEF/CBIO-facing scientific document.

## M13 — THE REPRODUCIBILITY ARTIFACT (`REPRODUCIBILITY_ARTIFACT.md`) — the human's deliverable
This is the single most important human-facing output. **The human knows nothing about this field.** Write an extremely long, extremely detailed, plain-language document that lets them reproduce the *entire* Year-1 project by reading it, and understand what you did and why. It must include:
1. **Plain-language overview** — what the project is, in everyday words (reuse/adapt the analogies: fibrils = tangles, strains = different origami folds, autophagy = garbage truck, templating tip = the mold's growing edge).
2. **A glossary** — every technical term defined simply, the first time and in one place.
3. **The "why" of every step** — for each module M1–M9, explain in plain words: what question it answered, why it matters, exactly what you did, which tool/command you ran, what the output means, and what the figure shows. Assume zero prior knowledge.
4. **Exact reproduction instructions** — from a blank computer: how to install everything (copy-paste commands), how to run each stage (`make` targets), expected runtime per tier, what files appear where, and how to read each result. Include the Colab one-click instructions for the GPU steps, written for a total beginner.
5. **How to read every figure** — a walkthrough of each figure and what a judge should take from it.
6. **What is real vs. pending** — crystal-clear about what fully ran vs. what needs a GPU/Colab click, so the human never misrepresents the work.
7. **The honest limitations & how to talk about them to judges** — a script for the "what are the weaknesses of your project?" question.
8. **How this feeds Years 2–6** — the short roadmap (Part 6) so they see where it's going.
9. **A "what to say you did" vs "what a mentor/tool did" section** — for ISEF disclosure integrity (the design tools are methods; the human designed/ran the pipeline).
Target length: as long as it needs to be to make a beginner fully self-sufficient. Do not summarize away detail. This document IS the deliverable the human asked for.

---

# PART 3 — ENGINEERING STANDARDS
- **Config-driven:** no magic numbers in code; all in `config/config.yaml`.
- **Reproducible env:** pinned `environment.yml`; a fresh machine reproduces the build via `make setup && make all`.
- **Determinism:** global seed; log seeds; note any nondeterministic GPU ops.
- **Provenance:** `results/PROVENANCE.json` records every external artifact (URL, ID, sha256, date, tool version).
- **Logging:** every stage logs to `results/run.log` and appends to `PROGRESS_LOG.md`.
- **Idempotence & caching:** re-running a stage skips completed work unless `--force`. Cache downloads.
- **Error handling:** if a tool/download fails, retry, then fall back to a documented alternative, then continue — never crash the whole pipeline; record the substitution.
- **Code quality:** type hints, docstrings, `ruff`/`black` formatting, small testable functions.

---

# PART 4 — COMPUTE STRATEGY (how to never get stuck)
1. On startup, `utils/compute.py` detects: CUDA GPU? cores? RAM? Writes `results/compute_profile.json`.
2. **Always fully run (CPU):** M1, M2, M3, M4 (reduced-ns real MD), M5, M6b (ProteinMPNN), M6c CPU-scorer, M6d active-learning loop, M6e selectivity, M6f, M7 (CPU-scale), M8, M9, M10, M11, M12, M13.
3. **GPU-tier steps (M6a RFdiffusion, M6c AF2-multimer):**
   - If `GPU_LOCAL`: download weights, run a real modest-scale campaign, record results.
   - If `CPU` only: (a) install & unit-test the code path; (b) run the smallest real job that executes end-to-end (proves correctness); (c) generate `notebooks/colab_*.ipynb` that run the full campaign one-click on Colab's free GPU, pre-filled with the M5 target spec; (d) label outputs "full-scale run pending GPU (Colab notebook provided)."
4. The active-learning loop must be architected so that Colab-generated backbones drop in without code changes (read backbones from `results/design/backbones/`, wherever they came from).
5. **Never** let a missing GPU stop the pipeline or degrade honesty. The CPU deliverable alone is a complete, defensible Year-1 project; the GPU steps are upside.

---

# PART 5 — DEFINITION OF DONE (final self-check; all must be TRUE)
- [ ] Repo builds from scratch via `make setup && make all` on a CPU-only machine without human input.
- [ ] All 8+ tau folds downloaded, cleaned, characterized; strain fingerprint JSON produced.
- [ ] Aggregation engine validated: PHF6/PHF6* ranked top (test passes); benchmark AUC reported.
- [ ] Real MD trajectories + analyses exist (actual ns logged, not fabricated).
- [ ] Design target spec produced; ProteinMPNN sequences generated; CPU interface scorer working.
- [ ] Active-learning loop ran ≥5 rounds; beats random-search baseline (test passes); learning curves plotted.
- [ ] Selectivity matrix produced; AD-selective leads identified; developability filtered.
- [ ] RFdiffusion + AF2-multimer either ran (GPU) or are installed, mini-demoed, and Colab-notebooked (CPU) — clearly labeled.
- [ ] In-silico validation of leads done; biosensor concept produced.
- [ ] ≥ ~30 unit + scientific tests, all passing; `make test` green; counts stated in report.
- [ ] Every result has a figure; graphical abstract exists.
- [ ] `YEAR1_SCIENTIFIC_REPORT.md` complete with honest "ran vs pending" section.
- [ ] `REPRODUCIBILITY_ARTIFACT.md` complete, beginner-usable, exhaustive.
- [ ] `PROVENANCE.json`, `PROGRESS_LOG.md`, `run.log` complete.
- [ ] No fabricated data anywhere; every number traceable to a real run.

---

# PART 6 — WHAT'S COMING IN YEARS 2–6 (so your Year-1 outputs are forward-compatible)
Persist reusable artifacts (target specs, the trained surrogate, the design engine as an importable package, the atlas tables) so future years plug in. Condensed roadmap:

- **Year 2 (computational + outsourced wet anchor):** upgrade the best binder into a **de-novo autophagy-recruiting degrader** (fuse the binder to an LC3/p62-recruiting module) and a **disaggregase** variant; model the ternary (binder–tau–LC3) complex; build a **kinetic "digital twin"** of tau aggregation and predict how the degrader shifts the curves; run in-silico deep mutational scanning + developability; order 2–3 designs and get one real **Thioflavin-T** aggregation/disaggregation assay done (outsourced/cell-free) as a real anchor. → **Design outputs from Year 1 (sequences, target spec, surrogate) are the direct inputs.**
- **Year 3 (wet lab, BCHM):** express designs; biophysics (SPR/BLI/ITC, CD); a 5-assay aggregation battery (ThT, RT-QuIC, filter-trap, TEM/AFM, DLS); experimental **strain-selectivity**; structural validation (HDX-MS / crosslinking-MS / microED — cryo-EM only as a bonus if a collaborator prioritizes it); yeast/mammalian display + NGS deep mutational scanning to affinity-mature; confirm LC3/p62 engagement.
- **Year 4 (wet lab, CELL):** tau FRET biosensor seeding cells; human iPSC-derived neurons (MAPT-mutant); cerebral organoids; single-cell multi-omics mechanism; safety/selectivity (spares functional tau); live-imaging of autophagic clearance. Framing = neuroprotection/recovery, not regeneration.
- **Year 5 (wet lab, BMED):** ***Drosophila* tauopathy model** (invertebrate; has a glial BBB) — express the degrader transgenically, measure rescue (climbing/locomotion, lifespan, neurodegeneration, tau burden); **human in-vitro BBB model** (transwell / microfluidic BBB-on-chip) to validate a brain-shuttle/delivery module. No vertebrates.
- **Year 6 (wet lab, TMED):** prevention/early-intervention arm in flies; generalize the platform to other strains (CBD/PSP) and other proteins (α-synuclein, TDP-43); combination with a neuroinflammation axis; a **self-improving closed loop** connecting wet data back to the Year-1 design engine; a conditional/logic-gated safety switch; patent + manuscript + honest path-to-clinic.
- **Regeneron STS (senior year):** the cumulative 6-year platform as one 20-page research report (individual work only; disclose AI-tool use).

---

# PART 7 — KEY FACTS, IDS, THRESHOLDS, AND SOURCES (verify before relying)
- **Tau:** UniProt **P10636**; 2N4R = 441 aa. PHF6\* = 275–280 VQIINK; PHF6 = 306–311 VQIVYK. Common core ≈ 306–378.
- **Strain PDB panel:** 5O3L, 5O3T (AD); 6NWP, 6NWQ (CTE); 6GX5 (PiD); 6TJO (CBD); 7P65 (PSP); 7P6D (AGD); 7P66 (GGT); 7P6A (GPT).
- **APIs:** UniProt `rest.uniprot.org/uniprotkb/P10636.fasta`; RCSB data `data.rcsb.org/rest/v1/core/entry/{id}`; RCSB files `files.rcsb.org/download/{id}.cif|.pdb`; RCSB search `search.rcsb.org/rcsbsearch/v2/query`.
- **Design thresholds (AF2-multimer filter):** interface pAE < 7.5, pLDDT > 85, ipTM > 0.7. Binder length start 60–120 aa.
- **Tools/repos:** RosettaCommons/RFdiffusion; dauparas/ProteinMPNN; sokrypton/ColabFold (or YoshitakaMo/localcolabfold); facebookresearch/esm (ESMFold); optionally jwohlwend/boltz; OpenMM; PDBFixer; freesasa; DSSP; PyMOL-open-source; scikit-learn.
- **Key papers to cite (search for exact refs):** Fitzpatrick 2017 Nature (AD tau cryo-EM); Falcon 2018/2019 (PiD/CTE); Zhang/Arakhamia 2020 (CBD/PTMs); Shi 2021 Nature (structure-based classification, PSP/AGD/GGT/GPT); von Bergen 2000 PNAS (PHF6 motif); Seidler/Eisenberg 2018 Nat Chem (VQIINK dominance, structure-based inhibitors); Watson 2023 Nature (RFdiffusion); Dauparas 2022 Science (ProteinMPNN); Pacesa 2025 Nature (BindCraft); Wu 2025 Science (IDR-binder design); Mirdita 2022 Nat Methods (ColabFold).

---

## FINAL WORD TO CLAUDE CODE
Build all of it. Real data, real code, real results, real tests, real figures, honest labels, exhaustive documentation. Where a GPU is missing, prepare and demonstrate and Colab-ify — never fake, never skip, never stall. Your last two acts are (1) the scientific report and (2) the beginner-proof reproducibility artifact. Then run the full test suite one final time and write the pass/fail counts into both documents. **Boil the ocean. Ship it complete. Go.**
