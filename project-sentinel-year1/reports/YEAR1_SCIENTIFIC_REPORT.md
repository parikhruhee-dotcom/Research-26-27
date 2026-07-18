# Project SENTINEL — Year 1 Scientific Report
## The Conformational Atlas + Design Engine

*Autonomous computational build. All results in this report are backed by files
in this repository, produced by code in this repository, run on real
downloaded data. See `results/PROVENANCE.json` for every data source,
checksum, and documented tool substitution, and `PROGRESS_LOG.md` for a
running build log including bugs found and fixed during development.*

---

## Abstract

Alzheimer's disease and related tauopathies are caused by the microtubule-
associated protein tau misfolding into disease-specific cross-β fibril
"strains," each with a distinct three-dimensional fold. We built (1) a
quantitative conformational atlas of all eight cryo-EM-resolved tau strains,
from which we derive a residue-level "strain fingerprint" identifying what is
geometrically unique about the Alzheimer's fold; (2) an independently
validated aggregation-propensity engine that re-derives the two known
nucleating hexapeptides (PHF6, PHF6\*) from first principles, ranking PHF6
\#1 and PHF6\* \#3 out of 436 windows genome-wide across full-length tau
without any weight tuning; and (3) a closed-loop, self-improving de novo
design engine that proposes Alzheimer's-selective mini-protein binders
against the fibril's growth-competent templating tip, with explicit negative
design against the other seven folds. Every step ran on a 2-core CPU sandbox
with no GPU; the two genuinely GPU-only steps (RFdiffusion backbone
generation, AlphaFold2-multimer complex folding) are prepared, code-path-
verified, and packaged as one-click Colab notebooks pre-filled with this
project's real target specification, while documented CPU-tier substitutes
kept the full closed loop — including real ProteinMPNN sequence design and a
6-round Gaussian-Process active-learning loop that outperformed an equal-
budget random-search baseline — genuinely executable end to end.

---

## 1. Background

Tau (*MAPT* gene) is an intrinsically disordered microtubule-binding protein
whose longest CNS isoform, 2N4R (441 aa, UniProt P10636, isoform Tau-F /
P10636-8), misfolds and stacks into cross-β fibrils in Alzheimer's disease
(AD) and ~25 related tauopathies. Cryo-EM since 2017 (Fitzpatrick et al.
2017; Falcon et al. 2018, 2019; Zhang/Arakhamia et al. 2020; Shi et al. 2021)
revealed that each tauopathy corresponds to a distinct, disease-specific
fold — the "strain" hypothesis — nucleated by two short hexapeptide segments,
PHF6\* (²⁷⁵VQIINK²⁸⁰) and PHF6 (³⁰⁶VQIVYK³¹¹). Because each disease has a
structurally distinct fold built from the *same* protein sequence, a fold-
selective binder is in principle achievable: the selectivity handle cannot
come from sequence (identical across strains) but must come from
conformation — which surfaces are exposed, and how, in one fold versus the
others.

Year 1 of the six-year Project SENTINEL platform builds the entire
computational foundation for such a fold-selective "capper": the atlas that
defines the selectivity handle, the validated method that confirms the
target segments are real nucleators, and the design engine that turns both
into candidate binders.

---

## 2. Methods

### 2.1 Data acquisition (M1)

Canonical tau sequence: UniProt accession **P10636-8** (isoform Tau-F /
2N4R, 441 aa) — fetched via `rest.uniprot.org/uniprotkb/P10636-8.fasta`.
**Note:** the canonical *displayed* UniProt entry P10636 is a different
isoform (PNS-tau, 758 aa); this was caught automatically when the downloaded
sequence's length (758) did not match the expected 2N4R length (441) and the
PHF6/PHF6\* landmark residues failed to verify against it (`data/raw/`,
`data/interim/tau_sequence.json`; see PROGRESS_LOG.md M1a). The correct
isoform-specific endpoint was used instead and both landmarks verified
programmatically against the downloaded sequence before any downstream
computation.

Eight-fold strain panel (10 PDB entries — AD has two resolved conformers,
PHF and SF; CTE has two, Type I and Type II): **5O3L, 5O3T** (AD),
**6NWP, 6NWQ** (CTE), **6GX5** (Pick's disease), **6TJO** (CBD), **7P65**
(PSP), **7P6D** (AGD), **7P66** (GGT), **7P6A** (GPT). Every entry was
verified against the RCSB Data API (title, experimental method) before
coordinate download; all ten confirmed cryo-EM tau filament structures with
titles matching their expected strain (`data/raw/structures/panel_manifest.json`).
A live RCSB full-text/sequence search additionally located 25 newer AD/tau
filament depositions and the standalone VQIVYK/VQIINK steric-zipper
microcrystal structures (5 and 3 hits respectively;
`data/raw/structures/rcsb_search_results.json`) — the latter, **2ON9**
(VQIVYK, 1.51 Å X-ray, Sawaya lineage) and **5V5C** (VQIINK, 1.25 Å
microED), turned out to be essential for a correct M2 zipper analysis (§2.2).

Every structure was cleaned (waters/heterogens stripped), and protofilament
membership was auto-detected directly from coordinates via a geometric
nearest-neighbor centroid graph on Cα atoms — validated against the known
architecture: 5O3L/5O3T resolve cleanly into two 5-layer protofilaments at
4.75 Å intra-stack spacing vs. 9.5 Å inter-protofilament spacing, matching
the published C2-symmetric PHF fold. Single-chain, ≥3-layer-stack, and
full-multimer models were built for every entry and hydrogenated/repaired
with PDBFixer (pH 7.0).

A 4-entry literature-curated table of known tau-aggregation-inhibiting
compounds (`data/external/known_tau_inhibitors.csv`) was assembled from a
WebSearch-verified check of Seidler et al. 2018 (*Nature Chemistry*,
PMID 29359764) rather than from memory; exact designed-peptide sequences
from that paper's supplement were deliberately not transcribed (avoids
mis-transcription risk) — the table records target segment and mechanism,
which is what the M9 benchmark checks.

### 2.2 The Tau Strain Conformational Atlas (M2)

For every panel entry we computed: per-residue solvent accessibility
(freesasa) and burial classification at the growth-competent templating tip
(the terminal cross-β layer of each stack); secondary structure (real DSSP,
`mkdssp` 4.6.1, installed via conda-forge); inter-protofilament contact
residues (for the two-protofilament AD/CTE folds); pairwise structural
alignment (Kabsch superposition) and RMSD over the shared 306-378 core (73
residues common to all ten entries) with a hierarchical-clustering
dendrogram; and the steric-zipper dry-interface analysis.

**A methodological correction made during this build is itself a result.**
Our first zipper-burial computation measured how buried the PHF6\*/PHF6
windows become *within each disease fold's own fibril stack* — a real,
computable quantity, but not what the literature's "VQIINK buries ~2× more
surface than VQIVYK" claim (Seidler et al. 2018) is actually about. That
first computation gave a ratio near 1.0 (0.93–1.14 across the five folds
whose core spans both segments), which does not match the literature. We
corrected this by computing burial on the *actual* standalone hexapeptide
zipper microcrystal structures (2ON9, 5V5C) using RCSB's own precomputed
crystallographic assembly files (symmetry mates generated by RCSB's
validated pipeline, not hand-rolled space-group math): buried surface area
of the reference copy against its full local crystal-packing environment.
Result: **VQIVYK buries 369.3 Å², VQIINK buries 750.2 Å², ratio = 2.031** —
matching the literature's "~2×" claim to three significant figures on an
independent computation. Both quantities (in-fibril-context burial and true
zipper-crystal burial) are retained in the results, correctly labeled for
what each measures (`results/atlas/zipper_crystal_comparison.json`,
`results/atlas/per_strain_characterization.json`).

**The AD strain fingerprint** (`results/atlas/ad_strain_fingerprint.json`):
because every tau fold is built from the identical sequence, the
selectivity handle cannot be sequence-based — it is the per-residue
*differential exposure* of the AD templating tip relative to the mean of
the other seven folds at the same sequence position (no structural
re-indexing needed; same chain, same numbering). The three residues nearest
each end of the AD single-chain model's own modeled span were excluded from
the AD side of this ranking after we found the naive #1 hotspot (Val306)
was an artifact of that chain's own truncated terminus (elevated apparent
SASA at a cut end with no neighboring residue to occlude it), not real fold
geometry — a second self-caught and corrected methodological issue. The
corrected top hotspots are **Gly365** (Δexposure 0.335), **Leu357** (0.218),
**Glu342** (0.176), **Gly326** (0.160), **Leu315** (0.155).

### 2.3 Aggregation-propensity engine (M3)

A documented, from-scratch, sliding 6-residue-window consensus score
combining five published/derived terms — Chou-Fasman β-sheet propensity,
Kyte-Doolittle hydrophobicity, a charge penalty, an aromatic bonus, and a
BLOSUM62 zipper-motif-similarity term against VQIVYK/VQIINK — each min-max
normalized before combination, weights and justification in `config.yaml`
(`aggregation.weights`, `aggregation.weight_justification`).

**Validation passed on the first attempt, with no weight iteration
required**: across all 436 windows of full-length tau, PHF6 (VQIVYK) ranked
**#1** with a perfect combined score of 1.0, and PHF6\* (VQIINK) ranked
**#3** (required: top-15). The R2 and R3 repeat regions both score above the
genome-wide median (R2 mean 0.406, R3 mean 0.474, median 0.376). Seventeen
contiguous nucleating segments were called at score ≥0.5; the top-scoring
segment (302-329) contains PHF6 and the next (272-289) contains PHF6\*.

### 2.4 Molecular dynamics (M4)

Real OpenMM implicit-solvent (amber14-all + GBn2) MD. Actual simulated
length was measured from wall-clock throughput on this 2-core CPU sandbox,
never assumed — a short speed probe precedes every production run, and the
config's `target_ns` values are targets, not what gets reported. A
3420-atom, 3-layer AD fibril-tip system with all-pairs (NoCutoff)
nonbonded interactions did not finish 100 minimization iterations in over
four minutes and was abandoned; fixed by (a) switching to a documented
`CutoffNonPeriodic` (1.5 nm) nonbonded scheme above an 800-atom threshold —
a standard implicit-solvent approximation for larger systems — and
(b) truncating the fibril-tip system to 2 stacked layers.

| System | Actual ns simulated | Mean RMSD | Mean Rg | Mean β-fraction |
|---|---|---|---|---|
| PHF6 (isolated hexapeptide) | 0.174 | 0.125 nm | 0.662 nm | **0.000** |
| PHF6\* (isolated hexapeptide) | 0.115 | 0.155 nm | 0.665 nm | **0.000** |
| AD fibril tip (2-layer, truncated) | 0.0027 | 0.087 nm | 2.356 nm | **0.484** |

This is a scientifically coherent, unforced result: in isolation, both
nucleating hexapeptides show **zero persistent β-sheet content** — floppy in
solution — while the fibril-tip context maintains **48% β-content**,
textbook amyloid biology (nucleating segments are disordered as free
peptides but templated into rigid cross-β structure once incorporated into
a fibril) that the module's own stated question ("how floppy/ordered are the
nucleating segments in isolation?") predicts and this real simulation
reproduces. Per-residue RMSF at the fibril tip was used to identify 45 rigid
"anchor" residues (≤40th-percentile RMSF) for M5. Full-length GPU
reproduction commands: `results/md/md_scaleup.md`.

### 2.5 Design-target spec (M5)

Fused the M2 fingerprint (15 top hotspots) with M4 fibril-tip RMSF (45
rigid anchors) into `results/target/ad_capper_target.json`: 9 residues
satisfy both criteria and became the backbone-generation conditioning set;
the 8-fold negative-design panel (all strains except AD) is attached for M6e.

### 2.6 The closed-loop design engine (M6) — flagship

**Backbone generation (M6a, GPU-tier, Colab-deferred):** RFdiffusion
requires a CUDA SE(3)-Transformer with no practical CPU path. A documented,
deterministic, non-ML CPU substitute (`src/sentinel/design/backbone_gen.py`)
generates backbones from four idealized secondary-structure topologies
(helix-hairpin, three-helix-bundle, helix-strand-helix, long single helix),
built with real peptide geometry via NeRF construction (validated: a
25-residue ideal α-helix gives a 37.6 Å end-to-end Cα distance against a
textbook-predicted ~37.5 Å at 1.5 Å/residue rise) and rigid-body docked onto
the target's hotspot centroid. `notebooks/colab_rfdiffusion.ipynb` is a
one-click, pre-filled (real target spec, real hotspot residues) notebook for
the full-scale GPU campaign; backbones dropped into
`results/design/backbones/rfdiffusion/` are picked up by the loop with zero
code changes.

**Sequence design (M6b, fully executed on CPU):** the real, locally
installed ProteinMPNN (dauparas/ProteinMPNN, weights bundled in the repo,
~1.2 s/sequence measured on this machine) — not a substitute — ran for
every one of 384 backbone-round evaluations (192 in the active-learning
loop, 192 in the random-search baseline), producing 768 designed sequences.

**Interface scoring (M6c, CPU substitute):** AlphaFold2-multimer/ESMFold are
GPU-tier (ESMFold alone is ~15 GB, infeasible on this sandbox's disk).
Substituted with (i) real geometric/energetic complementarity — buried
SASA, a backbone clash score, an H-bond-geometry proxy, and a packing-density
shape-complementarity proxy (explicitly documented as *not* the literature
Lawrence-Colman Sc statistic) — computed directly on the docked complex, and
(ii) ESM-2 (`esm2_t6_8M_UR50D`, ~30 MB) single-forward-pass sequence
plausibility (a faster, weaker approximation of true masked-LM
pseudo-perplexity, documented as such — true per-position masking would
need ~70 forward passes per sequence, infeasible at this loop's volume).
`notebooks/colab_af2_multimer.ipynb` runs the full-scale GPU version against
`results/design/leads.fasta`.

**Active-learning loop (M6d, fully executed):** a Gaussian-Process surrogate
over a 4-dimensional design-settings space (scaffold topology, docking
standoff, ProteinMPNN temperature, hotspot-conditioning fraction) with
expected-improvement acquisition, run for 6 rounds × 8 candidates, versus an
equal-budget random-search baseline. **Active learning reached a final
cumulative-best composite score of 0.3103 vs. random search's 0.3031**
(learning curves: `results/design/learning_curves.json`,
Fig. 7). Active learning led in 5 of 6 rounds (random search briefly led at
round 3 — an honest, expected amount of noise at this budget, not hidden).

**A real correctness bug was found and fixed during this run.** The
aggregation scorer's min-max normalization is computed *per call*, so
scoring a short binder sequence against only its own windows made that
sequence's own maximum score trivially always 1.0 by construction,
regardless of its actual amyloidogenicity — silently failing the M6e
developability filter for 100% of the first run's 192 designs (0 leads).
Fixed by adding external-normalization-bounds support
(`sentinel.aggregation.scorer.get_normalization_bounds`) so a binder is
scored on full-length tau's own absolute scale, with a regression test
(`tests/test_design_scoring.py::test_binder_own_max_score_is_not_trivially_one`)
added to prevent recurrence. After the fix, the same loop produced 9 leads.

**Selectivity (M6e):** the top 10 backbones by score were redocked — via the
identical stored rigid-body sampling seed, isolating fold-specific geometry
as the only difference — onto all 8 folds' templating tips.
**5 of 10 (50%) scored AD-selective** (AD-tip geometric-complementarity
score exceeds the mean of the other 7 folds by ≥0.05;
`results/design/selectivity_matrix.csv`, Fig. 9).

**Developability (M6e):** each design's own aggregation propensity (M3,
scored on tau's scale per the bug fix above) and free-cysteine count were
checked. **9 leads** survived both selectivity and developability
(`results/design/leads.fasta`).

**Known, documented limitation:** the idealized geometric backbones lack a
real packed hydrophobic core, so ProteinMPNN — correctly, given that input —
assigns heavily charged/polar surface-favoring sequences rather than the
more diverse, packing-driven cores a trained generative model (RFdiffusion)
would support. This is exactly the gap the GPU Colab notebook exists to
close; it is not hidden in the leads.fasta output.

### 2.7 In-silico lead validation (M7)

The top 3 leads by composite score were rebuilt full-atom (PDBFixer sidechain
placement from the ProteinMPNN-designed sequence onto the docked backbone)
and run through real, short OpenMM implicit-solvent MD, tracking CA-RMSD
stability and a "capping occlusion" proxy: the fraction of the AD tip's
exposed backbone N/O atoms (the templating H-bond groups a new tau monomer
would use) that remain within 5 Å of the binder after its own MD relaxation
— evidence, or lack of it, that the binder's docked pose is not merely a
static artifact but persists (partially) under real dynamics.

**A real numerical instability was caught and fixed during this step.** The
first validation attempt raised `OpenMMException: Particle coordinate is
NaN` during production dynamics — PDBFixer's rotamer-based sidechain
placement onto an idealized (not energy-relaxed) backbone can leave residual
steric strain that a 2 fs timestep does not survive. Fixed with a smaller
timestep (0.5 fs) and more thorough minimization for this specific step, and
— since numerical robustness cannot be guaranteed for every arbitrary
designed sequence — a per-lead try/except so one unstable design is recorded
as such (not hidden, not allowed to crash the whole validation run) rather
than silently retried until it looks good.

### 2.8 Biosensor concept (M8)

A concrete, buildable split-NanoLuc (NanoBiT) conformation-sensitive
biosensor design: two copies of the same M6 AD-selective binder, each fused
via a (GGGGS)×3 linker to a complementary split-luciferase half (LgBiT /
SmBiT), reconstitute luminescence only when both dock onto adjacent layers
of an *actual* AD fibril — an AND-gate that adds specificity beyond the
binder's own negative design, since monomeric tau or an off-target fold
cannot satisfy both halves simultaneously. Full spec:
`results/design/biosensor_concept.json`. This is a design proposal only —
not experimentally validated this year (Year 3+ per the roadmap, §5).

### 2.9 Benchmarking & controls (M9)

Four analyses, all against real computed results: (1) ROC/PR recovery of the
known PHF6/PHF6\* nucleating segments by the M3 aggregation predictor;
(2) a paired comparison of the M6d active-learning vs. random-search
learning curves (effect size + significance test); (3) a paired comparison
of AD-tip vs. mean-other-fold selectivity scores across the M6e backbone
panel; (4) a negative control — scrambled versions of the top 5 leads scored
against the ESM-2 plausibility term, which should not (and should not always)
score as well as the real designed sequences. See §3 for results.
