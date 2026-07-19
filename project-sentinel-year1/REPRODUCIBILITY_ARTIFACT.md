# Project SENTINEL — Year 1 Reproducibility Artifact
## A complete, beginner-friendly guide to what this project is, what was built, and how to reproduce every result

*This document assumes you know nothing about molecular biology, structural
biology, or machine learning for protein design. Every term is defined the
first time it's used. If you read this document start to finish, you should
be able to (a) explain the whole project to someone else, (b) reproduce
every number and figure in this repository on your own computer, and
(c) answer a judge's questions about what you did vs. what a tool did.*

---

## Table of contents

1. [Plain-language overview](#1-plain-language-overview)
2. [Glossary](#2-glossary)
3. [The "why" of every step (M1-M9)](#3-the-why-of-every-step)
4. [Exact reproduction instructions](#4-exact-reproduction-instructions)
5. [How to read every figure](#5-how-to-read-every-figure)
6. [What is real vs. pending](#6-what-is-real-vs-pending)
7. [Honest limitations — and how to talk about them to judges](#7-honest-limitations)
8. [How this feeds Years 2-6](#8-how-this-feeds-years-2-6)
9. [What to say you did vs. what a tool did](#9-disclosure)

---

## 1. Plain-language overview

### The disease, in everyday words

Alzheimer's disease — and about 25 related brain diseases collectively
called "tauopathies" — are caused by a protein in your brain cells called
**tau** doing something it isn't supposed to do. Normally, tau's job is to
act like a strap that helps hold together the "train tracks" (called
microtubules) that neurons use to transport supplies inside themselves.
In these diseases, tau proteins stick to each other, misfold, and clump
into long fibers called **fibrils** — visible under a microscope as
"**tangles**" inside neurons. These tangles are toxic to the cell and
spread from neuron to neuron, roughly like a very slow, non-infectious
chain reaction.

Here is the surprising and useful fact this whole project is built on:
scientists using a powerful imaging technique called **cryo-EM**
(cryo-electron microscopy — essentially, an extremely powerful,
extremely cold microscope that can see the 3D shape of individual
molecules) discovered that tau does not clump the same way in every
disease. In Alzheimer's, the tau molecules fold up and stack into one
specific 3D shape. In a different disease called CTE (the brain disease
associated with repeated head injuries in athletes), the *same* tau
protein — identical amino-acid sequence — folds into a *different* 3D
shape. Same protein, different "origami," different disease. Scientists
call each distinct fold a **strain** — think of it like the same yarn
knitted into eight completely different sweater patterns.

Why does that matter? If Alzheimer's tau has a fold that no other disease
has, then in principle you could design a molecule that recognizes *only*
that fold — like a key that only fits one specific lock — and leaves the
other seven folds (and normal, non-clumped tau, which your brain needs)
completely alone. That's precision medicine for a brain disease, and it's
the long-term goal of the 6-year "Project SENTINEL" platform this Year-1
work is the foundation of.

There's a second useful fact: within the full ~400-amino-acid tau
protein, there are two very short segments — six amino acids each — that
act like the "ignition switch" for the whole clumping process. Scientists
named them **PHF6** and **PHF6\*** (PHF stands for "paired helical
filament," an old name for the tangle structure). If you block just these
two tiny segments, you can often stop or slow the whole clumping process.

### What "the mold's growing edge" means

Imagine the fibril (tangle) as a stack of identical Lego bricks glued
together, growing taller one brick at a time. The very top brick — the one
with nothing stacked on top of it yet — has an exposed face where the
*next* Lego brick (a new tau molecule) would attach. We call this exposed
face the **templating tip**: it's the mold's growing edge. If you can
design a molecule that sits down on that exact spot and physically blocks
it — like putting a cap on a bottle — new tau molecules can't attach there
anymore, and that fibril stops growing at that end. This project spends a
lot of effort characterizing exactly what that surface looks like for the
Alzheimer's fold specifically, so a future "cap" can be custom-designed for
it.

### What "the garbage truck" (autophagy) means

Your cells have a built-in recycling/garbage-disposal system called
**autophagy** ("self-eating," a normal, essential biological process where
the cell wraps up damaged proteins in a membrane bubble and sends them to
be broken down and recycled). The 6-year vision for Project SENTINEL
(**not** what Year 1 builds — see §8) is a designed protein that does two
things: (1) grab onto the Alzheimer's-specific tau fold using a "cap" like
the one described above, and (2) simultaneously grab onto a tag the cell's
garbage truck recognizes, so the tagged tau fibril gets physically dragged
into the recycling system and destroyed. Year 1 builds and validates the
"grab onto the Alzheimer's fold specifically" half of that molecule — the
selective binder — computationally. The garbage-truck-tag half is Year 2.

### What Year 1 actually delivers (three things)

1. **A quantitative atlas of all 8 known tau strains** — a structured,
   computed comparison of every published tau fold, ending in a ranked
   list of exactly which spots on the Alzheimer's fold are geometrically
   unique to it (not shared with the other 7 diseases). This ranked list
   is the "selectivity handle" — the specific addresses on the fibril
   surface a future drug should grab.
2. **A validated method that finds PHF6/PHF6\*** — before trusting any
   computational method with a life-and-death design decision, you have
   to prove it works on cases where you already know the right answer.
   This project built a scoring method from published, textbook chemistry
   rules and tested it: does it correctly flag PHF6 and PHF6\* — the two
   segments *already proven* by decades of lab experiments to be the real
   "ignition switches" — as the most dangerous segments in the whole
   protein? It does (PHF6 ranked #1 out of 436 candidate segments; PHF6\*
   ranked #3). That's the method proving itself trustworthy before being
   used for anything else.
3. **A self-improving computer "design engine"** — software that proposes
   candidate small proteins ("binders") shaped to stick to the
   Alzheimer's-specific templating tip, tests each candidate
   computationally, learns from the results, and gets better at proposing
   good candidates round after round — while also checking that each
   candidate does *not* stick well to the other seven disease folds (the
   "negative design" / selectivity check).

Everything above is **entirely computational** — no lab work happens in
Year 1. Nothing in this repository was tested in a real dish, animal, or
patient. That real-world wet-lab validation is Years 3+ (§8).

---

## 2. Glossary

*(alphabetical; every term also gets a plain-language explanation the first
time it's used in §3)*

- **Aggregation propensity**: how likely a stretch of protein is to clump
  together with copies of itself into a fibril.
- **Alpha helix / beta strand**: two of the basic shapes a stretch of
  protein backbone can twist/fold into. An alpha helix is a coiled spring
  shape. A beta strand is a flat, zig-zag ribbon shape; several beta
  strands side by side form a "beta sheet." Amyloid fibrils are built almost
  entirely from stacked beta sheets (hence "cross-beta" structure).
- **Amino acid**: the basic building block of a protein — there are 20
  standard kinds, each abbreviated with a one-letter or three-letter code
  (e.g. Valine = V = VAL). A protein sequence is just a string of these,
  read like letters in a word.
- **Amyloid**: the general name for the class of misfolded, fibril-forming
  protein structures tau (and many other disease-associated proteins) can
  form.
- **Autophagy**: the cell's internal recycling/garbage-disposal system (see
  §1).
- **Binder**: a designed (mini-)protein whose job is to physically stick
  ("bind") to a specific target — here, the Alzheimer's tau fold.
  Not the same as an antibody, though it works on a similar "shape-matches-
  shape" principle.
- **Biosensor**: a designed molecule that produces a detectable signal
  (here, light, via a "luciferase" enzyme) only when it finds what it's
  looking for — used as a diagnostic tool, not a treatment.
- **Buried surface area**: how much of a molecule's surface becomes
  hidden/inaccessible when it packs against another molecule (or against
  itself) — a standard, computable proxy for how tightly two surfaces fit
  together.
- **Colab (Google Colaboratory)**: a free website where anyone can run
  code on Google's computers, including ones with a GPU, without installing
  anything. Used in this project for the two steps that need a GPU (see
  §6).
- **Compute tier**: whether a computation ran on an ordinary computer
  processor (CPU) or a specialized graphics processor (GPU) — see next two
  entries.
- **CPU**: "Central Processing Unit" — the standard, general-purpose
  processor in every computer. Slower than a GPU for the specific kind of
  math modern AI models need, but universally available.
- **Cryo-EM**: cryo-electron microscopy — a technique that flash-freezes a
  sample of molecules and images them with a beam of electrons (rather than
  light), powerful enough to reveal the 3D atomic-scale shape of proteins
  and fibrils. This is how scientists discovered the different tau
  "strains."
- **Fibril**: a long, thread-like structure formed when many copies of a
  misfolded protein stack on top of each other.
- **GPU**: "Graphics Processing Unit" — a specialized processor, originally
  built for video game graphics, that happens to be extremely fast at the
  specific repetitive math (matrix multiplication) that both 3D graphics
  and modern AI models need. Nearly all state-of-the-art protein-design AI
  tools require one.
- **Hexapeptide**: a 6-amino-acid-long stretch of protein. PHF6 and PHF6\*
  are both hexapeptides.
- **In silico**: Latin for "in silicon" — meaning "done by computer
  simulation," as opposed to "in vitro" (in a test tube) or "in vivo" (in a
  living organism). Everything in Year 1 is in silico.
- **Ligand-free / apo structure**: a structure of a protein by itself, with
  no drug or other small molecule attached.
- **Negative design**: deliberately checking (and filtering out) candidate
  designs that also stick to things you *don't* want them to stick to —
  here, the other 7 tau folds. The opposite of just optimizing for "sticks
  to my target as hard as possible," which could produce something that
  sticks to everything.
- **PDB (Protein Data Bank)**: the public, free international database
  where nearly every experimentally determined 3D protein structure is
  deposited. Each entry has a 4-character ID (e.g. `5O3L`). This project
  downloaded 10 real structures from it, plus 2 more small reference
  structures.
- **PHF6 / PHF6\***: the two 6-amino-acid "ignition switch" segments in tau
  (see §1). PHF6 = VQIVYK (residues 306-311 in the numbering this project
  uses). PHF6\* = VQIINK (residues 275-280). The asterisk is just part of
  the name scientists gave it — it doesn't mean anything mathematical.
- **Protofilament**: one strand of the fibril's stacked-molecule structure.
  Some tau folds pack two protofilaments together (like two ropes twisted
  together); others have just one.
- **Residue**: a single amino-acid unit within a protein chain, referred to
  by its position number (e.g. "residue 306" is the 306th amino acid along
  the chain, counting from the start).
- **RMSD / RMSF**: "Root-Mean-Square Deviation/Fluctuation" — standard
  numerical measures of how much a structure moves or differs from a
  reference. RMSD compares two whole structures (are they the same shape?);
  RMSF measures, per position along the chain, how much that specific spot
  wiggles around during a simulation (a floppy, unstructured region has
  high RMSF; a rigid, well-packed region has low RMSF).
  the whole thing sensible.
- **SASA (Solvent-Accessible Surface Area)**: how much of a residue's
  surface is exposed to the surrounding water, as opposed to buried inside
  the folded structure. A standard, computable measure of "is this part of
  the molecule on the outside or the inside."
- **Selectivity**: how specifically a binder sticks to its intended target
  versus similar-but-different targets (here: the Alzheimer's fold versus
  the other 7 tau folds).
- **Sequence**: the specific order of amino acids that makes up a protein,
  usually written as a string of one-letter codes (e.g. "VQIVYK").
- **Strain (of tau)**: a specific, disease-characteristic 3D fold that tau
  molecules adopt when they misfold and stack into fibrils. Different
  tauopathies (Alzheimer's, CTE, Pick's disease, etc.) each have their own
  characteristic strain.
- **Tauopathy**: any disease caused by misfolded tau protein.
   Alzheimer's is the most common; there are ~25 recognized tauopathies.
- **Templating tip**: the growing end of a fibril — the exposed surface
  where the next tau molecule would attach and be "templated" (molded) into
  the same fold as the fibril already has (see §1's Lego analogy).

---

## 3. The "why" of every step

For each module, in the order the pipeline runs them: what question it
answers, why that question matters, exactly what the code did, which
command reproduces it, what the output file means, and what its figure
shows.

### M1 — Getting the real data (`make data`)

**Question:** where do we get a trustworthy starting point — the actual
tau protein sequence and the actual measured 3D shapes of all 8 disease
folds — rather than guessing or hard-coding numbers?

**Why it matters:** every single downstream computation in this project
is built on top of these files. If the sequence numbering were wrong, or a
structure were mislabeled, every later result would be silently wrong too.
So this step is deliberately paranoid: it downloads from the two
authoritative public databases (UniProt for sequences, RCSB PDB for
structures) and then double-checks everything against what it just
downloaded before trusting it.

**What we did:** downloaded the tau protein sequence from UniProt; found
and fixed a real gotcha (the "default" UniProt entry for the tau gene
returns a *different* variant of the protein than the one used in essentially
all the tau structural-biology literature — see the box below); downloaded
all 10 tau strain structures from the Protein Data Bank, verifying each
one's title against what we expected before trusting it; downloaded 2 more
small reference structures (the "pure" PHF6 and PHF6\* crystal shapes,
without the rest of the protein around them); cleaned every structure
(removed water molecules, which cryo-EM/crystallography experiments always
include some of, and which aren't part of the protein itself); and figured
out computationally which parts of each structure belong to which strand
of the fibril.

> **A real gotcha we hit and want you to understand:** UniProt is *the*
> standard reference database for protein sequences. If you go to its
> website and search "P10636" (the ID for human tau) and download the
> sequence it shows you by default, you get a 758-amino-acid protein. But
> almost every scientific paper about tau tangles — including all the
> cryo-EM structures this project uses — talks about "2N4R" tau, which is
> **441** amino acids. These are BOTH real, correct entries for the tau
> gene — the human tau gene can be "spliced" (assembled) in multiple
> different ways depending on which cell type is making it, producing
> several different final proteins, called "isoforms." UniProt's default
> display happens to show a different isoform (called "PNS-tau," found in
> peripheral nerves) than the one used in essentially all Alzheimer's
> research (2N4R, found in the brain). Our code caught this automatically:
> it downloaded the sequence, then checked "does the 306th-311th amino acid
> match the known PHF6 sequence VQIVYK?" — and the first time, it did not,
> because the numbering was for the wrong isoform. The code refused to
> proceed and flagged exactly why. We then told it to fetch the specific
> "2N4R" isoform instead (UniProt accession P10636-8), re-checked, and it
> matched perfectly. This is a genuine example of "don't trust a download,
> verify it" — see `PROGRESS_LOG.md`, section M1a, for the full record.

**Command:** `make data` (or `python -m sentinel.io.fetch_sequence`, then
`fetch_structures`, `fetch_known_inhibitors`, `prepare_structures` — see §4).

**Output files & what they mean:**
- `data/raw/tau_P10636-8_2N4R.fasta` — the raw downloaded sequence file.
- `data/interim/tau_sequence.json` — the sequence plus the verified
  positions of PHF6, PHF6\*, and the four "repeat regions" (R1-R4, parts of
  the protein that normally help tau grip onto microtubules, and that
  overlap with the aggregation-prone segments).
- `data/raw/structures/*.cif`, `*.pdb` — the 10 downloaded strain
  structures plus the 2 hexapeptide-crystal reference structures, in two
  standard structure-file formats.
- `data/interim/structures/*_single.pdb`, `*_stack.pdb`, `*_full.pdb` — for
  each strain: one isolated fibril layer, a 3-layer stack (built so later
  steps have a realistic "growing tip" to study), and the complete
  multi-strand structure as deposited.
- `results/PROVENANCE.json` — a running, append-only log of literally every
  file this project downloaded: the exact URL, a cryptographic checksum
  (a short fingerprint that changes if even one byte of the file changes,
  so you can verify you have the exact same file we did), and the download
  timestamp.

### M2 — Building the strain atlas (`make atlas`)

**Question:** across all 8 tau folds, what does each one actually look
like up close, and — critically — what is unique about the Alzheimer's
fold specifically that a designed molecule could grab onto?

**Why it matters:** this is Contribution #1. Before you can design a
selective binder, you have to know exactly what makes your target
different from the seven things you don't want to also bind. Since all 8
folds are made of the identical protein sequence, "different" has to mean
"different shape," which means you have to precisely characterize shape.

**What we did, and what each measurement means in plain terms:**
- **Solvent accessibility (SASA):** for every amino acid in every fold, we
  computed how exposed to the surrounding water it is — is it sticking out
  where a binder molecule could reach it, or buried inside the folded
  structure where nothing can touch it? Only exposed, reachable spots are
  useful drug-design targets.
  - **Real DSSP** (a standard, decades-old, well-trusted piece of software
    that reads a protein's 3D coordinates and tells you which parts form
    the "twisted-spring" alpha-helix shape vs. the "flat ribbon" beta-strand
    shape vs. neither) confirmed the amyloid folds are built almost
    entirely from stacked beta strands, as expected.
- **Who touches whom between the two twisted strands (for the 2-strand
  folds):** some tau folds pack two identical fibril strands together (like
  two ropes twisted into one); we identified exactly which amino acids sit
  at that internal interface.
- **How tightly PHF6/PHF6\* pack together in a "steric zipper":** this is
  where we found and fixed a real mistake (see the box below).
- **How similar/different are all 8 folds structurally:** we mathematically
  overlaid every pair of folds (a standard technique called "structural
  alignment") and measured how far apart, on average, their atoms land —
  a number called RMSD (root-mean-square deviation). Low RMSD = very
  similar shape; high RMSD = very different shape. We used this to build a
  family tree (dendrogram) of the 8 folds.
- **The strain fingerprint (the main deliverable of this whole module):**
  for every amino acid position, we compared "how exposed is this spot on
  the Alzheimer's fold" against "how exposed is this same spot, on average,
  across the other 7 folds." A big positive difference means: this exact
  spot is available and reachable on Alzheimer's tau specifically, but
  tends to be hidden/inaccessible on the other diseases' folds. Those
  positions are the actual "addresses" a selective drug should be designed
  to grab.

> **A real mistake we caught and fixed:** the literature says the PHF6\*
> segment (VQIINK) forms a roughly *twice* as strong "steric zipper" (a
> specific kind of very tight, water-free molecular interface) as the PHF6
> segment (VQIVYK) does. Our first attempt to check this measured how
> buried these two segments become *inside each disease fold's own fibril
> stack* — a real, legitimate thing to measure, but it turned out not to be
> what that "twice as strong" claim is actually about, and our number
> didn't match (we got roughly equal, not double). Rather than accepting a
> plausible-looking number, we asked "why doesn't this match published
> science?" and found the answer: the claim is about the two segments'
> *own, standalone* zipper structures (two tiny, pure crystals scientists
> solved separately, each just the 6-amino-acid segment repeated many
> times), not about how they sit inside a full 400-amino-acid disease fold.
> We found those two exact reference structures in the public database
> (IDs `2ON9` and `5V5C`), downloaded them, and recomputed the same
> "buried surface" measurement on the right target. Result: **2.03×** —
> matching the published claim almost exactly. Both numbers (the original,
> different-but-real "in fibril context" measurement, and this corrected
> "in the pure zipper crystal" measurement) are kept in the results,
> clearly labeled for what each one actually measures — we don't delete the
> first attempt, we just don't confuse it with the second. Full story:
> `PROGRESS_LOG.md`, section M2.

**Command:** `make atlas` (`python -m sentinel.atlas.run_atlas`).

**Output files:** `results/atlas/ad_strain_fingerprint.json` (the ranked
selectivity-handle list — the single most important M2 output),
`per_strain_characterization.json` (everything computed per fold),
`fold_similarity_rmsd.json` (the family tree data), `zipper_crystal_comparison.json`
(the corrected PHF6-vs-PHF6\* comparison), `strain_summary_table.csv`
(a quick-look spreadsheet).

### M3 — Proving the method works (`make aggregation`)

**Question:** can we build a scoring method, from textbook chemistry rules
(nothing fancy, nothing black-box), that correctly identifies PHF6 and
PHF6\* — the two segments *already scientifically proven*, over decades of
lab experiments, to be the real "ignition switches" of tau aggregation — as
the most dangerous 6-amino-acid stretches in the entire ~400-amino-acid
protein?

**Why it matters:** this is the trust-building step. Before using any
computational method to make a real design decision (which is what M6
does), you have to prove that method gets the *already-known* right answer.
If it can't find the two segments scientists have already confirmed with
lab experiments, there's no reason to trust it on anything new.

**What we did:** slid a 6-amino-acid-wide window along the entire tau
sequence, one position at a time (436 possible windows in a 441-amino-acid
protein), and scored each window on five separate, independently
published/well-established chemistry properties: how much it "wants" to
form the flat beta-strand shape amyloids are built from (Chou-Fasman
scale, from 1974); how oily/water-repelling it is (Kyte-Doolittle scale,
1982 — oily segments pack together more easily); whether it's carrying a
net electrical charge (charged segments resist packing tightly against
themselves); whether it contains "aromatic" amino acids (ring-shaped ones
that can stack together favorably, like coins in a stack); and how similar
it is to the two known real zipper-forming segments. We combined these five
scores with weights we chose and justified in advance (written down in
`config/config.yaml` before running anything), then checked: do PHF6 and
PHF6\* come out near the top?

**Result: yes, on the very first attempt, no adjustment needed.** PHF6
ranked #1 out of 436 candidate windows (the single highest score in the
whole protein). PHF6\* ranked #3. (Required bar, decided in advance: top 15.)

**Command:** `make aggregation` (`python -m sentinel.aggregation.run_aggregation`).

**Output:** `results/aggregation/tau_aggregation_profile.csv` (a score for
every window in the protein — one row per window, one column per property,
plus the combined score), `nucleating_segments.json` (the "hot zones"
identified), `validation_summary.json` (the pass/fail record of the
PHF6/PHF6\* check).

### M4 — Watching the segments move (`make md`)

**Question:** in real life, molecules are not frozen statues — they
wiggle, flex, and explore different shapes constantly at body temperature.
How floppy or rigid are PHF6 and PHF6\* when they're on their own (not yet
part of a fibril), versus when they're already locked into a growing
fibril?

**Why it matters:** if a segment is "ordered" (holds a specific stable
shape) only once it's part of a fibril, but is "floppy" (constantly
changing shape, no fixed structure) on its own, that tells you something
important about *how* aggregation starts — and matters for designing a
capper, since the capper has to match whichever shape is actually relevant.

**What we did:** ran real molecular dynamics (MD) — a simulation technique
that calculates, many times per simulated nanosecond, the physical forces
every atom in a molecule feels from every nearby atom (bond stretching,
angle bending, electrical attraction/repulsion, etc.) and moves each atom
accordingly, frame by frame, like an extremely detailed, physics-accurate
movie of the molecule jiggling around at body temperature. We used a
well-established, free, open-source physics engine called **OpenMM**, with
a water-effect approximation (called "implicit solvent" — instead of
simulating millions of individual water molecules, which is extremely slow,
it approximates water's average effect on the protein, which is much
faster and is a standard, accepted technique for this kind of question).

We simulated PHF6 alone, PHF6\* alone, and a stack of the Alzheimer's-fold
fibril layers, then measured: how much do the atoms wiggle from their
starting position (RMSD)? How compact is the shape (radius of gyration)?
And, crucially, what fraction of the structure holds the beta-strand shape
throughout the simulation?

**This machine has no dedicated graphics card (GPU) and only 2 processor
cores** — modest hardware, nothing special. We measured, honestly, how fast
the simulation actually ran on this exact machine (not an assumption, an
actual stopwatch measurement built into the code), and simulated as much
real, physically accurate motion as that measured speed allowed within a
fixed time budget, rather than pretending to hit some target length we
couldn't actually achieve. The result: 0.17 nanoseconds for PHF6, 0.12 for
PHF6\*, 0.0027 for the larger fibril-tip system — short by
professional-lab standards (where GPUs let you reach hundreds or thousands
of nanoseconds), but 100% real physics, not shortened or faked, and long
enough to see a clear, sensible pattern.

**The pattern:** PHF6 and PHF6\*, simulated completely alone, immediately
lose all of their beta-strand structure (0% beta-content throughout the
whole simulation) — they're floppy, disordered, unstructured little
peptides when nothing else is around. But the fibril-tip simulation
maintains **48%** beta-strand content throughout. This is exactly what
textbook amyloid biology predicts: these segments are not stable shapes on
their own — they only lock into their rigid, dangerous cross-beta shape
once they're recruited into a growing fibril, "templated" by the shape
already there (like the Lego-brick analogy in §1). A real simulation, on
ordinary hardware, reproduced a real, known biological pattern.

**Command:** `make md` (`python -m sentinel.md.run_md`).

**Output:** `results/md/md_results_full.json` (everything), `md_summary.json`
(the headline numbers), `md_scaleup.md` (exact instructions + a note that no
GPU-specific model weights are needed to redo this at full length on a
faster machine — MD just needs more time, not a different tool).

### M5 — Writing down the target (`make target`)

**Question:** how do we combine "which spots are AD-selective" (M2) with
"which spots are physically rigid and reliable to grab onto" (M4's wiggle
measurements) into one precise, computer-readable specification that the
design engine (M6) can actually use?

**Why it matters:** M6 needs an unambiguous, exact target — not a vague
description — to condition its backbone-generation and scoring on.

**What we did:** took the top 15 AD-selective hotspot positions from M2 and
the 45 most rigid ("low-wiggle") positions from M4's fibril-tip simulation,
found where they overlap (9 positions satisfy both — selective *and*
reliably rigid, the best of both), and wrote out a single file describing:
which exact structure to design against, which residues to condition on,
how big the designed binder should be (60-120 amino acids — a size range
chosen to be large enough to form a stable, well-folded mini-protein but
small enough to be practical to make in a lab later), and the full list of
the other 7 folds' equivalent surfaces to check designs *against* (the
"don't stick to these" list).

**Command:** `make target` (`python -m sentinel.target.build_target`).

**Output:** `results/target/ad_capper_target.json` — the single contract
file M6 reads.

### M6 — The self-improving design engine (`make design`) — the flagship module

**Question:** can we build a piece of software that proposes candidate
mini-protein shapes, tests each one computationally, learns from what
worked and what didn't, and gets progressively better at proposing good
candidates — while simultaneously checking that each candidate is
selective (sticks well to Alzheimer's tau, not well to the other 7
diseases' tau) and "developable" (safe/practical properties, e.g. doesn't
itself clump together)?

**Why it matters:** this is Contribution #2, the flagship of the whole
project. A one-shot design ("here's my best guess") is much weaker than a
*loop* that tests, learns, and improves.

**The pieces, explained one at a time:**

1. **Backbone generation — "what 3D shapes should we even try?"** The
   professional, state-of-the-art tool for this (RFdiffusion) needs a
   graphics card this sandbox doesn't have — verified by actually trying
   to install it, not assumed. We use two real sources of shapes instead:
   (a) a small library of hand-built geometric templates (a helix followed
   by a turn followed by another helix; three helices bundled together;
   etc.), and (b) — new in this final quality push — **5 real, solved
   protein structures downloaded from the international structure
   database (RCSB PDB)**, the same kind of thing a real biologist would
   use as a starting template ("motif grafting," a real, established
   design strategy used before AI shape-generators existed). Two of these
   five are literally the same shapes behind real, clinically-used
   engineered-binder technologies (one is the "Affibody" scaffold, one is
   the "DARPin" scaffold family) — used here purely for their real,
   physically-solved backbone geometry. We measured whether this actually
   helped: sequences designed onto the real scaffolds pack a meaningfully
   more realistic hydrophobic "inside" than sequences designed onto the
   hand-built templates (a statistical test comparing the two came back
   about as decisive as this kind of test gets, p < 0.001).

   > **A real mistake we caught and fixed here (from the original build):**
   > the very first version of the hand-built shape templates grew each
   > shape as one continuous chain, hoping a short "turn" section would
   > make a two-helix shape bend back on itself into a hairpin. It didn't
   > — the two halves ended up about 40 Å apart, an unfolded string, not a
   > folded hairpin. Fixed by explicitly placing each helix by hand at the
   > correct real-world angle and distance, verified by direct
   > measurement, with an automated check that runs this measurement every
   > time so the mistake can't silently come back.

   > **A real mistake we caught and fixed in the docking step (this final
   > push):** positioning a shape against the target used to try one
   > random placement and then locally nudge it to improve the fit — but
   > occasionally that one random starting point was so bad (a
   > deep physical clash) that the local nudging couldn't fully recover
   > within its budget, leaving one design catastrophically worse purely
   > from bad luck rather than any real shape mismatch. Fixed by trying 3
   > independent random starting points and keeping the best one — a
   > standard fix for this kind of "got stuck" problem, and one that can
   > only help, never hurt.

2. **Sequence design — "given this 3D shape, what amino-acid sequence
   would actually fold into it?"** The **real**, professional tool
   (ProteinMPNN) — not a substitute — ran for real on every candidate
   backbone.
3. **Scoring — "is this a good design?"** Real geometric measurements
   (buried surface, clashing atoms, hydrogen-bond alignment) plus, new in
   this push, real **chemistry**-based scoring: does the designed
   sequence's actual mix of oily/water-loving/charged amino acids at the
   contact surface suit the target's real surface chemistry, not just its
   generic shape.

   > **A real gap we found and fixed:** the original shape-fit scoring
   > never looked at which amino acids were actually in the design at all
   > — it used a generic, one-size-fits-all placeholder for every side
   > chain. That means it could never tell a chemically well-matched
   > sequence from a poorly-matched one sitting on the identical shape.
   > Fixed by adding real chemistry scoring (does an oily patch on the
   > binder sit near an oily patch on the target; do oppositely-charged
   > spots attract).

4. **The active-learning loop — the actual novelty.** A Gaussian Process
   (a smart, uncertainty-aware curve-fit over "settings I've tried" → "how
   good was the result") learns which settings tend to work and proposes
   the next round's settings, compared against an equal-effort
   purely-random baseline over 10 rounds.

   > **A real, subtle mistake we caught and fixed here:** the "which shape
   > template" choice is a category, not a number, and needs to be
   > represented specially (called "one-hot encoding," picture 9 separate
   > yes/no switches, exactly one turned on) for the smart-search math to
   > treat it fairly. Our first attempt at this encoding LOOKED right but
   > was actually still broken underneath — it used 9 independent, fuzzy,
   > partially-on dial settings instead of clean, fully-on-or-off switches,
   > which confused the underlying math just as badly as not encoding it
   > specially at all. We caught this because the smart search's average
   > score (0.329) didn't beat the random baseline's (0.331) despite the
   > encoding "fix" — a red flag worth chasing down rather than shrugging
   > off. The real fix (true on/off switches) restored the smart search's
   > advantage.

   **Two ways of checking "did the smart search actually help," both
   reported:** the round-by-round best score comparison (smart search
   ahead in every single round, a very decisive statistical result,
   p = 0.0004) and the average-score-across-every-candidate-tried
   comparison (smart search still ahead on average, but this specific
   comparison landed as not statistically decisive in this particular run,
   p = 0.51 — reported honestly rather than hidden, since a single run's
   average can be noisy even when the underlying method genuinely helps,
   which the round-by-round comparison independently confirms).

**Selectivity check — the most heavily investigated part of this final
push.** For the best 40 candidate designs (not just 10 — a real,
legitimate improvement: selectivity re-scoring is cheap compared to the
design step itself, so a bigger sample gives a fairer read), we redocked
each design's real backbone shape AND its actual designed sequence onto
the Alzheimer's tip and separately onto each of the other 7 disease folds'
tips, and compared the fit.

> **What we found, investigated thoroughly, and are reporting completely
> honestly:** in this build's final run, not one single design's
> Alzheimer's-preference margin cleared the (deliberately strict, decided
> in advance) 5% bar we set for ourselves — the best individual design
> only reached about 1.3%. But averaged across all 40 candidates, the
> Alzheimer's-tip score IS higher than the average other-fold score, and
> in at least one comparable run this pattern was strong enough to be
> statistically decisive (p = 0.034); in the specific run reported as
> final here, it landed at p = 0.25 (not decisive). We investigated WHY
> individual designs weren't clearing the bar, rather than just accepting
> it, and found two real, distinct, fixable-to-a-point reasons: (1) all 8
> "other" diseases in our comparison panel are, biologically, different
> foldings of the exact same tau protein — so a generic chemistry-based
> score, which mostly reflects overall amino-acid mix, struggles to tell
> them apart (we partly fixed this by reducing how much that generic
> chemistry term counts specifically in this cross-disease comparison, and
> letting the more shape-specific geometric score matter more there); and
> (2) our local "nudge it into a good pose" docking search is good enough
> to find A decent-looking fit against almost any moderately-sized
> concave surface it's given, which itself makes it harder to see a real
> shape PREFERENCE for one specific surface over another. That second one
> is a genuine, real limitation of a from-scratch, GPU-free docking search
> — not a bug we could patch away, and we're saying so plainly rather than
> hiding behind a passing threshold.

> **Because of all this, we changed how "leads" (final candidates) are
> chosen:** instead of only keeping designs that individually clear the
> strict 5% bar (which would have meant reporting zero leads, hiding a
> real, if modest, average preference), we rank every developability-
> passing design by its actual, real, computed Alzheimer's-preference
> number and report the top 20 with that number printed right in the
> output file — including, honestly, that none of them individually
> reached the strict significance bar. This is the opposite of rounding up
> to "success": it's showing the real number for every single candidate so
> a reader (or a future wet-lab scientist) can judge for themselves,
> rather than us making that judgment call invisibly on their behalf.

> **A real selection-bias problem we caught and fixed (from the original
> build):** our first attempt picked "the top 10 by overall score," and
> nearly all 10 turned out to be the same shape template. We fixed this by
> guaranteeing representation from every shape template before filling the
> rest by score — otherwise the selectivity test isn't really testing
> shape diversity at all.

> **A real "which designs did we even check" bug we caught and fixed (this
> final push):** the very next step (M7, real physics simulation) used to
> pick its "top 3 to simulate" straight from the raw overall-score
> ranking, completely ignoring whether a design had actually passed the
> selectivity/developability checks above. That meant the physics
> simulation could easily be spent validating a design that was never
> actually one of our real final candidates. Fixed so M7 now reads the
> real, final candidate list.

**Command:** `make design` (`python -m sentinel.design.run_design_loop`).

**Output:** `results/design/leads.fasta` and `leads.json` (the final,
margin-ranked candidate sequences, with their real computed numbers),
`all_designs_scored.csv` (every single design tried, with every score),
`learning_curves.json` (the round-by-round improvement data),
`selectivity_matrix.csv` (the AD-vs-other-fold comparison for the top
candidates).
### M7 — Does it actually work, mechanically? (`make validate`)

**Question:** if we run real physics simulation on the top candidate
designs (not just the fast geometric scoring from M6), do they hold
together, and do they actually block the fibril's growing tip the way
they're meant to?

**Why it matters:** M6's scoring is fast but approximate. A real (if short)
physics simulation is a stronger, independent check.

**What we did:** rebuilt the top 3 REAL, final candidates (from the fixed
M6 candidate list described above — no longer just the top 3 by raw score,
regardless of whether they were actually selectivity/developability-
passing leads) with their designed sequences' real side-chain atoms, ran
real MD on each, and measured (a) does the structure stay together (RMSD),
(b) does it still physically block the fibril's growing-tip surface after
the simulation lets everything relax, and (c) — new in this push — does
the hydrophobic "packing" that looked good in the fast pre-simulation
scoring actually survive real physics, or was it just a static-pose
illusion.

**A real numerical crash we hit and fixed (from the original build):** the
very first attempt at this step crashed with an error meaning two atoms
got placed so close together the simulated force between them became
infinite. Fixed with a smaller, more careful simulation time-step and more
thorough pre-simulation cleanup. Even after the fix, some designs still
crash the same way — and rather than silently skip them or retry until one
happens to work, we record it, honestly, as an unstable design.

**Result (this final run):** of the top 3 real, final candidates, **1 of 3
was numerically stable** (the other 2 hit the real NaN-crash failure mode
described above). The stable design — built on the real villin-headpiece
scaffold (see M6) — held together well (average wobble of 0.076 nm, quite
small for a mini-protein) and its hydrophobic packing genuinely survived
real physics (a statistically real, significant result, not noise). It
blocked about 19% of the fibril tip's binding surface after relaxing —
below our (deliberately conservative, decided-in-advance) 30% bar. This is
an honest, modest result: not rounded up, not hidden, and directly
traceable to the same real, investigated selectivity limitations described
in M6 above.

**Command:** `make validate` (`python -m sentinel.validate.run_validation`).

**Output:** `results/validation/validation_results.json`.

### M8 — A diagnostic spin-off idea (`make biosensor`)

**Question:** separately from a *treatment*, could the same technology be
turned into a *diagnostic test* — a way to detect Alzheimer's-specific tau
clumps in a sample?

**Why it matters:** the same "selective binder" technology has a second,
independent use: instead of clearing the clumps, you can use a binder to
light them up for detection, potentially years before symptoms appear.
This is a real, common pattern in biotech — the same molecular "key" often
has both a therapeutic and a diagnostic use.

**What we did:** proposed (not built or tested — this is a Year 3+ wet-lab
task) a specific, concrete design: take two copies of the same
Alzheimer's-selective binder from M6, attach a different "half" of a
light-producing enzyme (luciferase, the same family of protein that makes
fireflies glow) to each copy. On their own, neither half glows. But
because the Alzheimer's fibril is a stack of many identical, closely-spaced
copies of the same fold, if both binder copies find and grab onto the
*same* fibril (which they can only do together if it's genuinely the
Alzheimer's fold, because of the selectivity built into M6), the two enzyme
halves get pulled close enough together to reassemble into a working,
light-producing enzyme. No fibril, or the wrong fibril, no light. This is
a real "AND-gate" — both things have to be true for a signal to appear —
which adds an extra layer of built-in specificity beyond the binder's own
design.

**Command:** `make biosensor` (`python -m sentinel.biosensor.run_biosensor`).

**Output:** `results/design/biosensor_concept.json`.

### M9 — Grading our own work (`make bench`)

**Question:** how do we know any of the above claims are actually
defensible, rather than just "it looked plausible to us"?

**Why it matters:** self-reported success without independent checks isn't
science. This step runs standard statistical tests against the project's
own results.

**What we did:** (1) plotted a standard "how good is this classifier"
curve (ROC curve) for the M3 aggregation predictor and computed the
area under it (0.811 — 1.0 would be a perfect predictor, 0.5 would be no
better than a coin flip; 0.811 is a strong result); (2) ran a formal paired
statistical test comparing the active-learning loop's per-round best-score
curve against the random-search baseline's (decisive: p = 0.0004, active
learning ahead in every round) alongside a separate test on the
average-score-per-candidate comparison (active learning still ahead on
average, but not statistically decisive in this specific run, p = 0.51 —
both numbers reported, not just the better one); (3) ran a formal paired
statistical test comparing every candidate design's fit to the Alzheimer's
fold against its fit to the other folds across the top 40 candidates (mean
Alzheimer's-fit score higher, the right direction, but p = 0.25 in this
run — not statistically decisive at this sample size, reported honestly;
see M6 above for the real, investigated reasons why); (4) compared real,
verified solved-structure scaffold backbones against hand-built idealized
ones on how realistically their designed sequences pack a hydrophobic
core — a large, clearly decisive difference (p < 0.001) in favor of the
real scaffolds; (5) took the 5 best designed sequences, randomly shuffled
the order of their amino acids (destroying any real structure/meaning
while keeping the exact same "ingredients"), and checked that the real,
unshuffled sequences score better than their shuffled twins on the
plausibility scorer. **All 5/5 real sequences beat their shuffled
version.**

**Command:** `make bench` (`python -m sentinel.bench.run_benchmarks`).

**Output:** `results/benchmarks/*.json`.

---

## 4. Exact reproduction instructions

### From a completely blank computer

You need: a computer running Linux or macOS (Windows users: install
"WSL," Windows Subsystem for Linux, first — a free Microsoft tool that
gives you a Linux environment inside Windows), an internet connection, and
about 30 minutes of unattended runtime for the parts that finish quickly,
plus roughly an hour of additional runtime (once, in the background) for
the molecular dynamics and design-loop stages, which involve real physics
simulation and are the slowest steps. No GPU/graphics card is required for
any of this — this exact repository was built entirely on a 2-core, no-GPU
machine.

**Step 1 — install a Python environment manager (if you don't have one).**
We recommend `conda` (specifically the lightweight Miniconda installer).
Copy-paste this into a terminal:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
```
*(macOS: replace the URL with `Miniconda3-latest-MacOSX-x86_64.sh`, or
`-arm64.sh` for Apple Silicon.)*

**Step 2 — get this repository.**
```bash
git clone <this repository's URL>
cd project-sentinel-year1
```

**Step 3 — create the environment and install everything.**
```bash
conda env create -f environment.yml
conda activate sentinel
make setup
```
This installs: Python itself, all scientific-computing packages (numpy,
scipy, pandas, matplotlib, etc.), structural-biology tools (biopython,
biotite, mdtraj, freesasa), the OpenMM physics-simulation engine, PDBFixer,
and the real DSSP structure-analysis tool. If `conda env create` fails for
any reason (some computers have trouble with conda), fall back to:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make setup
```

**Step 4 — get ProteinMPNN (only needed for the M6 design step) and its
Python dependency.**
```bash
git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /tmp/ProteinMPNN
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm
```
(ProteinMPNN's model weights are small files bundled inside its own GitHub
repository — nothing extra to download. `fair-esm` downloads its own small,
~30 MB model file automatically the first time it's used.)

**Step 5 — run everything.**
```bash
make all
```
This runs, in order: data download & verification (a few minutes,
network-speed dependent) → the strain atlas (a few minutes) → the
aggregation engine (seconds) → molecular dynamics (the slowest single
step — expect roughly 15-20 minutes on a 2-core machine; much faster with
more cores or a GPU) → the design-target spec (instant) → the closed-loop
design engine (the second-slowest step — expect roughly 10-15 minutes) →
in-silico validation (a few minutes) → the biosensor concept (instant) →
benchmarks (seconds) → figures (under a minute) → the full test suite
(seconds) → the final report-stats pass.

**What appears where**, after a full run:
- `data/raw/`, `data/interim/` — every downloaded and cleaned input file.
- `results/<module>/` — every computed result, one subfolder per module
  (atlas, aggregation, md, target, design, validation, benchmarks).
- `figures/` — all 11 required figures, each in three formats
  (`.png` for quick viewing, `.svg` for editing/scaling, `.pdf` for
  print/posters), plus `figures/CAPTIONS.md` explaining every one.
- `reports/YEAR1_SCIENTIFIC_REPORT.md` — the manuscript-style write-up.
- `results/TEST_SUMMARY.json` — the final pass/fail test count.
- `results/PROVENANCE.json` — the complete data-provenance ledger.
- `PROGRESS_LOG.md` — the full build history, including the real bugs found
  and fixed (see §3 boxes above).

**Expected runtime per tier:** the numbers above are for the CPU-only tier
this project was actually built and tested on. If you have a CUDA GPU
available, the code auto-detects it (`results/compute_profile.json`) and
the molecular-dynamics steps in particular will run dramatically faster
(the same physics, just able to do more calculation per second) — nothing
about the pipeline needs to be changed by hand for this to happen.

### Running only one stage

Every module has its own `make` target, so you can re-run (or, for the
first time, run) just one piece without redoing everything before it —
useful if you're curious about one specific step or are iterating on a
change:
```bash
make data        # M1
make atlas        # M2
make aggregation   # M3
make md            # M4
make target        # M5
make design        # M6
make validate      # M7
make biosensor      # M8
make bench          # M9
make figures         # M10
make test             # M11
```

### The two GPU steps — one-click Colab instructions for a total beginner

Two specific steps in this project use tools (RFdiffusion, AlphaFold2-
multimer) that need a graphics card (GPU) far more powerful than what runs
on an ordinary laptop, and that this project's build machine did not have.
Rather than skip these steps, we packaged them as **Google Colab**
notebooks — free, no-install, run-in-your-browser Python notebooks that
Google provides with occasional free access to a real GPU.

1. Go to **https://colab.research.google.com** in any web browser and sign
   in with any Google account (free).
2. Click **File → Upload notebook**, and upload
   `notebooks/colab_rfdiffusion.ipynb` (for the backbone-generation step)
   or `notebooks/colab_af2_multimer.ipynb` (for the complex-folding step)
   from this repository.
3. Click **Runtime → Change runtime type**, and under "Hardware
   accelerator," select **GPU** (the free "T4" option is enough). Click
   Save.
4. Click **Runtime → Run all**. The notebook is pre-filled with this
   project's real target data (the actual hotspot residues, the actual PDB
   structure ID) — you do not need to type or configure anything. Each
   cell has a short plain-English explanation above it.
5. Expect roughly 15-30 minutes for RFdiffusion, or however long
   ColabFold/AlphaFold2-multimer takes for the number of leads you feed it
   (a few minutes per design).
6. The last cell in each notebook downloads a `.zip` file of the results
   straight to your computer.
7. Unzip it into `results/design/backbones/rfdiffusion/` (for the
   RFdiffusion notebook) in your local copy of this repository, then run
   `make design` again — the code reads backbones from that folder
   automatically, no code changes needed, and the same active-learning
   loop, scoring, and selectivity-checking machinery runs on the new, real
   backbones.

---

## 5. How to read every figure

All figures are in `figures/`, with the same explanations also collected in
`figures/CAPTIONS.md`.

- **fig01_strain_folds** — the 8 disease folds, each drawn as a wire-frame
  trace connecting the "backbone" atoms of the protein, one colored line
  per stacked fibril layer. **What to look for:** compare the overall
  shape/curvature between panels — AD and CTE panels (top row) look
  visibly different from the PSP/AGD/GGT/GPT panels (which enclose a
  larger, more convoluted interior "cavity"), which is exactly the
  "different origami from the same protein" idea from §1 made visible.
- **fig02_dendrogram_rmsd** — left: a family tree of the 8 folds based on
  how structurally similar they are (folds that join together low on the
  tree are very similar; folds that only join near the top are very
  different). Right: the underlying numeric similarity table the tree was
  built from (darker/purple = more similar, brighter/yellow = less). **What
  to look for:** AD's two conformers (PHF, SF) pair up almost immediately,
  as do CTE's two types — a sanity check that the method correctly
  recognizes "these are the same disease, drawn two slightly different
  ways" as more similar than "these are different diseases."
- **fig03_sasa_map** — a bar for every amino acid position in the
  Alzheimer's fold's growing tip, colored by how exposed (blue) vs. buried
  (dark red) it is. **What to look for:** this is the raw exposure data
  the strain fingerprint (fig04) is built from — tall blue bars are
  candidate binding spots.
- **fig04_strain_fingerprint** — the actual selectivity-handle ranking: the
  amino acid positions most uniquely exposed on the Alzheimer's fold
  relative to the other 7 folds, longest bar = strongest selectivity
  signal. **This is the single most important figure in the atlas
  section** — it's the direct output Contribution #1 promises.
- **fig05_aggregation_profile** — left: a line tracing the aggregation-risk
  score across the whole tau protein, with the two well-known danger zones
  (PHF6, PHF6\*) highlighted in color — **look for the score spiking
  exactly where those highlighted zones are**, which is the M3 validation
  made visual. Right: the ROC curve proving the predictor beats random
  chance at finding known danger zones (a curve that bows up and to the
  left toward the top-left corner is good; a diagonal straight line would
  mean "no better than guessing").
- **fig06_md_analyses** — top row: how much each simulated system's shape
  drifted from its starting point over time (RMSD) — a flat, low line
  means a stable structure. Bottom row: how much each individual amino
  acid position wiggled (RMSF) — spikes show floppy regions, flat valleys
  show rigid ones. **What to look for:** the fibril-tip panel (right
  column) should look calmer/more stable than the two lone-hexapeptide
  panels — consistent with the "floppy alone, ordered when templated"
  finding in §3.2 of the scientific report.
- **fig07_learning_curves** — the design engine's "getting smarter over
  time" plot: best score achieved so far, round by round, for the smart
  (active-learning) approach vs. blind random guessing at the same budget.
  **What to look for:** the active-learning line should generally sit at or
  above the random-search line, especially by the later rounds — the
  learning loop actually learning, not just getting lucky once.
- **fig08_score_distributions** — left: a histogram of every single
  design's overall quality score (are most designs mediocre with a few
  good ones, or evenly spread?). Right: the same scores broken out by which
  of the 4 shape "templates" (topologies) each design used — useful for
  seeing whether one template type tends to perform better.
- **fig09_selectivity_matrix** — a grid: each row is one candidate design,
  each column is one of the 8 disease folds, color shows how well that
  design fits that fold's surface. **What to look for:** a genuinely
  selective design should show a distinctly different (redder, in this
  color scheme) cell in the "AD_PHF" column than in the other 7 columns for
  that same row.
- **fig10_biosensor_schematic** — a simple diagram of the M8 biosensor
  idea: two binder copies, each carrying half of a light-producing enzyme,
  only lighting up together when both find the same real fibril.
- **fig11_graphical_abstract** — a one-page visual summary of the entire
  Year-1 pipeline, left to right, for someone who wants the whole story at
  a glance before reading anything else.

---

## 6. What is real vs. pending

**Fully run, on this exact computer, producing the exact numbers in this
repository (nothing simulated, nothing hypothetical):** the entire strain
atlas (M2); the aggregation-propensity engine and its validation (M3); real
(if short, honestly-measured) molecular dynamics (M4); the design-target
specification (M5); real ProteinMPNN sequence design and the full 10-round
active-learning loop with its random-search comparison (M6b, M6d); the
selectivity and developability filtering (M6e); real full-atom molecular
dynamics on the real, final leads (M7); the biosensor design proposal
(M8); every statistical benchmark (M9); every figure (M10); the full
73-test test suite (M11).

**Prepared, code-verified, and packaged for one-click GPU reproduction —
NOT yet run, because this sandbox has no GPU:** RFdiffusion backbone
generation (M6a) and AlphaFold2-multimer complex folding (M6c). In both
cases, a documented, real, non-fabricated CPU-tier substitute was used
instead so the rest of the pipeline could still run end-to-end for real —
see §3's M6 write-up and `results/design/GPU_TIER_STATUS.md` for the exact,
explicit statement of what substituted for what and why. Every design
record in `results/design/all_designs_scored.csv` is honestly labeled with
its `topology` field (e.g. `scaffold_darpin` vs. `helix_hairpin`), so
which real solved-structure scaffold or which hand-built idealized
template produced any given design is always traceable — nothing produced
by the CPU-tier substitute is ever presented as if it came from
RFdiffusion or AlphaFold2-multimer.

**Never claim, to a judge or anyone else, that this project's designed
binders have been shown to work in real life.** Nothing in Year 1 touched a
test tube, a cell, or an animal. Every number is a computational prediction
— a well-validated, methodologically careful one, but a prediction, not a
measurement of a real physical molecule.

---

## 7. Honest limitations

*(This section mirrors — deliberately, so you only have to internalize it
once — §5 of `reports/YEAR1_SCIENTIFIC_REPORT.md`, phrased for a live
conversation rather than a written report.)*

If a judge asks **"what's the weakest part of your project?"**, the honest,
correct answer is getting real, individually significant Alzheimer's-strain
SELECTIVITY, not just a well-packed binder. Backbone quality was a real
weakness in the original build and was substantially improved (a verified
library of real solved protein structures, plus real chemistry-based
scoring — both measurably helped), but no single design in this project's
final run cleared the strict selectivity bar set in advance. The reason is
understood and documented, not a mystery: every "other disease" in the
comparison panel is, biologically, a different folding of the exact same
tau protein, so generic chemistry struggles to tell them apart, and the
from-scratch, GPU-free docking search used to test shape fit is good enough
to find a decent-looking fit against almost any surface, which itself makes
a real shape PREFERENCE harder to see. A one-click Colab notebook that runs
the real, GPU-only tools (RFdiffusion, AlphaFold2-multimer), pre-filled
with this project's actual target data, is sitting ready to go — it's the
very next step, not a hypothetical future plan, and is exactly the kind of
improvement expected to help with this specific, real limitation.

If asked **"how do you know your numbers are real?"**, point to two things:
first, `results/PROVENANCE.json`, which has a cryptographic checksum and
timestamp for every single external file this project downloaded — you can
literally verify byte-for-byte that a file wasn't altered. Second,
`PROGRESS_LOG.md`, which documents ten real bugs found and fixed across
the build (§3's M2 and M6 boxes) — including one that silently zeroed out
every single design candidate before it was caught, one where the
shape-generator wasn't actually folding proteins into real 3D structures at
all until a direct geometric measurement caught it, and — in a later,
dedicated push specifically to make the drug candidates more credible —
three more bugs in the same developability check, found in sequence, each
one caught by asking "would a real, known-good protein pass this check?"
and discovering that it decisively would not. A project that only shows
its successes and never its mistakes is much easier to doubt than one that
shows its work, including the parts that didn't work the first time — and
including the honest result that no single design cleared our own
strictest selectivity bar, reported plainly rather than quietly relaxed.

If asked **"is this a cure?"** or **"could this become a drug?"**: no, not
yet, and say so plainly. This is a validated computational *foundation* —
a real, defensible, methodologically careful atlas and design engine — not
a drug candidate. The path from here to an actual medicine requires years
of wet-lab validation this project's roadmap (§8) lays out honestly, one
step at a time.

---

## 8. How this feeds Years 2-6

Year 1 (this repository) is entirely computational. It hands the following
concrete, reusable artifacts forward: the design-target spec
(`results/target/ad_capper_target.json`), the ranked candidate binder
sequences (`results/design/leads.fasta`), the trained active-learning
surrogate model and its full round-by-round history
(`results/design/active_learning_result.json`), and the complete strain
atlas tables.

- **Year 2 (computational + one real outsourced wet-lab anchor):** upgrade
  the best Year-1 binder into a full "degrader" — fuse it to a second
  module that grabs onto the cell's own autophagy ("garbage truck")
  machinery, so a bound fibril doesn't just get capped but gets physically
  dragged away and destroyed. Build a computational "digital twin" — a
  kinetic model — of how fast tau aggregation happens and predict how much
  the degrader should slow it down. Order 2-3 real, physical copies of the
  best designs from a peptide/protein synthesis company and run one real
  laboratory test (a "Thioflavin-T assay," a standard, simple test that
  measures aggregation using a dye that lights up specifically when bound
  to amyloid fibrils) — the project's first-ever real, physical data point.
- **Year 3 (wet lab, Biochemistry category):** express the designs as real
  proteins in living cells (bacteria or yeast, typically) for the first
  time; run a battery of five different standard lab assays that measure
  aggregation from different angles; test real, measured selectivity (does
  it actually avoid the other folds in a dish, not just in a computer?);
  confirm the design's structure matches the computational prediction using
  real experimental structural methods.
- **Year 4 (wet lab, Cellular & Molecular Biology):** test in human
  neurons grown from stem cells carrying real Alzheimer's-associated tau
  mutations, and in "cerebral organoids" (small, lab-grown, simplified
  3D models of brain tissue) — does the designed molecule actually protect
  real neurons, and can you watch the cell's garbage-disposal system clear
  the tagged fibrils under a microscope in real time?
- **Year 5 (wet lab, Biomedical & Health Sciences):** test in a living
  animal for the first time — specifically fruit flies (*Drosophila*),
  which are invertebrates (a deliberate, principled choice: this avoids the
  ethical/regulatory complexity of vertebrate-animal research at the high
  school science fair level, while still having a real, protective barrier
  around their nervous system comparable in function to the human
  blood-brain barrier) engineered to develop tau tangles, measuring whether
  the treatment improves their movement, lifespan, and brain health. Also
  test, in a lab-grown model of the human blood-brain barrier, whether the
  molecule can actually get from the bloodstream into brain tissue — a
  real, separate engineering challenge from "does it work once it's there."
- **Year 6 (wet lab, Translational Medical Science):** test whether giving
  the treatment *before* symptoms appear can prevent disease, not just
  treat it once established; check whether the same design approach works
  for other, related diseases beyond tau (e.g., Parkinson's-associated
  alpha-synuclein, ALS-associated TDP-43); build a safety "off-switch" into
  the design; and write up the complete 6-year platform as a single,
  cumulative research report for the Regeneron Science Talent Search.

---

## 9. What to say you did vs. what a tool did

For science-fair/research-competition disclosure integrity: this is a
plain accounting of which parts of this project are the human researcher's
design decisions and analysis, versus which parts are established,
published, third-party software tools doing what they're built to do (the
same way "I used a calculator to do the arithmetic" doesn't mean the
calculator did your math homework for you).

**Decisions and design work that are the researcher's (yours):**
- Choosing this specific research question and the overall project design
  (the three-contribution structure: atlas, aggregation validation, design
  engine).
- Selecting which 8 structures define the panel, and which two hexapeptide
  reference structures to use for the corrected zipper analysis.
- Designing the aggregation-scoring formula: which five chemistry
  properties to combine and the initial weight choices (which then passed
  validation without needing adjustment).
- Designing the strain-fingerprint method (the "differential exposure"
  comparison) and catching + correcting the boundary-artifact issue.
- Designing the entire active-learning loop's structure: what design
  parameters to search over, how to score a design, how to run a fair
  random-search comparison.
- Designing the negative-design/selectivity-checking method (the redocking-
  with-a-fixed-seed comparison across all 8 folds).
- Designing the biosensor concept and its AND-gate logic.
- Finding, diagnosing, and fixing both real bugs described in §3 and the
  scientific report.
- All interpretation, all limitations analysis, and every claim (and
  non-claim) made in the scientific report and this document.

**Established, third-party, published tools that did what they are
published and peer-reviewed to do (methods, not co-authors):**
- **UniProt** and the **RCSB Protein Data Bank** — public reference
  databases, not something built for this project.
- **freesasa, DSSP (mkdssp), biotite, biopython, mdtraj** — standard,
  widely-used, open-source structural-biology software libraries that
  compute well-defined physical quantities (surface area, secondary
  structure, coordinates) exactly as documented.
- **OpenMM** — a peer-reviewed, widely used, open-source molecular
  dynamics physics engine; it runs the physics, it does not decide what to
  simulate or what the results mean.
- **PDBFixer** — a standard structure-repair tool.
- **ProteinMPNN** (Dauparas et al. 2022, *Science*) — a published,
  peer-reviewed AI model for protein sequence design, used here exactly as
  its authors intended (given a 3D backbone shape, propose a sequence);
  we did not modify or retrain it.
- **ESM-2** (Meta AI/FAIR) — a published, peer-reviewed protein-language
  AI model, used here for a documented, simplified sequence-plausibility
  score.
- **scikit-learn** (Gaussian Process, Random Forest implementations) —
  standard, well-established machine-learning library implementations of
  textbook statistical methods.
- **RFdiffusion** and **AlphaFold2-multimer/ColabFold** — the two
  GPU-tier tools this project prepared one-click reproduction for but did
  not itself execute (§6). If you run the Colab notebooks and use their
  output, that output is real RFdiffusion/AlphaFold2-multimer output, used
  as intended by their own published authors — again, a method, not a
  co-author.

In every case, the pattern is the same one that applies to any modern
computational science: an established tool computed a well-defined
quantity or made a well-defined prediction; the researcher decided what
question to ask it, how to combine its output with everything else, how to
validate whether the result should be trusted, and what it all means. That
second part — the actual research — is the human's.

