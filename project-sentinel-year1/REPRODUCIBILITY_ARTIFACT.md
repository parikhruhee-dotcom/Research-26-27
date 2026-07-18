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

*(Continued in the sections below.)*
