# Figure captions

## fig01_strain_folds
The eight cryo-EM tau fibril folds in this project's panel, each rendered as its stacked-layer C-alpha trace (one line per fibril layer; AD_PHF and AD_SF are two conformers of Alzheimer's tau, CTE_I/II two conformers of chronic traumatic encephalopathy tau). Panels are NOT on a common scale/orientation.

## fig02_dendrogram_rmsd
Left: hierarchical clustering of the 8+2 conformers by structural similarity. Right: the underlying pairwise Kabsch-RMSD matrix over the shared 306-378 core. Near-identical conformer pairs (AD PHF/SF, CTE I/II) cluster tightly; the AD/CTE folds are structurally distant from the R2-containing folds.

## fig03_sasa_map
Relative solvent-accessible surface area (SASA) at the AD fold's growth-competent templating tip, per residue, colored by burial class (buried/intermediate/exposed; Tien 2013 max-ASA scale, thresholds in config.yaml).

## fig04_strain_fingerprint
The AD strain fingerprint: residues most differentially exposed on the AD templating tip relative to the mean of the other 7 folds at the same sequence position (same tau chain — no re-indexing). These are the selectivity-handle residues used to condition backbone generation (M6).

## fig05_aggregation_profile
Left: consensus aggregation-propensity score across full-length tau (2N4R), PHF6/PHF6* highlighted (PHF6 ranks #1, PHF6* ranks #3 of 436 windows). Right: ROC curve recovering known nucleating segments (AUC=0.8107).

## fig06_md_analyses
Real OpenMM implicit-solvent MD: RMSD (top) and per-residue RMSF (bottom) for PHF6, PHF6*, and the truncated 2-layer AD fibril growing tip. Actual simulated length (not the config target) is stated per panel — see results/md/md_scaleup.md for full-length GPU reproduction.

## fig07_learning_curves
Cumulative best composite design score per round: the Gaussian-Process active-learning loop vs. an equal-budget random-search baseline (M9 ablation).

## fig08_score_distributions
Distribution of composite design scores across all evaluated designs (left) and broken out by scaffold topology (right).

## fig09_selectivity_matrix
Selectivity matrix: each top-scoring backbone's geometric-complementarity score when redocked (identical rigid-body sampling seed) onto each of the 8 folds' templating tips. Higher (redder) on the reference AD column than the negative-design columns indicates AD-selective shape complementarity.

## fig10_biosensor_schematic
Cryo-EM shows the AD fold packs as a stack of IDENTICAL monomer layers along the fibril axis (>=2 copies of the same protofilament chain within a few Angstroms of each other — verified directly in this project's own M1d protofilament clustering, e.g. (see results/design/biosensor_concept.json for the full design spec).

## fig11_graphical_abstract
Graphical abstract: the Year-1 pipeline from real cryo-EM structures to a benchmarked, AD-selective design engine.

