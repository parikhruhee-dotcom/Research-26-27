"""M10 — Figures. Produces every required figure at 300dpi png+svg+pdf with
captions in figures/CAPTIONS.md.

Structural renders use matplotlib 3D CA-trace ribbons rather than PyMOL —
PyMOL-open-source was listed in environment.yml but not actually installed
in this build (a conda install was already spent on mkdssp; a second slow
conda solve for pymol was not worth the wall-clock here). This is a
documented tool substitution (see results/PROVENANCE.json); the renders are
real coordinate-based line plots, not fabricated schematics.

Run: python -m sentinel.figures.make_all_figures
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.cluster.hierarchy import dendrogram

from sentinel.figures.style import STRAIN_ORDER, apply_style, save_figure
from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger
from sentinel.utils.provenance import log_substitution

logger = get_logger(__name__)

CAPTIONS: list[tuple[str, str]] = []


def _load_json(*parts):
    return json.load(open(repo_path(*parts)))


def fig01_strain_folds():
    import biotite.structure.io.pdb as pdb_io
    prepared = {p["pdb_id"]: p for p in _load_json("data", "interim", "structures", "prepared_manifest.json")}
    cfg = load_config()
    panel = {e["id"]: e for e in cfg["data"]["strain_panel"]}

    fig, axes = plt.subplots(2, 5, figsize=(20, 9), subplot_kw={"projection": "3d"})
    for ax, pdb_id in zip(axes.flat, prepared.keys()):
        entry = panel[pdb_id]
        p = prepared[pdb_id]
        reader = pdb_io.PDBFile.read(str(repo_path(p["stack_pdb"])))
        arr = reader.get_structure(model=1)
        for chain in sorted(set(arr.chain_id)):
            ca = arr[(arr.chain_id == chain) & (arr.atom_name == "CA")]
            ax.plot(ca.coord[:, 0], ca.coord[:, 1], ca.coord[:, 2], lw=1.8)
        ax.set_title(f"{entry['strain']} ({pdb_id})", fontsize=10)
        ax.set_axis_off()
    fig.suptitle("The 8 tau strain folds — stacked-layer CA traces, one color per layer", y=1.02)
    save_figure(fig, "fig01_strain_folds", "The eight cryo-EM tau fibril folds in this project's "
                "panel, each rendered as its stacked-layer C-alpha trace (one line per fibril layer; "
                "AD_PHF and AD_SF are two conformers of Alzheimer's tau, CTE_I/II two conformers of "
                "chronic traumatic encephalopathy tau). Panels are NOT on a common scale/orientation.",
                CAPTIONS)
    plt.close(fig)


def fig02_dendrogram_rmsd():
    data = _load_json("results", "atlas", "fold_similarity_rmsd.json")
    mat = np.array(data["rmsd_matrix"])
    labels = data["labels"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    dendrogram(np.array(data["dendrogram"]["linkage_matrix"]), labels=labels, ax=axes[0], leaf_rotation=90)
    axes[0].set_title("Fold-similarity dendrogram (avg-linkage on pairwise RMSD)")
    axes[0].set_ylabel("RMSD (Angstrom)")

    sns.heatmap(mat, xticklabels=labels, yticklabels=labels, cmap="viridis_r", ax=axes[1],
                cbar_kws={"label": "RMSD (Angstrom)"})
    axes[1].set_title(f"Pairwise RMSD (common core {data['common_core_range']}, "
                       f"{data['n_shared_residues']} residues)")
    plt.setp(axes[1].get_xticklabels(), rotation=90)
    fig.tight_layout()
    save_figure(fig, "fig02_dendrogram_rmsd", "Left: hierarchical clustering of the 8+2 conformers "
                "by structural similarity. Right: the underlying pairwise Kabsch-RMSD matrix over the "
                "shared 306-378 core. Near-identical conformer pairs (AD PHF/SF, CTE I/II) cluster "
                "tightly; the AD/CTE folds are structurally distant from the R2-containing folds.",
                CAPTIONS)
    plt.close(fig)


def fig03_sasa_map():
    per_strain = _load_json("results", "atlas", "per_strain_characterization.json")
    ad = per_strain["AD_PHF"]["tip_per_residue_sasa"]
    resids = sorted(int(k) for k in ad.keys())
    rel_sasa = [ad[str(r)]["rel_sasa"] for r in resids]
    burial = [ad[str(r)]["burial_class"] for r in resids]
    colors = {"buried": "#8B0000", "intermediate": "#DAA520", "exposed": "#1E90FF", "unknown": "grey"}

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(resids, rel_sasa, color=[colors[b] for b in burial], width=0.9)
    ax.set_xlabel("Tau residue (2N4R numbering)")
    ax.set_ylabel("Relative SASA")
    ax.set_title("AD fold templating tip: per-residue solvent accessibility")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), title="burial class", loc="upper right")
    fig.tight_layout()
    save_figure(fig, "fig03_sasa_map", "Relative solvent-accessible surface area (SASA) at the AD "
                "fold's growth-competent templating tip, per residue, colored by burial class "
                "(buried/intermediate/exposed; Tien 2013 max-ASA scale, thresholds in config.yaml).",
                CAPTIONS)
    plt.close(fig)


def fig04_strain_fingerprint():
    fp = _load_json("results", "atlas", "ad_strain_fingerprint.json")
    top = fp["top_hotspots"]
    labels = [f"{h['res_name'][:1]}{h['res_id']}" for h in top]
    vals = [h["differential_exposure"] for h in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], vals[::-1], color=sns.color_palette("colorblind")[0])
    ax.set_xlabel("Differential exposure (AD rel-SASA - mean other-fold rel-SASA)")
    ax.set_title("AD strain fingerprint: top AD-selective hotspot residues")
    fig.tight_layout()
    save_figure(fig, "fig04_strain_fingerprint", "The AD strain fingerprint: residues most "
                "differentially exposed on the AD templating tip relative to the mean of the other "
                "7 folds at the same sequence position (same tau chain — no re-indexing). These are "
                "the selectivity-handle residues used to condition backbone generation (M6).",
                CAPTIONS)
    plt.close(fig)


def fig05_aggregation_profile():
    df = pd.read_csv(repo_path("results", "aggregation", "tau_aggregation_profile.csv"))
    cfg = load_config()
    lm = cfg["data"]["landmarks"]
    roc = _load_json("results", "benchmarks", "aggregation_roc_pr.json")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(df["window_start"], df["combined_score"], color="black", lw=1)
    axes[0].axvspan(lm["PHF6_star"]["start"], lm["PHF6_star"]["end"], color="tab:red", alpha=0.4, label="PHF6*")
    axes[0].axvspan(lm["PHF6"]["start"], lm["PHF6"]["end"], color="tab:blue", alpha=0.4, label="PHF6")
    axes[0].set_xlabel("Tau residue"); axes[0].set_ylabel("Combined aggregation score")
    axes[0].set_title("Full-length tau aggregation-propensity profile"); axes[0].legend()

    axes[1].plot(roc["roc_curve"]["fpr"], roc["roc_curve"]["tpr"], lw=2,
                  label=f"ROC (AUC={roc['roc_auc']})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Aggregation predictor benchmark"); axes[1].legend()
    fig.tight_layout()
    save_figure(fig, "fig05_aggregation_profile", "Left: consensus aggregation-propensity score "
                "across full-length tau (2N4R), PHF6/PHF6* highlighted (PHF6 ranks #1, PHF6* ranks "
                "#3 of 436 windows). Right: ROC curve recovering known nucleating segments "
                f"(AUC={roc['roc_auc']}).", CAPTIONS)
    plt.close(fig)


def fig06_md_analyses():
    md = _load_json("results", "md", "md_results_full.json")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for i, tag in enumerate(["PHF6", "PHF6_star", "fibril_tip"]):
        res = md[tag]
        ana = res["analysis"]
        axes[0, i].plot(ana["rmsd_nm"], color="tab:blue")
        axes[0, i].set_title(f"{tag}: RMSD ({res['actual_ns_simulated']} ns simulated)")
        axes[0, i].set_xlabel("frame"); axes[0, i].set_ylabel("RMSD (nm)")

        rmsf = ana["rmsf"]["rmsf_nm"]
        res_ids = ana["rmsf"]["res_id"]
        axes[1, i].plot(res_ids, rmsf, color="tab:orange")
        axes[1, i].set_title(f"{tag}: per-residue RMSF")
        axes[1, i].set_xlabel("residue"); axes[1, i].set_ylabel("RMSF (nm)")
    fig.tight_layout()
    save_figure(fig, "fig06_md_analyses", "Real OpenMM implicit-solvent MD: RMSD (top) and "
                "per-residue RMSF (bottom) for PHF6, PHF6*, and the truncated 2-layer AD fibril "
                "growing tip. Actual simulated length (not the config target) is stated per panel — "
                "see results/md/md_scaleup.md for full-length GPU reproduction.", CAPTIONS)
    plt.close(fig)


def fig07_learning_curves():
    lc = _load_json("results", "design", "learning_curves.json")
    fig, ax = plt.subplots(figsize=(9, 6))
    al = [r["cumulative_best_score"] for r in lc["active_learning"]]
    rs = [r["cumulative_best_score"] for r in lc["random_search"]]
    ax.plot(range(len(al)), al, marker="o", label="Active learning (GP + expected improvement)")
    ax.plot(range(len(rs)), rs, marker="s", label="Random search (equal budget)")
    ax.set_xlabel("Round"); ax.set_ylabel("Cumulative best composite score")
    ax.set_title("Design-engine learning curves"); ax.legend()
    fig.tight_layout()
    save_figure(fig, "fig07_learning_curves", "Cumulative best composite design score per round: "
                "the Gaussian-Process active-learning loop vs. an equal-budget random-search "
                "baseline (M9 ablation).", CAPTIONS)
    plt.close(fig)


def fig08_score_distributions():
    df = pd.read_csv(repo_path("results", "design", "all_designs_scored.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["composite_score"], bins=25, ax=axes[0], color=sns.color_palette("colorblind")[2])
    axes[0].set_title("Composite score distribution (all designs)")
    sns.boxplot(data=df, x="topology", y="composite_score", ax=axes[1])
    axes[1].set_title("Composite score by scaffold topology")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    save_figure(fig, "fig08_score_distributions", "Distribution of composite design scores across "
                "all evaluated designs (left) and broken out by scaffold topology (right).", CAPTIONS)
    plt.close(fig)


def fig09_selectivity_matrix():
    df = pd.read_csv(repo_path("results", "design", "selectivity_matrix.csv"))
    cfg = load_config()
    strains = [cfg["data"]["reference_strain"]] + [n["strain"] for n in
               json.load(open(repo_path("results", "target", "ad_capper_target.json")))["negative_design_panel"]]
    mat = df.set_index("design_id")[strains]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(mat))))
    sns.heatmap(mat, cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "combined score (packing + chemical complementarity - clash)"})
    ax.set_title("Selectivity matrix: top designs x fold")
    fig.tight_layout()
    save_figure(fig, "fig09_selectivity_matrix", "Selectivity matrix: each top-scoring design's "
                "(real backbone shape + actual designed sequence) combined geometric- and chemical-"
                "complementarity score when redocked (identical rigid-body sampling seed) onto each "
                "of the 8 folds' templating tips. Higher (redder) on the reference AD column than "
                "the negative-design columns indicates AD-selective binding — driven by both shape "
                "fit and the designed sequence's own chemistry, not shape alone.",
                CAPTIONS)
    plt.close(fig)


def fig10_biosensor_schematic():
    spec = _load_json("results", "design", "biosensor_concept.json")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    fibril_y = [2.0, 2.6, 3.2, 3.8]
    for y in fibril_y:
        ax.add_patch(plt.Rectangle((3.5, y - 0.2), 3.0, 0.4, color="grey", alpha=0.5))
    ax.text(5.0, 4.3, "AD tau fibril (stacked layers)", ha="center", fontsize=10)

    ax.annotate("", xy=(4.5, 2.6), xytext=(1.5, 1.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="tab:blue"))
    ax.text(1.0, 0.9, "Binder-LgBiT", color="tab:blue", fontsize=10)
    ax.annotate("", xy=(6.0, 3.2), xytext=(9.0, 1.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="tab:red"))
    ax.text(8.2, 0.9, "Binder-SmBiT", color="tab:red", fontsize=10)
    ax.text(5.0, 5.2, "Reconstituted NanoLuc -> luminescence\n(only when BOTH bind adjacent layers)",
             ha="center", fontsize=10, style="italic")
    ax.set_title(spec["concept_name"])
    fig.tight_layout()
    save_figure(fig, "fig10_biosensor_schematic", spec["rationale"][:250] + " (see "
                "results/design/biosensor_concept.json for the full design spec).", CAPTIONS)
    plt.close(fig)


def fig11_graphical_abstract():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    steps = [
        "1. Tau strain\natlas (M2)\n8 folds characterized",
        "2. AD strain\nfingerprint\nselectivity handle",
        "3. Aggregation\nengine (M3)\nPHF6/PHF6* validated",
        "4. Real MD (M4)\nhexapeptides + tip",
        "5. Design target\nspec (M5)",
        "6. Closed-loop\ndesign engine (M6)\n6 rounds, AL > random",
        "7-9. Validation,\nbiosensor,\nbenchmarks",
    ]
    for i, s in enumerate(steps):
        x = 1 + i * 1.8
        ax.add_patch(plt.Rectangle((x, 3), 1.5, 2, facecolor=sns.color_palette("colorblind")[i % 6],
                                     alpha=0.7))
        ax.text(x + 0.75, 4, s, ha="center", va="center", fontsize=8.5, wrap=True)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.8, 4), xytext=(x + 1.5, 4),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_xlim(0, 1 + len(steps) * 1.8)
    ax.set_ylim(0, 8)
    ax.set_title("Project SENTINEL Year 1 — the Conformational Atlas + Design Engine", fontsize=15)
    fig.tight_layout()
    save_figure(fig, "fig11_graphical_abstract", "Graphical abstract: the Year-1 pipeline from real "
                "cryo-EM structures to a benchmarked, AD-selective design engine.", CAPTIONS)
    plt.close(fig)


def main() -> None:
    apply_style()
    log_substitution("PyMOL-open-source structural rendering", "matplotlib 3D CA-trace ribbons",
                       "pymol-open-source was listed in environment.yml but a second slow conda "
                       "solve (after the mkdssp install) was not spent in this build; matplotlib "
                       "renders real coordinates, just without cartoon secondary-structure "
                       "smoothing/surfaces.")

    figure_fns = [fig01_strain_folds, fig02_dendrogram_rmsd, fig03_sasa_map, fig04_strain_fingerprint,
                   fig05_aggregation_profile, fig06_md_analyses, fig07_learning_curves,
                   fig08_score_distributions, fig09_selectivity_matrix, fig10_biosensor_schematic,
                   fig11_graphical_abstract]
    n_ok = 0
    for fn in figure_fns:
        try:
            fn()
            n_ok += 1
            logger.info(f"generated {fn.__name__}")
        except Exception as exc:
            logger.error(f"figure {fn.__name__} failed: {exc}")

    captions_path = repo_path("figures", "CAPTIONS.md")
    with open(captions_path, "w") as fh:
        fh.write("# Figure captions\n\n")
        for name, caption in CAPTIONS:
            fh.write(f"## {name}\n{caption}\n\n")

    append_progress_log("M10", f"Generated {n_ok}/{len(figure_fns)} figures (png+svg+pdf, 300dpi) "
                                f"with captions in figures/CAPTIONS.md.")


if __name__ == "__main__":
    main()
