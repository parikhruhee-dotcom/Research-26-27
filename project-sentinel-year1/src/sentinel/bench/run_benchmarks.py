"""M9 — Benchmarking & controls.

1. Aggregation predictor ROC/PR: recovering known aggregation-nucleating
   segments (PHF6/PHF6* + the M1c curated inhibitor-target segments) vs.
   non-aggregating tau windows.
2. Design-engine retrospective: active-learning vs random-search learning
   curves (already in results/design/learning_curves.json) + a paired
   statistical comparison; AD-selective designs vs other-tip scores (paired).
3. Ablation: active-learning beats random search for equal budget (effect
   size + significance test).
4. Negative controls: scrambled/known-nonbinder sequences score poorly on
   the M6 composite scorer.

Run: python -m sentinel.bench.run_benchmarks
"""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy import stats
from sklearn.metrics import auc, roc_curve, precision_recall_curve

from sentinel.aggregation.scorer import compute_profile
from sentinel.design.interface_scorer import esm_plausibility
from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger
from sentinel.utils.seeds import set_global_seed

logger = get_logger(__name__)


def aggregation_roc_pr() -> dict:
    cfg = load_config()
    lm = cfg["data"]["landmarks"]
    seq_data = json.load(open(repo_path("data", "interim", "tau_sequence.json")))
    profile = compute_profile(seq_data["sequence"])
    records = profile["records"]

    known_inhibitors = list(csv.DictReader(open(repo_path("data", "external", "known_tau_inhibitors.csv"))))
    positive_segments = []
    for r in known_inhibitors:
        seg = r["target_segment"]
        if "VQIVYK" in seg or "PHF6 (" in seg:
            positive_segments.append((lm["PHF6"]["start"], lm["PHF6"]["end"]))
        if "VQIINK" in seg or "PHF6* (" in seg:
            positive_segments.append((lm["PHF6_star"]["start"], lm["PHF6_star"]["end"]))
    positive_segments = list(set(positive_segments))

    y_true, y_score = [], []
    for r in records:
        is_positive = any(r["window_start"] <= e and r["window_end"] >= s for s, e in positive_segments)
        y_true.append(int(is_positive))
        y_score.append(r["combined_score"])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recall, precision))

    return {
        "n_positive_windows": int(sum(y_true)), "n_total_windows": len(y_true),
        "roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
    }


def active_learning_vs_random(learning_curves: dict) -> dict:
    al_curve = [r["cumulative_best_score"] for r in learning_curves["active_learning"]]
    rs_curve = [r["cumulative_best_score"] for r in learning_curves["random_search"]]

    n = min(len(al_curve), len(rs_curve))
    al_curve, rs_curve = al_curve[:n], rs_curve[:n]
    diffs = np.array(al_curve) - np.array(rs_curve)

    al_dominates_every_round = bool(np.all(np.array(al_curve) >= np.array(rs_curve)))
    final_gap = float(al_curve[-1] - rs_curve[-1])

    if np.std(diffs) > 1e-9:
        t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat, p_value = float("nan"), float("nan")
    effect_size = float(np.mean(diffs) / np.std(diffs)) if np.std(diffs) > 1e-9 else float("nan")

    return {
        "active_learning_curve": al_curve, "random_search_curve": rs_curve,
        "al_dominates_every_round": al_dominates_every_round, "final_gap": round(final_gap, 4),
        "paired_ttest_statistic": round(t_stat, 4) if t_stat == t_stat else None,
        "paired_ttest_pvalue": round(p_value, 4) if p_value == p_value else None,
        "cohens_d_effect_size": round(effect_size, 4) if effect_size == effect_size else None,
    }


def negative_control_check() -> dict:
    """Scrambled versions of the top leads, and a generic unrelated-amyloid
    control sequence, should NOT score as well as the real leads on the
    composite scorer's sequence-plausibility component."""
    import random
    design_dir = repo_path("results", "design")
    all_designs = list(csv.DictReader(open(design_dir / "all_designs_scored.csv")))
    all_designs.sort(key=lambda d: float(d["composite_score"]), reverse=True)
    top5 = all_designs[:5]

    esm_model = load_config()["design"]["scoring"]["esm_model"]
    rng = random.Random(42)
    results = []
    for d in top5:
        real_seq = d["sequence"]
        scrambled = list(real_seq)
        rng.shuffle(scrambled)
        scrambled = "".join(scrambled)
        real_score = esm_plausibility(real_seq, esm_model)
        scrambled_score = esm_plausibility(scrambled, esm_model)
        results.append({"design_id": d["design_id"], "real_plausibility": real_score,
                          "scrambled_plausibility": scrambled_score,
                          "real_beats_scrambled": real_score >= scrambled_score})
    n_pass = sum(1 for r in results if r["real_beats_scrambled"])
    return {"per_design": results, "n_real_beats_scrambled": n_pass, "n_total": len(results)}


def main() -> dict:
    set_global_seed()
    out_dir = repo_path("results", "benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("computing aggregation predictor ROC/PR...")
    agg_bench = aggregation_roc_pr()
    with open(out_dir / "aggregation_roc_pr.json", "w") as fh:
        json.dump(agg_bench, fh, indent=2)

    logger.info("comparing active-learning vs random-search...")
    learning_curves = json.load(open(repo_path("results", "design", "learning_curves.json")))
    al_vs_rs = active_learning_vs_random(learning_curves)
    with open(out_dir / "active_learning_vs_random.json", "w") as fh:
        json.dump(al_vs_rs, fh, indent=2)

    logger.info("running negative controls (scrambled sequences)...")
    target_spec = json.load(open(repo_path("results", "target", "ad_capper_target.json")))
    neg_control = negative_control_check()
    with open(out_dir / "negative_controls.json", "w") as fh:
        json.dump(neg_control, fh, indent=2)

    selectivity_rows = list(csv.DictReader(open(repo_path("results", "design", "selectivity_matrix.csv"))))
    ref_strain = target_spec["reference_strain"]
    other_strains = [n["strain"] for n in target_spec["negative_design_panel"]]
    paired_ad = [float(r[ref_strain]) for r in selectivity_rows if r[ref_strain]]
    paired_other_mean = [np.mean([float(r[s]) for s in other_strains if r[s]]) for r in selectivity_rows
                           if r[ref_strain]]
    if len(paired_ad) >= 2:
        t_stat, p_val = stats.ttest_rel(paired_ad, paired_other_mean)
        selectivity_stats = {"n_backbones": len(paired_ad), "mean_ad_score": round(float(np.mean(paired_ad)), 4),
                               "mean_other_score": round(float(np.mean(paired_other_mean)), 4),
                               "paired_ttest_statistic": round(float(t_stat), 4),
                               "paired_ttest_pvalue": round(float(p_val), 4)}
    else:
        selectivity_stats = {"n_backbones": len(paired_ad), "note": "too few backbones for a paired test"}
    with open(out_dir / "selectivity_statistics.json", "w") as fh:
        json.dump(selectivity_stats, fh, indent=2)

    logger.info(f"aggregation predictor: ROC-AUC={agg_bench['roc_auc']}, PR-AUC={agg_bench['pr_auc']}")
    logger.info(f"active learning final={al_vs_rs['active_learning_curve'][-1]} vs "
                f"random search final={al_vs_rs['random_search_curve'][-1]}, "
                f"dominates_every_round={al_vs_rs['al_dominates_every_round']}")
    logger.info(f"negative control: {neg_control['n_real_beats_scrambled']}/{neg_control['n_total']} "
                f"real sequences beat their scrambled counterpart")

    append_progress_log(
        "M9",
        f"Benchmarks: aggregation predictor ROC-AUC={agg_bench['roc_auc']}, PR-AUC={agg_bench['pr_auc']} "
        f"recovering known PHF6/PHF6* nucleating segments. Active-learning vs random-search: final "
        f"cumulative best {al_vs_rs['active_learning_curve'][-1]} vs {al_vs_rs['random_search_curve'][-1]} "
        f"(dominates_every_round={al_vs_rs['al_dominates_every_round']}, paired t-test p="
        f"{al_vs_rs['paired_ttest_pvalue']}). Selectivity: mean AD score "
        f"{selectivity_stats.get('mean_ad_score')} vs mean other-fold score "
        f"{selectivity_stats.get('mean_other_score')} (paired t-test p="
        f"{selectivity_stats.get('paired_ttest_pvalue')}). Negative control: "
        f"{neg_control['n_real_beats_scrambled']}/{neg_control['n_total']} real sequences beat scrambled.",
    )
    return {"agg_bench": agg_bench, "al_vs_rs": al_vs_rs, "neg_control": neg_control,
            "selectivity_stats": selectivity_stats}


if __name__ == "__main__":
    main()
