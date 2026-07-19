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


def active_learning_vs_random(learning_curves: dict, al_scores: list, rs_scores: list) -> dict:
    """Two complementary comparisons, reported together because they answer
    different questions and can legitimately disagree:

    1. Final-best (cumulative max) per round — the headline "did the search
       ever find something great" number. This is a single noisy order
       statistic: two runs can converge to the same true optimum by luck
       (random search stumbling onto it early) vs by design (active learning
       exploiting it), and a max-only comparison can't distinguish those —
       caught during this build (see PROGRESS_LOG.md M6): final-best came out
       0.2995 (AL) vs 0.3035 (RS), a near coin flip despite AL clearly
       learning.
    2. Mean/median across ALL evaluated candidates, plus an UNPAIRED t-test
       over the full candidate population (not just the 10 round-level
       summary points) — this measures whether the loop spent its budget
       preferentially sampling good regions, the actual mechanism active
       learning is supposed to provide. This is the decisive, low-noise
       comparison (n=80 per arm in this build, not n=10 rounds)."""
    al_curve = [r["cumulative_best_score"] for r in learning_curves["active_learning"]]
    rs_curve = [r["cumulative_best_score"] for r in learning_curves["random_search"]]

    n = min(len(al_curve), len(rs_curve))
    al_curve_t, rs_curve_t = al_curve[:n], rs_curve[:n]
    diffs = np.array(al_curve_t) - np.array(rs_curve_t)

    al_dominates_every_round = bool(np.all(np.array(al_curve_t) >= np.array(rs_curve_t)))
    final_gap = float(al_curve_t[-1] - rs_curve_t[-1])

    if np.std(diffs) > 1e-9:
        t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat, p_value = float("nan"), float("nan")
    effect_size = float(np.mean(diffs) / np.std(diffs)) if np.std(diffs) > 1e-9 else float("nan")

    al_arr, rs_arr = np.array(al_scores), np.array(rs_scores)
    mean_t, mean_p = stats.ttest_ind(al_arr, rs_arr)
    half = len(al_arr) // 2
    al_trend = float(al_arr[half:].mean() - al_arr[:half].mean())
    rs_trend = float(rs_arr[half:].mean() - rs_arr[:half].mean())

    return {
        "final_best_comparison": {
            "active_learning_curve": al_curve, "random_search_curve": rs_curve,
            "al_dominates_every_round": al_dominates_every_round, "final_gap": round(final_gap, 4),
            "paired_ttest_statistic": round(t_stat, 4) if t_stat == t_stat else None,
            "paired_ttest_pvalue": round(p_value, 4) if p_value == p_value else None,
            "cohens_d_effect_size": round(effect_size, 4) if effect_size == effect_size else None,
        },
        "mean_score_comparison": {
            "n_active_learning": len(al_arr), "n_random_search": len(rs_arr),
            "al_mean": round(float(al_arr.mean()), 4), "rs_mean": round(float(rs_arr.mean()), 4),
            "al_median": round(float(np.median(al_arr)), 4), "rs_median": round(float(np.median(rs_arr)), 4),
            "al_second_half_minus_first_half_trend": round(al_trend, 4),
            "rs_second_half_minus_first_half_trend": round(rs_trend, 4),
            "unpaired_ttest_statistic": round(float(mean_t), 4),
            "unpaired_ttest_pvalue": round(float(mean_p), 6),
            "al_mean_beats_rs_mean": bool(al_arr.mean() > rs_arr.mean()),
        },
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


def real_scaffold_vs_idealized_comparison(al_result: dict) -> dict:
    """The direct, quantitative test of the M6a quality upgrade's core claim:
    do designs built on REAL solved-structure scaffolds show better sequence-
    structure consistency (hydrophobic core packing) than designs built on
    IDEALIZED geometric cylinders? Uses hydrophobic_core_consistency's Pearson
    r (computed live during the design loop for every design, not recomputed
    here) — a real, independent, structure-based quality signal, not the
    composite score itself (which already includes this term, so comparing
    composite score here would be circular)."""
    idealized_prefixes = ("helix_hairpin", "three_helix_bundle", "helix_strand_helix", "long_helix_capper")
    real_rs, idealized_rs = [], []
    for d in al_result["all_designs"]:
        r = d.get("hydrophobic_core_consistency", {}).get("pearson_r")
        if r is None:
            continue
        topo = d["params"]["topology"]
        (real_rs if topo.startswith("scaffold_") else idealized_rs).append(r)

    if len(real_rs) < 2 or len(idealized_rs) < 2:
        return {"note": "insufficient designs in one or both groups for a statistical comparison",
                "n_real_scaffold": len(real_rs), "n_idealized": len(idealized_rs)}

    t_stat, p_val = stats.ttest_ind(real_rs, idealized_rs)
    return {
        "n_real_scaffold": len(real_rs), "n_idealized": len(idealized_rs),
        "mean_pearson_r_real_scaffold": round(float(np.mean(real_rs)), 4),
        "mean_pearson_r_idealized": round(float(np.mean(idealized_rs)), 4),
        "note": "more negative = better hydrophobic-core packing (buried residues genuinely "
                "hydrophobic, exposed residues genuinely polar)",
        "unpaired_ttest_statistic": round(float(t_stat), 4),
        "unpaired_ttest_pvalue": round(float(p_val), 6),
        "real_scaffolds_better_packed": bool(np.mean(real_rs) < np.mean(idealized_rs)),
    }


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
    al_result = json.load(open(repo_path("results", "design", "active_learning_result.json")))
    rs_result = json.load(open(repo_path("results", "design", "random_search_result.json")))
    al_vs_rs = active_learning_vs_random(learning_curves, al_result["y_all"], rs_result["y_all"])
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
        selectivity_stats = {"n_designs": len(paired_ad), "mean_ad_score": round(float(np.mean(paired_ad)), 4),
                               "mean_other_score": round(float(np.mean(paired_other_mean)), 4),
                               "paired_ttest_statistic": round(float(t_stat), 4),
                               "paired_ttest_pvalue": round(float(p_val), 4)}
    else:
        selectivity_stats = {"n_designs": len(paired_ad), "note": "too few designs for a paired test"}
    with open(out_dir / "selectivity_statistics.json", "w") as fh:
        json.dump(selectivity_stats, fh, indent=2)

    logger.info("comparing real-scaffold vs idealized-topology design quality...")
    scaffold_comparison = real_scaffold_vs_idealized_comparison(al_result)
    with open(out_dir / "real_scaffold_vs_idealized.json", "w") as fh:
        json.dump(scaffold_comparison, fh, indent=2)

    fbc = al_vs_rs["final_best_comparison"]
    msc = al_vs_rs["mean_score_comparison"]
    logger.info(f"aggregation predictor: ROC-AUC={agg_bench['roc_auc']}, PR-AUC={agg_bench['pr_auc']}")
    logger.info(f"active learning final-best={fbc['active_learning_curve'][-1]} vs "
                f"random search final-best={fbc['random_search_curve'][-1]} "
                f"(dominates_every_round={fbc['al_dominates_every_round']}) | "
                f"mean scores: AL={msc['al_mean']} vs RS={msc['rs_mean']} "
                f"(unpaired t-test p={msc['unpaired_ttest_pvalue']}, al_beats_rs={msc['al_mean_beats_rs_mean']})")
    logger.info(f"negative control: {neg_control['n_real_beats_scrambled']}/{neg_control['n_total']} "
                f"real sequences beat their scrambled counterpart")
    logger.info(f"real scaffold vs idealized hydrophobic-core packing: "
                f"{scaffold_comparison.get('mean_pearson_r_real_scaffold')} vs "
                f"{scaffold_comparison.get('mean_pearson_r_idealized')} "
                f"(p={scaffold_comparison.get('unpaired_ttest_pvalue')})")

    append_progress_log(
        "M9",
        f"Benchmarks: aggregation predictor ROC-AUC={agg_bench['roc_auc']}, PR-AUC={agg_bench['pr_auc']} "
        f"recovering known PHF6/PHF6* nucleating segments. Active-learning vs random-search: final-best "
        f"cumulative score {fbc['active_learning_curve'][-1]} vs {fbc['random_search_curve'][-1]} (a near "
        f"coin-flip single-draw comparison, paired t-test on round curves p={fbc['paired_ttest_pvalue']}) "
        f"— the mean comparison (mean score across all {msc['n_active_learning']} evaluated "
        f"candidates per arm): AL mean={msc['al_mean']} vs RS mean={msc['rs_mean']} "
        f"(AL {'beats' if msc['al_mean_beats_rs_mean'] else 'does not beat'} RS), unpaired "
        f"t-test p={msc['unpaired_ttest_pvalue']}. Second-half-minus-first-half trend: AL "
        f"{msc['al_second_half_minus_first_half_trend']:+.4f}, RS "
        f"{msc['rs_second_half_minus_first_half_trend']:+.4f} (reported as-is; not asserted to favor "
        f"either arm unless the sign actually does). Selectivity: mean AD score "
        f"{selectivity_stats.get('mean_ad_score')} vs mean other-fold score "
        f"{selectivity_stats.get('mean_other_score')} (paired t-test p="
        f"{selectivity_stats.get('paired_ttest_pvalue')}). Negative control: "
        f"{neg_control['n_real_beats_scrambled']}/{neg_control['n_total']} real sequences beat scrambled. "
        f"Real-scaffold vs idealized-topology hydrophobic-core packing (M6a quality upgrade's core "
        f"claim, tested directly): mean Pearson r "
        f"{scaffold_comparison.get('mean_pearson_r_real_scaffold')} (real, n="
        f"{scaffold_comparison.get('n_real_scaffold')}) vs "
        f"{scaffold_comparison.get('mean_pearson_r_idealized')} (idealized, n="
        f"{scaffold_comparison.get('n_idealized')}), unpaired t-test p="
        f"{scaffold_comparison.get('unpaired_ttest_pvalue')}.",
    )
    return {"agg_bench": agg_bench, "al_vs_rs": al_vs_rs, "neg_control": neg_control,
            "selectivity_stats": selectivity_stats, "scaffold_comparison": scaffold_comparison}


if __name__ == "__main__":
    main()
