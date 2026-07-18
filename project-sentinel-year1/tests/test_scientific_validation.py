"""The 5 required scientific validation tests (brief M11). These gate the
project's correctness — if any of these fail, the underlying method is wrong
and must be fixed, not the test."""
import json

import numpy as np
import pytest


def test_phf6_ranked_top(tau_sequence, config):
    from sentinel.aggregation.scorer import compute_profile
    lm = config["data"]["landmarks"]
    profile = compute_profile(tau_sequence)
    by_start = {r["window_start"]: r["rank"] for r in profile["records"]}

    top_n = config["aggregation"]["validation"]["top_n_required"]
    assert by_start[lm["PHF6"]["start"]] <= top_n, "PHF6 must rank in the top-N aggregation windows"
    assert by_start[lm["PHF6_star"]["start"]] <= top_n, "PHF6* must rank in the top-N aggregation windows"


def test_vqiink_buries_more_than_vqivyk(repo_root):
    """Reproduces the literature finding (Seidler 2018 Nat Chem) that VQIINK
    forms a more dominant steric zipper than VQIVYK, using the real
    hexapeptide microcrystal structures (2ON9, 5V5C)."""
    from sentinel.atlas.zipper import zipper_crystal_comparison
    result = zipper_crystal_comparison()
    assert result["VQIINK"]["total_buried_A2"] > result["VQIVYK"]["total_buried_A2"]
    # the literature claim is specifically ~2x; assert the right order of magnitude, not
    # an exact hardcoded ratio (that would be over-fit to one run)
    assert 1.3 <= result["VQIINK_over_VQIVYK_ratio"] <= 3.0


def test_ad_selective_designs_prefer_ad_tip(repo_root):
    import csv
    path = repo_root / "results" / "design" / "selectivity_matrix.csv"
    if not path.exists():
        pytest.skip("results/design/selectivity_matrix.csv not present — run `make design` first")
    rows = list(csv.DictReader(open(path)))
    if not rows:
        pytest.skip("no backbones scored yet")

    target_spec_path = repo_root / "results" / "target" / "ad_capper_target.json"
    target_spec = json.load(open(target_spec_path))
    ref = target_spec["reference_strain"]
    others = [n["strain"] for n in target_spec["negative_design_panel"]]

    ad_scores = np.array([float(r[ref]) for r in rows])
    other_scores = np.array([np.mean([float(r[s]) for s in others]) for r in rows])
    # paired: on average, AD-tip score should exceed the mean other-tip score
    assert ad_scores.mean() > other_scores.mean()


def test_active_learning_beats_random(repo_root):
    path = repo_root / "results" / "design" / "learning_curves.json"
    if not path.exists():
        pytest.skip("results/design/learning_curves.json not present — run `make design` first")
    curves = json.load(open(path))
    al_final = curves["active_learning"][-1]["cumulative_best_score"]
    rs_final = curves["random_search"][-1]["cumulative_best_score"]
    assert al_final >= rs_final, (
        f"active learning ({al_final}) should dominate an equal-budget random search "
        f"baseline ({rs_final}) by the final round"
    )


def test_determinism(tau_sequence):
    from sentinel.aggregation.scorer import compute_profile
    from sentinel.utils.seeds import set_global_seed

    set_global_seed(42)
    p1 = compute_profile(tau_sequence)
    set_global_seed(42)
    p2 = compute_profile(tau_sequence)
    scores1 = [r["combined_score"] for r in p1["records"]]
    scores2 = [r["combined_score"] for r in p2["records"]]
    assert scores1 == scores2

    from sentinel.design.geometry import build_backbone
    c1 = build_backbone("H" * 10)
    c2 = build_backbone("H" * 10)
    assert np.allclose(c1["CA"], c2["CA"])
