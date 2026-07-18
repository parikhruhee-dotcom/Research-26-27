import numpy as np
import pytest

from sentinel.aggregation.scorer import compute_profile, zipper_compatibility_score


def test_zipper_compatibility_perfect_match():
    score = zipper_compatibility_score("VQIVYK", ["VQIVYK", "VQIINK"])
    assert score == pytest.approx(1.0)


def test_zipper_compatibility_mismatch_lower():
    match_score = zipper_compatibility_score("VQIVYK", ["VQIVYK"])
    mismatch_score = zipper_compatibility_score("GGGGGG", ["VQIVYK"])
    assert mismatch_score < match_score


def test_compute_profile_scores_normalized_0_1(tau_sequence):
    profile = compute_profile(tau_sequence)
    scores = [r["combined_score"] for r in profile["records"]]
    assert min(scores) == pytest.approx(0.0, abs=1e-6)
    assert max(scores) == pytest.approx(1.0, abs=1e-6)


def test_compute_profile_window_count(tau_sequence):
    profile = compute_profile(tau_sequence)
    expected_windows = len(tau_sequence) - profile["window_size"] + 1
    assert len(profile["records"]) == expected_windows


def test_compute_profile_deterministic(tau_sequence):
    p1 = compute_profile(tau_sequence)
    p2 = compute_profile(tau_sequence)
    scores1 = [r["combined_score"] for r in p1["records"]]
    scores2 = [r["combined_score"] for r in p2["records"]]
    assert np.allclose(scores1, scores2)
