import numpy as np
import pytest

from sentinel.aggregation.scorer import compute_profile, get_normalization_bounds
from sentinel.design.selectivity import developability_filter


@pytest.fixture(scope="module")
def tau_reference(tau_sequence):
    profile = compute_profile(tau_sequence)
    scores = [r["combined_score"] for r in profile["records"]]
    bounds = get_normalization_bounds(profile)
    return scores, bounds


def test_binder_own_max_score_is_not_trivially_one(tau_reference):
    """Regression test for a real bug found during the build: scoring a short
    binder sequence with its own independent min-max normalization always
    gives max score == 1.0 by construction, making developability filtering
    meaningless. Scoring on tau's external scale must NOT do this."""
    tau_scores, tau_bounds = tau_reference
    bland_seq = "SEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEK"
    d = developability_filter(bland_seq, tau_scores, tau_bounds)
    assert d["max_own_aggregation_score"] != pytest.approx(1.0)


def test_developability_flags_known_amyloid_motif(tau_reference):
    tau_scores, tau_bounds = tau_reference
    amyloidogenic_seq = "AAAAAAAVQIVYKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    d = developability_filter(amyloidogenic_seq, tau_scores, tau_bounds)
    assert d["developability_passed"] is False
    assert d["percentile_vs_tau_windows"] > 90


def test_developability_passes_bland_sequence(tau_reference):
    tau_scores, tau_bounds = tau_reference
    bland_seq = "SEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEK"
    d = developability_filter(bland_seq, tau_scores, tau_bounds)
    assert d["developability_passed"] is True


def test_cysteine_check():
    tau_scores = [0.5] * 10
    tau_bounds = {"__combined__": (0.0, 1.0)}
    from sentinel.utils.config import load_config
    cfg = load_config()
    seq_with_cys = "ACDEFGHIKLMNPQRSTVWY"
    d = developability_filter(seq_with_cys, tau_scores, tau_bounds)
    assert d["n_cysteines"] == 1
    if cfg["design"]["developability"]["disallow_free_cysteines"]:
        assert d["cysteine_check_passed"] is False
