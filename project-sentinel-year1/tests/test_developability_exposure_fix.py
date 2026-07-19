"""Regression test for a real bug found during the M6a quality upgrade: the
developability filter originally flagged ANY high-scoring aggregation window
as a liability, including ones that are actually buried inside a normal,
well-packed hydrophobic core (which is EXPECTED to score high on beta-
propensity/hydrophobicity terms — that's what makes it a core). Measured on
a real run: 522/640 designs failed purely on this basis, mean percentile 84%
against tau's own (intrinsically disordered, low-baseline) window
distribution — even designs built on real, hyperstable, evolved scaffolds
failed. Fixed by only counting windows that are also solvent-EXPOSED
(mean relative SASA above the config exposure threshold) as real
developability liabilities."""
import json

import pytest


@pytest.fixture(scope="module")
def tau_reference(tau_sequence):
    from sentinel.aggregation.scorer import compute_profile, get_normalization_bounds
    profile = compute_profile(tau_sequence)
    scores = [r["combined_score"] for r in profile["records"]]
    bounds = get_normalization_bounds(profile)
    return scores, bounds


def test_buried_amyloidogenic_window_does_not_fail_when_structure_shown_buried(tau_reference, config):
    """The core scientific claim of the fix: a VQIVYK-like amyloidogenic
    stretch that is structurally BURIED (a real hydrophobic core) must NOT
    trigger the developability failure once real structural context is
    supplied, even though the exact same sequence WOULD fail without that
    context (the pre-fix, sequence-only behavior)."""
    from sentinel.design.selectivity import developability_filter
    tau_scores, tau_bounds = tau_reference

    seq = "AAAAAAAVQIVYKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    buried_threshold = config["atlas"]["buried_rel_sasa_threshold"]
    all_buried_sasa = {i + 1: buried_threshold / 2 for i in range(len(seq))}

    without_structure = developability_filter(seq, tau_scores, tau_bounds)
    with_all_buried = developability_filter(seq, tau_scores, tau_bounds, all_buried_sasa)

    assert without_structure["developability_passed"] is False
    assert with_all_buried["developability_passed"] is True
    assert with_all_buried["scoring_basis"] == "exposed_windows_only"


def test_exposed_amyloidogenic_window_still_fails_with_structure(tau_reference, config):
    """The other half of the claim: an amyloidogenic stretch that IS
    genuinely solvent-exposed must still fail — the fix narrows the check to
    real liabilities, it does not neuter it."""
    from sentinel.design.selectivity import developability_filter
    tau_scores, tau_bounds = tau_reference

    seq = "AAAAAAAVQIVYKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    exposed_threshold = config["atlas"]["exposed_rel_sasa_threshold"]
    all_exposed_sasa = {i + 1: exposed_threshold + 0.2 for i in range(len(seq))}

    with_all_exposed = developability_filter(seq, tau_scores, tau_bounds, all_exposed_sasa)
    assert with_all_exposed["developability_passed"] is False


def test_no_structure_falls_back_to_original_whole_sequence_behavior(tau_reference):
    from sentinel.design.selectivity import developability_filter
    tau_scores, tau_bounds = tau_reference
    bland_seq = "SEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEKDQNSEK"
    d = developability_filter(bland_seq, tau_scores, tau_bounds)
    assert d["scoring_basis"] == "whole_sequence_no_structural_context"
