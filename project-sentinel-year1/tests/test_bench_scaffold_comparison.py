"""Test for the real-scaffold-vs-idealized-topology quality benchmark — the
direct, quantitative test of whether the M6a scaffold-library upgrade
actually produces better-packed designs than idealized cylinders."""
from sentinel.bench.run_benchmarks import real_scaffold_vs_idealized_comparison


def _design(topology, r):
    return {"params": {"topology": topology}, "hydrophobic_core_consistency": {"pearson_r": r}}


def test_correctly_splits_real_vs_idealized_by_name_prefix():
    al_result = {"all_designs": [
        _design("scaffold_protA_bdomain", -0.4),
        _design("scaffold_villin", -0.5),
        _design("scaffold_darpin", -0.3),
        _design("helix_hairpin", -0.1),
        _design("long_helix_capper", 0.05),
        _design("three_helix_bundle", 0.0),
    ]}
    result = real_scaffold_vs_idealized_comparison(al_result)
    assert result["n_real_scaffold"] == 3
    assert result["n_idealized"] == 3
    assert result["real_scaffolds_better_packed"] is True
    assert result["mean_pearson_r_real_scaffold"] < result["mean_pearson_r_idealized"]


def test_handles_missing_pearson_r_gracefully():
    al_result = {"all_designs": [
        _design("scaffold_protA_bdomain", None),
        _design("helix_hairpin", None),
    ]}
    result = real_scaffold_vs_idealized_comparison(al_result)
    assert "note" in result
    assert result["n_real_scaffold"] == 0 and result["n_idealized"] == 0


def test_insufficient_data_reports_note_not_crash():
    al_result = {"all_designs": [_design("scaffold_villin", -0.2)]}
    result = real_scaffold_vs_idealized_comparison(al_result)
    assert "note" in result
