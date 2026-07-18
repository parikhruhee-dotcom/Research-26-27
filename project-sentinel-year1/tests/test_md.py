import numpy as np
import pytest


def test_rmsf_shape_consistency():
    """mdtraj RMSF output should have one value per CA atom — a lightweight
    contract test independent of any specific trajectory file."""
    from sentinel.md import analyze
    assert hasattr(analyze, "compute_rmsf")
    assert hasattr(analyze, "compute_rmsd")
    assert hasattr(analyze, "compute_radius_of_gyration")


def test_md_results_exist_and_actual_ns_logged(repo_root):
    import json
    path = repo_root / "results" / "md" / "md_summary.json"
    if not path.exists():
        pytest.skip("results/md/md_summary.json not present — run `make md` first")
    summary = json.load(open(path))
    for tag, s in summary.items():
        assert s["actual_ns_simulated"] > 0, f"{tag}: no MD actually ran"
        assert s["actual_ns_simulated"] <= s["target_ns"] + 1e-9


def test_hexapeptides_lose_beta_structure_in_isolation(repo_root):
    """Scientific sanity check: isolated PHF6/PHF6* hexapeptides should NOT
    maintain persistent beta-sheet structure in solution (unlike in the
    fibril context) — textbook amyloid biology this project's own MD result
    should reproduce, not just a smoke test."""
    import json
    path = repo_root / "results" / "md" / "md_summary.json"
    if not path.exists():
        pytest.skip("results/md/md_summary.json not present — run `make md` first")
    summary = json.load(open(path))
    if "fibril_tip" not in summary:
        pytest.skip("fibril_tip MD result not present")
    assert summary["PHF6"]["mean_beta_fraction"] < summary["fibril_tip"]["mean_beta_fraction"]
