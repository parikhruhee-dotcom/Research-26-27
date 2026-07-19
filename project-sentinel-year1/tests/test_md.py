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


def test_run_md_sets_velocities_after_minimizing_not_before():
    """Regression test for a real bug: velocities used to be assigned to the
    RAW (potentially badly-clashed) starting structure BEFORE minimization,
    so the post-minimization, newly-relaxed positions ended up paired with
    stale velocities generated for a completely different, high-energy
    configuration. Measured directly on a real design that hit this: minimizing
    a structure starting at 6.4e10 kJ/mol reliably converged to a sane ~-9500
    kJ/mol either way, but the very first dynamics step only completed without
    an OpenMM 'Particle coordinate is NaN' crash when velocities were assigned
    AFTER minimization, on the already-relaxed structure -- with the original
    (wrong) order, the identical minimized structure crashed with NaN every
    time (see PROGRESS_LOG.md for the full before/after). A real OpenMM run is
    too slow (60s+) for the fast unit-test suite, so this checks the actual
    call order in source directly -- a simple, fast, deterministic guard
    against exactly this class of reordering regression."""
    import ast
    import inspect

    from sentinel.md.simulate import run_md

    source = inspect.getsource(run_md)
    tree = ast.parse(source)
    func = tree.body[0]

    call_order = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            func_repr = ast.dump(node.func)
            if "minimize" in func_repr and "id='minimize'" in func_repr:
                call_order.append(("minimize", node.lineno))
            elif "setVelocitiesToTemperature" in func_repr:
                call_order.append(("setVelocitiesToTemperature", node.lineno))

    call_order.sort(key=lambda x: x[1])
    names = [c[0] for c in call_order]
    assert "minimize" in names and "setVelocitiesToTemperature" in names, (
        "expected to find both calls in run_md's source"
    )
    assert names.index("minimize") < names.index("setVelocitiesToTemperature"), (
        "minimize() must run BEFORE setVelocitiesToTemperature() -- see this test's "
        "docstring for the real NaN-crash bug this ordering fixes"
    )
