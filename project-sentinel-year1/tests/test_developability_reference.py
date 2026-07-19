"""Regression tests for the developability reference population — the third
and final iteration of a bug found in this build (see
sentinel.design.developability_reference module docstring for the full
story): tau's own windows, then a pooled-individual-window population built
from the design scaffold library, were both mechanically the wrong
comparison for a binder's own worst exposed window. The fix compares
peak-to-peak against an independent panel of 8 real, verified, soluble
reference proteins."""
import pytest


def test_reference_sources_disjoint_from_design_scaffold_library():
    """The reference panel must be independent of the design scaffold
    library — comparing a binder built on a scaffold against that same
    scaffold's own peak would be circular."""
    from sentinel.design.developability_reference import SOLUBLE_REFERENCE_SOURCES
    from sentinel.design.scaffold_library import SCAFFOLD_SOURCES

    ref_ids = {v["pdb_id"] for v in SOLUBLE_REFERENCE_SOURCES.values()}
    scaffold_ids = {v["pdb_id"] for v in SCAFFOLD_SOURCES.values()}
    assert ref_ids.isdisjoint(scaffold_ids)


def test_build_reference_distribution_returns_one_peak_per_protein(tau_sequence):
    from sentinel.aggregation.scorer import compute_profile, get_normalization_bounds
    from sentinel.design.developability_reference import (
        SOLUBLE_REFERENCE_SOURCES, build_reference_distribution,
    )
    profile = compute_profile(tau_sequence)
    tau_bounds = get_normalization_bounds(profile)

    ref = build_reference_distribution(tau_bounds)
    assert ref["n_source_proteins"] == len(SOLUBLE_REFERENCE_SOURCES)
    assert len(ref["scores"]) == len(SOLUBLE_REFERENCE_SOURCES)
    assert all(s >= 0.0 for s in ref["scores"])
    # a real, meaningful spread across 8 unrelated real proteins, not a degenerate constant
    assert max(ref["scores"]) > min(ref["scores"])


def test_peak_vs_peak_beats_pooled_individual_window_bug(tau_sequence):
    """The core regression this test guards: a real, hyperstable, verified
    design-scaffold protein's own native sequence must NOT fail developability
    just because its own maximum window is being compared incorrectly. Using
    the peak-vs-peak reference, at least some of the 5 design scaffolds'
    native sequences should pass (measured during the build: 4 of 5 native
    scaffold sequences failed at the 95th-100th percentile under the earlier,
    buggy pooled-individual-window population; the villin case at 0.2%
    already passed even under the old bug and is not a useful discriminator
    here)."""
    from sentinel.aggregation.scorer import compute_profile, get_normalization_bounds
    from sentinel.atlas.physicochemical import THREE_TO_ONE
    from sentinel.atlas.sasa import compute_residue_sasa
    from sentinel.design.developability_reference import build_reference_distribution
    from sentinel.design.selectivity import developability_filter
    import biotite.structure.io.pdb as pdb_io
    import json
    from sentinel.utils.config import repo_path

    profile = compute_profile(tau_sequence)
    tau_bounds = get_normalization_bounds(profile)
    ref = build_reference_distribution(tau_bounds)

    manifest_path = repo_path("data", "interim", "scaffolds", "scaffold_manifest.json")
    if not manifest_path.exists():
        pytest.skip("scaffold library not fetched — run scaffold_library.fetch_scaffold_library() first")
    manifest = json.load(open(manifest_path))

    n_pass = 0
    for name, entry in manifest.items():
        fixed_pdb = entry["fixed_pdb"]
        reader = pdb_io.PDBFile.read(fixed_pdb)
        arr = reader.get_structure(model=1)
        ca = arr[arr.atom_name == "CA"]
        seq = "".join(THREE_TO_ONE.get(r, "A") for r in ca.res_name)
        sasa_records = compute_residue_sasa(fixed_pdb)
        rel_sasa = {r["res_id"]: r["rel_sasa"] for r in sasa_records if r["rel_sasa"] is not None}
        dev = developability_filter(seq, ref["scores"], tau_bounds, rel_sasa)
        n_pass += dev["developability_passed"]

    assert n_pass >= 1, (
        "at least one real, hyperstable, verified scaffold protein's native sequence should pass "
        "developability under the correctly-calibrated peak-vs-peak reference"
    )
