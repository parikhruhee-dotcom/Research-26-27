"""Tests for the real solved-structure scaffold library (M6a quality
upgrade — see scaffold_library.py docstring for the scientific rationale:
motif-grafting design with real hyperstable folds instead of idealized
cylinders)."""
import pytest

from sentinel.design.scaffold_library import SCAFFOLD_SOURCES


@pytest.fixture(scope="module")
def scaffolds_available(repo_root):
    manifest_path = repo_root / "data" / "interim" / "scaffolds" / "scaffold_manifest.json"
    if not manifest_path.exists():
        pytest.skip("scaffold library not fetched yet — run `python -m sentinel.design.scaffold_library`")
    return manifest_path


def test_all_scaffold_sources_have_required_fields():
    for name, spec in SCAFFOLD_SOURCES.items():
        assert "pdb_id" in spec and len(spec["pdb_id"]) == 4
        assert "chain" in spec
        assert "expected_length" in spec and spec["expected_length"] > 0
        assert "description" in spec and len(spec["description"]) > 20


def test_scaffold_backbones_load_with_expected_size(scaffolds_available):
    from sentinel.design.scaffold_library import load_scaffold_backbone
    for name, spec in SCAFFOLD_SOURCES.items():
        coords = load_scaffold_backbone(name)
        n_res = coords["CA"].shape[0]
        assert abs(n_res - spec["expected_length"]) <= 5, (
            f"{name}: expected ~{spec['expected_length']} residues, got {n_res}"
        )
        for atom in ["N", "CA", "C", "O"]:
            assert coords[atom].shape == (n_res, 3)


def test_scaffold_backbones_have_real_bond_lengths(scaffolds_available):
    """Sanity check that these are real, physically valid backbones (not
    corrupted during extraction) — N-CA and CA-C distances should match
    standard peptide geometry throughout."""
    import numpy as np
    from sentinel.design.scaffold_library import load_scaffold_backbone

    for name in SCAFFOLD_SOURCES:
        coords = load_scaffold_backbone(name)
        n_ca = np.linalg.norm(coords["N"] - coords["CA"], axis=1)
        ca_c = np.linalg.norm(coords["CA"] - coords["C"], axis=1)
        assert np.median(n_ca) == pytest.approx(1.46, abs=0.15)
        assert np.median(ca_c) == pytest.approx(1.52, abs=0.15)
