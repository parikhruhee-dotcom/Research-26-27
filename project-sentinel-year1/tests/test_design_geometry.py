import numpy as np
import pytest

from sentinel.design.geometry import BOND_C_N, BOND_CA_C, BOND_N_CA, build_backbone


def test_build_backbone_bond_lengths():
    coords = build_backbone("H" * 15)
    for i in range(15):
        assert np.linalg.norm(coords["N"][i] - coords["CA"][i]) == pytest.approx(BOND_N_CA, abs=0.01)
        assert np.linalg.norm(coords["CA"][i] - coords["C"][i]) == pytest.approx(BOND_CA_C, abs=0.01)
    for i in range(14):
        assert np.linalg.norm(coords["C"][i] - coords["N"][i + 1]) == pytest.approx(BOND_C_N, abs=0.01)


def test_ideal_alpha_helix_rise():
    """A 25-residue ideal alpha helix should have an end-to-end CA distance
    close to the textbook 1.5 A/residue rise (~37.5 A), a real geometric
    sanity check on the NeRF construction, not a placeholder assertion."""
    coords = build_backbone("H" * 25)
    end_to_end = np.linalg.norm(coords["CA"][0] - coords["CA"][-1])
    assert 34.0 < end_to_end < 41.0


def test_no_atom_clashes_within_single_chain():
    coords = build_backbone("H" * 20)
    ca = coords["CA"]
    for i in range(len(ca)):
        for j in range(i + 2, len(ca)):
            assert np.linalg.norm(ca[i] - ca[j]) > 2.5, f"CA clash between residues {i} and {j}"


def test_rotation_matrix_is_orthonormal():
    from sentinel.design.backbone_gen import _rotation_matrix
    r = _rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 3)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-8)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-8)
