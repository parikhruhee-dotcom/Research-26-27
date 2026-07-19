"""Tests for interface_scorer.chemical_complementarity — a real bug fix:
geometric_complementarity() is entirely sequence-blind (generic ideal-
geometry CB, no side-chain chemistry), so it cannot tell a well-chosen
sequence from a poorly-chosen one on the same rigid backbone. See that
function's docstring for the measured impact on AD-selectivity."""
import numpy as np

from sentinel.design.interface_scorer import chemical_complementarity


def test_hydrophobic_match_scores_higher_than_mismatch():
    binder_ca = np.array([[0.0, 0.0, 0.0]])
    target_ca = np.array([[3.0, 0.0, 0.0]])

    hydrophobic_match = chemical_complementarity(binder_ca, "L", target_ca, ["LEU"])
    hydrophobic_mismatch = chemical_complementarity(binder_ca, "L", target_ca, ["ASP"])

    assert hydrophobic_match["hydrophobic_complementarity"] > hydrophobic_mismatch["hydrophobic_complementarity"]


def test_opposite_charge_attracts_same_charge_repels():
    binder_ca = np.array([[0.0, 0.0, 0.0]])
    target_ca = np.array([[3.0, 0.0, 0.0]])

    opposite = chemical_complementarity(binder_ca, "K", target_ca, ["ASP"])  # Lys+ near Asp-
    same = chemical_complementarity(binder_ca, "K", target_ca, ["ARG"])       # Lys+ near Arg+

    assert opposite["charge_complementarity"] == 1.0
    assert same["charge_complementarity"] == -1.0


def test_no_contacts_beyond_cutoff_returns_zero_not_nan():
    binder_ca = np.array([[0.0, 0.0, 0.0]])
    target_ca = np.array([[1000.0, 0.0, 0.0]])
    result = chemical_complementarity(binder_ca, "L", target_ca, ["LEU"], contact_distance_A=8.0)
    assert result["n_chemical_contacts"] == 0
    assert result["hydrophobic_complementarity"] == 0.0
    assert result["charge_complementarity"] == 0.0
