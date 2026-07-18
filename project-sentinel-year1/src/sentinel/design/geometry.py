"""Idealized-geometry backbone construction via NeRF (Natural Extension
Reference Frame) — a standard, real, deterministic method for building
Cartesian coordinates from bond lengths/angles/dihedrals. Used by
backbone_gen.py to build mini-protein scaffold backbones from a topology
string (secondary-structure per residue) when no GPU is available to run
RFdiffusion.

Standard peptide geometry constants (Engh & Huber 1991, widely used defaults):
  N-CA = 1.458 A, CA-C = 1.525 A, C-N = 1.329 A
  N-CA-C = 111.2 deg, CA-C-N = 116.2 deg, C-N-CA = 121.7 deg
Ideal secondary-structure dihedrals (Ramachandran-favored regions):
  alpha-helix:  phi=-57 deg,  psi=-47 deg
  beta-strand:  phi=-120 deg, psi=+120 deg
  loop (PPII-like, generic coil): phi=-75 deg, psi=+145 deg
"""
from __future__ import annotations

import numpy as np

BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
ANGLE_N_CA_C = np.radians(111.2)
ANGLE_CA_C_N = np.radians(116.2)
ANGLE_C_N_CA = np.radians(121.7)
OMEGA = np.radians(180.0)  # trans peptide bond

SS_DIHEDRALS = {
    "H": (np.radians(-57.0), np.radians(-47.0)),   # alpha-helix (phi, psi)
    "E": (np.radians(-120.0), np.radians(120.0)),  # beta-strand
    "L": (np.radians(-75.0), np.radians(145.0)),   # loop
}


def _nerf(a: np.ndarray, b: np.ndarray, c: np.ndarray, bond_len: float, bond_angle: float,
          dihedral: float) -> np.ndarray:
    """Place a 4th atom given 3 preceding atoms (a-b-c) and the bond length
    (c-d), bond angle (b-c-d), and dihedral (a-b-c-d). Standard NeRF formula."""
    bc = (c - b) / np.linalg.norm(c - b)
    ab = b - a
    n = np.cross(ab, bc)
    n = n / np.linalg.norm(n)
    m = np.cross(n, bc)

    d2 = np.array([
        -bond_len * np.cos(bond_angle),
        bond_len * np.sin(bond_angle) * np.cos(dihedral),
        bond_len * np.sin(bond_angle) * np.sin(dihedral),
    ])
    basis = np.stack([bc, m, n], axis=1)
    return c + basis @ d2


def build_backbone(ss_string: str) -> dict:
    """Build N/CA/C/O coordinates for a backbone whose per-residue secondary
    structure is given by ss_string (chars 'H'/'E'/'L'). Returns arrays keyed
    by atom name, each shape (n_residues, 3).

    Chain-growth convention: phi_i = dihedral(C_{i-1},N_i,CA_i,C_i), psi_i =
    dihedral(N_i,CA_i,C_i,N_{i+1}), omega ~ 180 deg (trans peptide bond).
    """
    n_res = len(ss_string)
    coords = {"N": np.zeros((n_res, 3)), "CA": np.zeros((n_res, 3)), "C": np.zeros((n_res, 3)),
              "O": np.zeros((n_res, 3))}

    # seed the first residue with a fixed local frame
    coords["N"][0] = np.array([0.0, 0.0, 0.0])
    coords["CA"][0] = coords["N"][0] + np.array([BOND_N_CA, 0.0, 0.0])
    theta = np.pi - ANGLE_N_CA_C
    coords["C"][0] = coords["CA"][0] + BOND_CA_C * np.array([np.cos(theta), np.sin(theta), 0.0])

    psi_prev = SS_DIHEDRALS[ss_string[0]][1]
    for i in range(1, n_res):
        phi_i, psi_i = SS_DIHEDRALS[ss_string[i]]
        prev_n, prev_ca, prev_c = coords["N"][i - 1], coords["CA"][i - 1], coords["C"][i - 1]

        coords["N"][i] = _nerf(prev_n, prev_ca, prev_c, BOND_C_N, ANGLE_CA_C_N, psi_prev)
        coords["CA"][i] = _nerf(prev_ca, prev_c, coords["N"][i], BOND_N_CA, ANGLE_C_N_CA, OMEGA)
        coords["C"][i] = _nerf(prev_c, coords["N"][i], coords["CA"][i], BOND_CA_C, ANGLE_N_CA_C, phi_i)
        psi_prev = psi_i

    for i in range(n_res):
        # carbonyl O: standard idealization, CA-C-O angle 120.8 deg, ~anti to psi
        dihedral_ref = SS_DIHEDRALS[ss_string[i]][1] + np.pi
        coords["O"][i] = _nerf(coords["N"][i], coords["CA"][i], coords["C"][i], 1.231,
                                 np.radians(120.8), dihedral_ref)

    return coords
