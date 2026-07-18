"""Published per-residue scales used by the consensus aggregation-propensity
score. Every scale is a standard, citable reference — none invented."""
from __future__ import annotations

# Chou & Fasman 1974 Biochemistry, beta-sheet conformational propensity (P_beta).
CHOU_FASMAN_BETA = {
    "ALA": 0.83, "ARG": 0.93, "ASN": 0.89, "ASP": 0.54, "CYS": 1.19,
    "GLN": 1.10, "GLU": 0.37, "GLY": 0.75, "HIS": 0.87, "ILE": 1.60,
    "LEU": 1.30, "LYS": 0.74, "MET": 1.05, "PHE": 1.38, "PRO": 0.55,
    "SER": 0.75, "THR": 1.19, "TRP": 1.37, "TYR": 1.47, "VAL": 1.70,
}

# Kyte & Doolittle 1982 J Mol Biol hydrophobicity.
KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

POSITIVE = {"ARG", "LYS", "HIS"}
NEGATIVE = {"ASP", "GLU"}
AROMATIC = {"PHE", "TYR", "TRP"}

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def net_charge(one_letter_window: str) -> float:
    charge = 0.0
    for aa in one_letter_window:
        res = ONE_TO_THREE.get(aa)
        if res in POSITIVE:
            charge += 1
        elif res in NEGATIVE:
            charge -= 1
    return charge
