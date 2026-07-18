"""Physicochemical surface characterization: Kyte-Doolittle hydrophobicity,
charge classification, aromaticity — per-residue, sequence-based (no external
service needed, standard published scales)."""
from __future__ import annotations

# Kyte & Doolittle 1982 J Mol Biol hydrophobicity scale
KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

POSITIVE = {"ARG", "LYS", "HIS"}
NEGATIVE = {"ASP", "GLU"}
AROMATIC = {"PHE", "TYR", "TRP"}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def charge_class(res_name: str) -> str:
    if res_name in POSITIVE:
        return "positive"
    if res_name in NEGATIVE:
        return "negative"
    return "neutral"


def residue_physchem(res_name: str) -> dict:
    return {
        "hydrophobicity_kd": KYTE_DOOLITTLE.get(res_name, 0.0),
        "charge_class": charge_class(res_name),
        "is_aromatic": res_name in AROMATIC,
        "one_letter": THREE_TO_ONE.get(res_name, "X"),
    }
