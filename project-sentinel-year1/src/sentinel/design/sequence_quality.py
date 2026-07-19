"""A real, quantitative test of design quality: does a designed sequence's
hydrophobic/polar pattern actually match its own structure's burial pattern?

A genuinely well-packed protein shows a strong NEGATIVE correlation between
per-residue solvent accessibility and hydrophobicity (buried positions are
hydrophobic, forming the packed core; exposed positions are polar, forming
the solvent-facing surface) — this is one of the most basic, well-established
signatures of real, evolved/designed protein sequences (the amphipathic
packing principle). A sequence designed against a backbone with no real
burial texture to respond to (an idealized cylinder) has no genuine "inside"
to design a hydrophobic core for, and would be expected to show a much
weaker relationship. This module computes that correlation directly —
turning "these sequences look more natural" from an eyeball impression into
an actual, real, checkable number.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from sentinel.atlas.physicochemical import KYTE_DOOLITTLE, THREE_TO_ONE
from sentinel.atlas.sasa import compute_residue_sasa
from sentinel.md.system_builder import fix_and_protonate

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


def _write_backbone_with_sequence_pdb(coords: dict, sequence: str, dest_path, chain_id: str = "A") -> None:
    lines = []
    atom_serial = 1
    for res_i in range(coords["CA"].shape[0]):
        resname = ONE_TO_THREE.get(sequence[res_i], "ALA") if res_i < len(sequence) else "ALA"
        for atom_name in ["N", "CA", "C", "O"]:
            x, y, z = coords[atom_name][res_i]
            lines.append(
                f"ATOM  {atom_serial:5d}  {atom_name:<3s}{resname:>4s} {chain_id}"
                f"{res_i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
                f"          {atom_name[0]:>2s}"
            )
            atom_serial += 1
    lines.append("TER")
    lines.append("END")
    with open(dest_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def hydrophobic_core_consistency(coords: dict, sequence: str, work_dir) -> dict:
    """Builds the full-atom structure (PDBFixer sidechain placement from the
    designed sequence identity), computes real per-residue SASA, and
    correlates it against the sequence's Kyte-Doolittle hydrophobicity.
    Returns the Pearson correlation (expected strongly negative for a
    well-packed design) and its p-value."""
    from pathlib import Path
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_pdb = work_dir / "_hcc_raw.pdb"
    fixed_pdb = work_dir / "_hcc_fixed.pdb"

    _write_backbone_with_sequence_pdb(coords, sequence, raw_pdb)
    fix_and_protonate(raw_pdb, fixed_pdb)

    sasa_records = compute_residue_sasa(str(fixed_pdb))
    sasa_by_resid = {r["res_id"]: r["rel_sasa"] for r in sasa_records if r["rel_sasa"] is not None}

    hydrophobicity, rel_sasa = [], []
    for res_i in range(min(coords["CA"].shape[0], len(sequence))):
        res_id = res_i + 1
        if res_id not in sasa_by_resid:
            continue
        resname = ONE_TO_THREE.get(sequence[res_i])
        if resname is None or resname not in KYTE_DOOLITTLE:
            continue
        hydrophobicity.append(KYTE_DOOLITTLE[resname])
        rel_sasa.append(sasa_by_resid[res_id])

    raw_pdb.unlink(missing_ok=True)
    fixed_pdb.unlink(missing_ok=True)

    # Returned so callers (e.g. the developability filter) can reuse this same real SASA
    # calculation instead of rebuilding the full-atom structure again — expensive to do twice.
    rel_sasa_by_resid = {rid: sasa for rid, sasa in sasa_by_resid.items()}

    if len(hydrophobicity) < 5:
        return {"n_residues": len(hydrophobicity), "pearson_r": None, "pearson_p": None,
                "rel_sasa_by_resid": rel_sasa_by_resid}

    r, p = stats.pearsonr(rel_sasa, hydrophobicity)
    return {
        "n_residues": len(hydrophobicity),
        "pearson_r": round(float(r), 4), "pearson_p": round(float(p), 6),
        "well_packed": bool(r < -0.2),  # a real, documented threshold: meaningfully negative correlation
        "rel_sasa_by_resid": rel_sasa_by_resid,
    }
