"""SASA computation via freesasa, and relative-SASA-based burial classification.

Relative SASA (rSASA) = SASA / max_ASA(residue_type), using the Tien et al.
2013 theoretical (Sanders/Rose extended-tripeptide) max-ASA scale — a
standard, citable reference scale (config: atlas.max_asa_scale).
"""
from __future__ import annotations

import freesasa

from sentinel.utils.config import load_config

# Tien et al. 2013 PLoS ONE, theoretical max ASA (A^2), extended tripeptide scale (Table 1).
MAX_ASA_TIEN2013 = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}


def compute_residue_sasa(pdb_path: str) -> list[dict]:
    """Per-residue total SASA for every residue in a PDB file, via freesasa."""
    structure = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()

    records = []
    for chain_id, residues in residue_areas.items():
        for res_num, area in residues.items():
            resname = area.residueType
            total_sasa = area.total
            max_asa = MAX_ASA_TIEN2013.get(resname)
            rel_sasa = (total_sasa / max_asa) if max_asa else None
            records.append({
                "chain": chain_id,
                "res_id": int(res_num),
                "res_name": resname,
                "sasa_A2": round(total_sasa, 3),
                "rel_sasa": round(rel_sasa, 4) if rel_sasa is not None else None,
                "sidechain_sasa_A2": round(area.sideChain, 3),
                "mainchain_sasa_A2": round(area.mainChain, 3),
            })
    return records


def classify_burial(rel_sasa: float | None) -> str:
    cfg = load_config()
    if rel_sasa is None:
        return "unknown"
    buried_t = cfg["atlas"]["buried_rel_sasa_threshold"]
    exposed_t = cfg["atlas"]["exposed_rel_sasa_threshold"]
    if rel_sasa < buried_t:
        return "buried"
    if rel_sasa >= exposed_t:
        return "exposed"
    return "intermediate"


def isolated_chain_sasa(pdb_path: str, chain_id: str) -> dict[int, float]:
    """SASA of one named chain extracted FROM pdb_path and computed alone (used
    as the 'isolated' reference for buried-surface-area-upon-complexation
    calculations). Extracting from the same file that will also supply the
    'in-complex' SASA guarantees identical atom sets/coordinates for that
    chain in both calculations."""
    import biotite.structure.io.pdb as pdb_io
    import tempfile
    from pathlib import Path

    reader = pdb_io.PDBFile.read(str(pdb_path))
    arr = reader.get_structure(model=1)
    sub = arr[arr.chain_id == chain_id]
    if sub.array_length() == 0:
        raise ValueError(f"chain {chain_id!r} not found in {pdb_path}")
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    writer = pdb_io.PDBFile()
    writer.set_structure(sub)
    writer.write(str(tmp_path))
    recs = compute_residue_sasa(str(tmp_path))
    tmp_path.unlink()
    return {r["res_id"]: r["sasa_A2"] for r in recs}
