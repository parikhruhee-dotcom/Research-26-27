"""Build OpenMM-ready starting structures for the two MD systems (M4):
  1. Bare hexapeptides (PHF6, PHF6*) — starting coordinates taken from the
     real zipper microcrystal structures (2ON9 chain A = VQIVYK, 5V5C chain A
     = VQIINK), a single monomer extracted, waters/symmetry mates stripped.
  2. The fibril growing tip — the AD (or any strain's) 3-layer stacked model
     from M1d directly.
Both are cleaned/protonated with PDBFixer before simulation.
"""
from __future__ import annotations

import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb_io
from openmm.app import PDBFile
from pdbfixer import PDBFixer

from sentinel.utils.config import repo_path


def extract_hexapeptide_pdb(source_cif: str, chain_id: str, dest_pdb) -> None:
    cif = pdbx.CIFFile.read(str(source_cif))
    arr = pdbx.get_structure(cif, model=1)
    arr = arr[(~arr.hetero) & (arr.chain_id == chain_id)]
    writer = pdb_io.PDBFile()
    writer.set_structure(arr)
    writer.write(str(dest_pdb))


def fix_and_protonate(raw_pdb, fixed_pdb, ph: float = 7.0) -> None:
    fixer = PDBFixer(filename=str(raw_pdb))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with open(fixed_pdb, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


def build_hexapeptide_system(motif: str) -> str:
    """motif: 'PHF6' or 'PHF6_star'. Returns path to the fixed, protonated PDB."""
    sources = {
        "PHF6": (repo_path("data", "raw", "structures", "2ON9.cif"), "A"),
        "PHF6_star": (repo_path("data", "raw", "structures", "5V5C.cif"), "A"),
    }
    source_cif, chain_id = sources[motif]
    interim_dir = repo_path("data", "interim", "md")
    interim_dir.mkdir(parents=True, exist_ok=True)
    raw_pdb = interim_dir / f"{motif}_raw.pdb"
    fixed_pdb = interim_dir / f"{motif}_fixed.pdb"
    extract_hexapeptide_pdb(source_cif, chain_id, raw_pdb)
    fix_and_protonate(raw_pdb, fixed_pdb)
    return str(fixed_pdb)


def build_fibril_tip_system(stack_pdb_path: str, tag: str) -> str:
    interim_dir = repo_path("data", "interim", "md")
    interim_dir.mkdir(parents=True, exist_ok=True)
    fixed_pdb = interim_dir / f"{tag}_tip_fixed.pdb"
    fix_and_protonate(stack_pdb_path, fixed_pdb)
    return str(fixed_pdb)
