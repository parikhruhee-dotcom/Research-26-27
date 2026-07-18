"""Core definition & topology, and inter-protofilament interface residues."""
from __future__ import annotations

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import numpy as np


def interprotofilament_contacts(full_pdb_path: str, protofilament_chains: dict[str, list[str]],
                                  cutoff_A: float = 4.5) -> list[dict]:
    """Residues in one protofilament with any heavy atom within cutoff_A of any
    heavy atom in a different protofilament — the physical packing interface
    between protofilaments (only meaningful when >1 protofilament present)."""
    groups = list(protofilament_chains.values())
    if len(groups) < 2:
        return []

    reader = pdb_io.PDBFile.read(str(full_pdb_path))
    arr = reader.get_structure(model=1)
    arr = arr[arr.element != "H"]

    contacts = []
    # only compare the first two protofilament groups (sufficient for a pairwise interface report)
    group_a, group_b = groups[0], groups[1]
    mask_a = np.isin(arr.chain_id, group_a)
    mask_b = np.isin(arr.chain_id, group_b)
    sub_a, sub_b = arr[mask_a], arr[mask_b]
    if sub_a.array_length() == 0 or sub_b.array_length() == 0:
        return []

    cell_list = struc.CellList(sub_b, cell_size=cutoff_A)
    contact_res = set()
    for i in range(sub_a.array_length()):
        neighbors = cell_list.get_atoms(sub_a.coord[i], radius=cutoff_A)
        if len(neighbors) > 0:
            contact_res.add(("A", int(sub_a.res_id[i]), sub_a.res_name[i]))
            for j in neighbors:
                contact_res.add(("B", int(sub_b.res_id[j]), sub_b.res_name[j]))

    for side, res_id, res_name in sorted(contact_res):
        contacts.append({"protofilament_side": side, "res_id": res_id, "res_name": res_name})
    return contacts


def core_topology(strain_entry: dict, dssp_records: list[dict], prepared_entry: dict) -> dict:
    from sentinel.atlas.secondary_structure import beta_strand_count
    chain0 = dssp_records[0]["chain"] if dssp_records else None
    n_beta_strands_per_layer = beta_strand_count(dssp_records, chain0) if chain0 else 0

    return {
        "pdb_id": strain_entry["id"],
        "strain": strain_entry["strain"],
        "core_start": strain_entry["core_start"],
        "core_end": strain_entry["core_end"],
        "core_length": strain_entry["core_end"] - strain_entry["core_start"] + 1,
        "n_beta_strands_per_layer": n_beta_strands_per_layer,
        "protofilament_count": prepared_entry["n_protofilaments_detected"],
        "protofilament_sizes": prepared_entry["protofilament_sizes"],
        "citation": strain_entry["citation"],
    }
