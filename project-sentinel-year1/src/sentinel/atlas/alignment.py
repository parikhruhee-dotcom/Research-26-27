"""Pairwise structural alignment (Kabsch superposition on CA atoms of the
shared core span) across all 8 folds, and hierarchical-clustering dendrogram
of the resulting RMSD matrix."""
from __future__ import annotations

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from sentinel.utils.config import load_config


def load_ca_by_resid(pdb_path: str, chain_id: str) -> dict[int, np.ndarray]:
    reader = pdb_io.PDBFile.read(str(pdb_path))
    arr = reader.get_structure(model=1)
    ca = arr[(arr.chain_id == chain_id) & (arr.atom_name == "CA")]
    return {int(rid): coord for rid, coord in zip(ca.res_id, ca.coord)}


def common_core_range(strain_panel: list[dict]) -> tuple[int, int]:
    starts = [e["core_start"] for e in strain_panel]
    ends = [e["core_end"] for e in strain_panel]
    return max(starts), min(ends)


def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    a = coords_a - coords_a.mean(axis=0)
    b = coords_b - coords_b.mean(axis=0)
    h = a.T @ b
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    corr = np.diag([1, 1, d])
    r = vt.T @ corr @ u.T
    a_rot = (r @ a.T).T
    return float(np.sqrt(np.mean(np.sum((a_rot - b) ** 2, axis=1))))


def pairwise_rmsd_matrix(prepared_by_pdb: dict, strain_panel: list[dict]) -> dict:
    core_start, core_end = common_core_range(strain_panel)
    shared_resids = list(range(core_start, core_end + 1))

    per_strain_coords = {}
    labels = []
    for entry in strain_panel:
        pdb_id = entry["id"]
        prepared = prepared_by_pdb[pdb_id]
        ca_by_res = load_ca_by_resid(prepared["single_pdb"], prepared["single_chain_used"])
        coords = np.array([ca_by_res[r] for r in shared_resids if r in ca_by_res])
        used_resids = [r for r in shared_resids if r in ca_by_res]
        per_strain_coords[pdb_id] = (used_resids, coords)
        labels.append(f"{entry['strain']}_{pdb_id}")

    n = len(strain_panel)
    rmsd_mat = np.zeros((n, n))
    ids = [e["id"] for e in strain_panel]
    for i in range(n):
        for j in range(i + 1, n):
            res_i, coord_i = per_strain_coords[ids[i]]
            res_j, coord_j = per_strain_coords[ids[j]]
            common = sorted(set(res_i) & set(res_j))
            if len(common) < 3:
                rmsd_mat[i, j] = rmsd_mat[j, i] = float("nan")
                continue
            ci = np.array([coord_i[res_i.index(r)] for r in common])
            cj = np.array([coord_j[res_j.index(r)] for r in common])
            d = kabsch_rmsd(ci, cj)
            rmsd_mat[i, j] = rmsd_mat[j, i] = d

    return {"ids": ids, "labels": labels, "rmsd_matrix": rmsd_mat.tolist(),
            "common_core_range": [core_start, core_end], "n_shared_residues": len(shared_resids)}


def build_dendrogram_linkage(rmsd_result: dict) -> dict:
    cfg = load_config()
    method = cfg["atlas"]["dendrogram_linkage"]
    mat = np.array(rmsd_result["rmsd_matrix"])
    mat = np.nan_to_num(mat, nan=np.nanmax(mat[~np.isnan(mat)]) * 1.5 if np.isnan(mat).any() else mat)
    condensed = squareform(mat, checks=False)
    z = linkage(condensed, method=method)
    return {"linkage_matrix": z.tolist(), "labels": rmsd_result["labels"], "method": method}
