"""M1d — clean and prepare each strain-panel structure.

For every entry in the panel:
  1. Load the mmCIF, strip waters/heteroatoms, keep only ordered protein chains.
  2. Identify which chains belong to which protofilament using a geometric
     nearest-neighbor graph on CA centroids (stacked fibril layers sit ~4.7-5.0 A
     apart along the helical axis; separate protofilaments sit much farther
     apart — see PROGRESS_LOG.md M1d for the empirical distances measured on
     this exact panel, e.g. 5O3L/5O3T resolve cleanly into two 5-layer
     protofilaments at 4.75 A intra-stack vs 9.5 A inter-protofilament).
  3. Pick the largest protofilament cluster, order its chains along the
     filament axis (PCA on that cluster's centroids), and write:
       - a single-protofilament, single-layer model (`*_single.pdb`)
       - a stacked-layers model with >= config min_stacked_layers (`*_stack.pdb`)
  4. Run PDBFixer on both to add missing atoms/hydrogens.

Run: python -m sentinel.io.prepare_structures
"""
from __future__ import annotations

import itertools
import json

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb_io
import numpy as np
from openmm.app import PDBFile
from pdbfixer import PDBFixer

from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger
from sentinel.utils.provenance import log_substitution

logger = get_logger(__name__)


def load_clean_ca_model(cif_path) -> struc.AtomArray:
    cif = pdbx.CIFFile.read(str(cif_path))
    arr = pdbx.get_structure(cif, model=1)
    arr = arr[~arr.hetero]
    return arr


def cluster_protofilaments(arr: struc.AtomArray) -> dict[str, list[str]]:
    """Group chain IDs into protofilaments via a nearest-neighbor centroid graph."""
    ca = arr[arr.atom_name == "CA"]
    chains = sorted(set(ca.chain_id))
    centroids = {ch: ca[ca.chain_id == ch].coord.mean(axis=0) for ch in chains}

    pairs = list(itertools.combinations(chains, 2))
    dists = {(a, b): float(np.linalg.norm(centroids[a] - centroids[b])) for a, b in pairs}
    if not dists:
        return {chains[0]: chains} if chains else {}

    min_d = min(dists.values())
    threshold = min_d * 1.6  # empirically: intra-stack ~4.7-5.0 A, inter-protofilament >= 9.5 A (see docstring)

    # union-find over chains connected by an edge below threshold
    parent = {c: c for c in chains}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (a, b), d in dists.items():
        if d <= threshold:
            union(a, b)

    clusters: dict[str, list[str]] = {}
    for c in chains:
        root = find(c)
        clusters.setdefault(root, []).append(c)

    # order chains within each cluster along that cluster's principal axis
    ordered_clusters = {}
    for root, members in clusters.items():
        if len(members) == 1:
            ordered_clusters[root] = members
            continue
        pts = np.array([centroids[m] for m in members])
        pts_centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts_centered)
        axis = vt[0]
        proj = pts_centered @ axis
        order = np.argsort(proj)
        ordered_clusters[root] = [members[i] for i in order]
    return ordered_clusters


def write_chains_subset(arr: struc.AtomArray, chain_ids: list[str], dest) -> None:
    mask = np.isin(arr.chain_id, chain_ids)
    sub = arr[mask]
    dest.parent.mkdir(parents=True, exist_ok=True)
    pdb_file = pdb_io.PDBFile()
    pdb_file.set_structure(sub)
    pdb_file.write(str(dest))


def fix_with_pdbfixer(raw_pdb_path, fixed_pdb_path) -> dict:
    fixer = PDBFixer(filename=str(raw_pdb_path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with open(fixed_pdb_path, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    n_missing_res = sum(len(v) for v in fixer.missingResidues.values())
    n_atoms = sum(1 for _ in fixer.topology.atoms())
    return {"missing_residues_filled": n_missing_res, "n_atoms_after_fix": n_atoms}


def prepare_one(entry: dict, min_layers: int) -> dict:
    pdb_id = entry["pdb_id"]
    cif_path = repo_path(entry["cif_path"])
    arr = load_clean_ca_model(cif_path)
    clusters = cluster_protofilaments(arr)
    best_root = max(clusters, key=lambda r: len(clusters[r]))
    ordered_chains = clusters[best_root]

    n_layers_available = len(ordered_chains)
    if n_layers_available < min_layers:
        logger.warning(f"{pdb_id}: only {n_layers_available} layers in largest protofilament "
                        f"(< configured min_stacked_layers={min_layers}); using all available layers "
                        f"and documenting the limitation rather than fabricating layers.")
    stack_chains = ordered_chains[-min(min_layers, n_layers_available):] if n_layers_available >= 2 \
        else ordered_chains
    single_chain = [ordered_chains[len(ordered_chains) // 2]]  # a middle layer, least edge-affected

    interim_dir = repo_path("data", "interim", "structures")
    raw_single = interim_dir / f"{pdb_id}_single_raw.pdb"
    raw_stack = interim_dir / f"{pdb_id}_stack_raw.pdb"
    raw_full = interim_dir / f"{pdb_id}_full_raw.pdb"
    all_chains = [c for members in clusters.values() for c in members]
    write_chains_subset(arr, single_chain, raw_single)
    write_chains_subset(arr, stack_chains, raw_stack)
    write_chains_subset(arr, all_chains, raw_full)

    fixed_single = interim_dir / f"{pdb_id}_single.pdb"
    fixed_stack = interim_dir / f"{pdb_id}_stack.pdb"
    fixed_full = interim_dir / f"{pdb_id}_full.pdb"
    fix_info_single = fix_with_pdbfixer(raw_single, fixed_single)
    fix_info_stack = fix_with_pdbfixer(raw_stack, fixed_stack)
    fix_info_full = fix_with_pdbfixer(raw_full, fixed_full)

    raw_single.unlink()
    raw_stack.unlink()
    raw_full.unlink()

    return {
        "pdb_id": pdb_id,
        "strain": entry["strain"],
        "n_protofilaments_detected": len(clusters),
        "protofilament_sizes": sorted((len(v) for v in clusters.values()), reverse=True),
        "n_layers_used_for_stack": len(stack_chains),
        "single_chain_used": single_chain[0],
        "stack_chains_used": stack_chains,
        "single_pdb": str(fixed_single.relative_to(repo_path())),
        "stack_pdb": str(fixed_stack.relative_to(repo_path())),
        "full_pdb": str(fixed_full.relative_to(repo_path())),
        "full_chains_by_protofilament": {r: v for r, v in clusters.items()},
        "pdbfixer_single": fix_info_single,
        "pdbfixer_stack": fix_info_stack,
        "pdbfixer_full": fix_info_full,
        "layers_below_target": n_layers_available < min_layers,
    }


def main() -> list[dict]:
    cfg = load_config()
    manifest_path = repo_path("data", "raw", "structures", "panel_manifest.json")
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    min_layers = cfg["data"]["min_stacked_layers"]
    results = []
    for entry in manifest:
        try:
            res = prepare_one(entry, min_layers)
            results.append(res)
            logger.info(f"{entry['pdb_id']}: {res['n_protofilaments_detected']} protofilament(s) "
                        f"detected {res['protofilament_sizes']}, using {res['n_layers_used_for_stack']} "
                        f"layers for the stack model.")
        except Exception as exc:
            logger.error(f"structure preparation failed for {entry['pdb_id']}: {exc}")
            results.append({"pdb_id": entry["pdb_id"], "strain": entry["strain"], "error": str(exc)})

    out_path = repo_path("data", "interim", "structures", "prepared_manifest.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    n_ok = sum(1 for r in results if "error" not in r)
    n_short = sum(1 for r in results if r.get("layers_below_target"))
    log_substitution(
        "DSSP/secondary-structure geometry via literature symmetry annotation",
        "geometric nearest-neighbor centroid-graph clustering (this script) to auto-detect "
        "protofilament membership and layer order directly from coordinates",
        "The brief does not specify an algorithm for separating protofilaments/ordering layers; "
        "cryo-EM deposits do not tag this explicitly in a machine-readable way. A distance-based "
        "graph on CA centroids reproduces the known symmetry (verified: 5O3L/5O3T split cleanly "
        "into two 5-layer protofilaments at 4.75 A intra-stack vs 9.5 A inter-protofilament, "
        "matching the published C2-symmetric PHF architecture).",
    )
    append_progress_log(
        "M1d",
        f"Prepared single-protofilament and >= {min_layers}-layer stacked models for {n_ok}/{len(results)} "
        f"strain-panel structures (PDBFixer: missing atoms/residues filled, hydrogens added at pH 7.0, "
        f"heterogens/waters stripped). {n_short} structures had fewer than {min_layers} layers available "
        f"in their largest deposited protofilament and used all available layers instead (documented per "
        f"entry in data/interim/structures/prepared_manifest.json, field 'layers_below_target').",
    )
    assert n_ok == len(results), "not all structures prepared successfully"
    return results


if __name__ == "__main__":
    main()
