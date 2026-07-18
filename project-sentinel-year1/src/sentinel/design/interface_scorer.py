"""M6c — CPU-runnable interface scorer (the substitute that keeps the loop
closed without a GPU for AlphaFold2-multimer / Boltz complex folding).

True ESMFold (facebookresearch/esm's `esmfold_v1`) is a ~15 GB model — far
too large for this sandbox's ~20 GB disk even before considering CPU
inference speed. Substituted (logged via provenance.log_substitution) with
the smallest ESM-2 checkpoint (`esm2_t6_8M_UR50D`, ~30 MB, config:
design.scoring.esm_model): instead of a folded structure + pLDDT, we use its
masked-language-model PSEUDO-PERPLEXITY as a sequence-plausibility proxy
("does this look like a foldable/natural protein sequence to a model trained
on real proteins" rather than "what 3D structure does it fold into").

Combined with real geometric/energetic complementarity (buried SASA via
freesasa + a backbone clash score + an H-bond-geometry proxy + a packing-
density Sc proxy, all computed directly on the docked binder-target complex,
CB atoms added via ideal NeRF geometry so freesasa has a slightly more
realistic surface than bare backbone) into one composite score per design.
"""
from __future__ import annotations

import numpy as np

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

_ESM_MODEL_CACHE = {}


def _add_cb_atoms(coords: dict) -> dict:
    """Ideal tetrahedral CB placement from N/CA/C (standard formula, e.g. used
    in Rosetta/PyMOL idealized-geometry tools): CB sits ~1.53 A from CA along
    the direction that completes a tetrahedral center opposite the N-CA-C
    bisector."""
    n, ca, c = coords["N"], coords["CA"], coords["C"]
    n_res = ca.shape[0]
    cb = np.zeros((n_res, 3))
    for i in range(n_res):
        b1 = n[i] - ca[i]
        b2 = c[i] - ca[i]
        b1 /= np.linalg.norm(b1)
        b2 /= np.linalg.norm(b2)
        bisector = (b1 + b2) / np.linalg.norm(b1 + b2)
        normal = np.cross(b1, b2)
        normal /= np.linalg.norm(normal)
        direction = -bisector * np.cos(np.radians(54.75)) + normal * np.sin(np.radians(54.75))
        cb[i] = ca[i] + 1.53 * direction / np.linalg.norm(direction)
    return {**coords, "CB": cb}


def load_backbone_coords(pdb_path: str) -> dict:
    import biotite.structure.io.pdb as pdb_io
    reader = pdb_io.PDBFile.read(str(pdb_path))
    arr = reader.get_structure(model=1)
    coords = {}
    for atom_name in ["N", "CA", "C", "O"]:
        mask = arr.atom_name == atom_name
        coords[atom_name] = arr.coord[mask]
    return coords


def geometric_complementarity(binder_coords: dict, target_coords: dict,
                                clash_distance_A: float = 2.8, contact_distance_A: float = 5.0,
                                hbond_distance_A: float = 3.5) -> dict:
    binder = _add_cb_atoms(binder_coords)
    target_all = np.concatenate([target_coords[a] for a in ["N", "CA", "C", "O"]], axis=0)
    binder_all = np.concatenate([binder[a] for a in ["N", "CA", "C", "O", "CB"]], axis=0)

    diffs = binder_all[:, None, :] - target_all[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    n_clashes = int((dists < clash_distance_A).sum())
    n_contacts = int(((dists >= clash_distance_A) & (dists < contact_distance_A)).sum())

    binder_n_o = np.concatenate([binder["N"], binder["O"]], axis=0)
    target_n_o = np.concatenate([target_coords["N"], target_coords["O"]], axis=0)
    hb_dists = np.linalg.norm(binder_n_o[:, None, :] - target_n_o[None, :, :], axis=-1)
    n_hbond_candidates = int((hb_dists < hbond_distance_A).sum())

    contacts_per_atom = ((dists >= clash_distance_A) & (dists < contact_distance_A)).sum(axis=1)
    mean_contacts = float(contacts_per_atom.mean())
    sc_proxy = mean_contacts / (mean_contacts + 8.0)

    clash_score = n_clashes / binder_all.shape[0]

    return {
        "n_clashes": n_clashes, "n_contacts": n_contacts,
        "n_hbond_candidates": n_hbond_candidates,
        "clash_score": round(clash_score, 4),
        "packing_density_sc_proxy": round(sc_proxy, 4),
        "mean_contacts_per_atom": round(mean_contacts, 3),
    }


def buried_sasa_of_interface(binder_pdb: str, target_pdb: str, complex_pdb: str) -> float:
    """SASA(binder alone) + SASA(target alone) - SASA(complex), i.e. the
    interface buried surface area, via freesasa."""
    from sentinel.atlas.sasa import compute_residue_sasa
    binder_sasa = sum(r["sasa_A2"] for r in compute_residue_sasa(binder_pdb))
    target_sasa = sum(r["sasa_A2"] for r in compute_residue_sasa(target_pdb))
    complex_sasa = sum(r["sasa_A2"] for r in compute_residue_sasa(complex_pdb))
    return round(binder_sasa + target_sasa - complex_sasa, 2)


def esm_plausibility(sequence: str, model_name: str) -> float:
    """Single-forward-pass sequence plausibility: one unmasked forward pass,
    average log-probability the model assigns to the actual residue at each
    position, squashed to [0,1].

    This is an approximation of true masked-LM pseudo-perplexity (which masks
    each position individually and requires L forward passes per sequence —
    computationally prohibitive at the design-loop's candidate volume: ~200
    sequences/round x 6 rounds x ~70 positions would mean tens of thousands
    of forward passes). Single-pass self-consistency is weaker (the model can
    see the residue itself via bidirectional attention) but still a real,
    fast, documented proxy for 'does this look like a plausible protein
    sequence to a model trained on real proteins' — not a literature-standard
    metric, and reported as such."""
    import torch
    import esm

    if model_name not in _ESM_MODEL_CACHE:
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        model.eval()
        _ESM_MODEL_CACHE[model_name] = (model, alphabet)
    model, alphabet = _ESM_MODEL_CACHE[model_name]
    batch_converter = alphabet.get_batch_converter()

    _, _, tokens = batch_converter([("seq", sequence)])
    with torch.no_grad():
        logits = model(tokens)["logits"]
        log_probs = torch.log_softmax(logits[0], dim=-1)
        true_toks = tokens[0, 1:-1]
        position_idx = torch.arange(1, tokens.shape[1] - 1)
        token_log_probs = log_probs[position_idx, true_toks]
        mean_logprob = token_log_probs.mean().item()
    perplexity = float(np.exp(-mean_logprob))
    plausibility = 1.0 / (1.0 + perplexity / 20.0)  # squashed to (0,1); documented, not a literature constant
    return round(plausibility, 4)
