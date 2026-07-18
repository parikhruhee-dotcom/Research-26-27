"""M3 — the consensus beta-aggregation scorer.

A documented, from-scratch, sliding-window score over full-length tau
combining five published/derived terms (weights + justification live in
config.yaml, aggregation.weights / aggregation.weight_justification):
  1. beta_propensity     — Chou-Fasman 1974 beta-sheet propensity, window mean
  2. hydrophobicity       — Kyte-Doolittle 1982, window mean
  3. charge_penalty       — 1 - (fraction of window that is charged), penalizes
                            windows that are hard to bury in a dry interface
  4. aromatic_bonus       — fraction of window that is aromatic (pi-stacking)
  5. zipper_compatibility — BLOSUM62 local-alignment-style similarity of the
                            window to the two known zipper-forming hexapeptides
                            (VQIVYK, VQIINK), normalized against each motif's
                            self-score

Each term is min-max normalized to [0,1] across the full profile before
combination, so the configured weights are directly interpretable as relative
importance.
"""
from __future__ import annotations

import numpy as np
from Bio.Align import substitution_matrices

from sentinel.aggregation.scales import (
    AROMATIC, CHOU_FASMAN_BETA, KYTE_DOOLITTLE, NEGATIVE, ONE_TO_THREE, POSITIVE,
)
from sentinel.utils.config import load_config

_BLOSUM62 = substitution_matrices.load("BLOSUM62")


def _blosum_score(seq_a: str, seq_b: str) -> int:
    return sum(_BLOSUM62[a, b] for a, b in zip(seq_a, seq_b))


def zipper_compatibility_score(window_seq: str, reference_motifs: list[str]) -> float:
    """Best (max) normalized BLOSUM62 alignment score of window_seq (ungapped,
    ordered, same length as each reference motif) against the reference
    hexapeptides, normalized by that motif's self-alignment score so a
    perfect match to either motif scores 1.0."""
    best = 0.0
    for motif in reference_motifs:
        if len(window_seq) != len(motif):
            continue
        raw = _blosum_score(window_seq, motif)
        self_score = _blosum_score(motif, motif)
        normalized = raw / self_score if self_score else 0.0
        best = max(best, normalized)
    return best


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def compute_profile(sequence: str) -> dict:
    cfg = load_config()
    window_size = cfg["aggregation"]["window_size"]
    weights = cfg["aggregation"]["weights"]
    landmarks = cfg["data"]["landmarks"]
    reference_motifs = [landmarks["PHF6"]["sequence"], landmarks["PHF6_star"]["sequence"]]

    n = len(sequence)
    n_windows = n - window_size + 1
    raw = {"beta_propensity": [], "hydrophobicity": [], "charge_penalty": [],
           "aromatic_bonus": [], "zipper_compatibility": []}
    window_starts = []

    for i in range(n_windows):
        window = sequence[i:i + window_size]
        three = [ONE_TO_THREE[aa] for aa in window]
        beta = float(np.mean([CHOU_FASMAN_BETA[r] for r in three]))
        hydro = float(np.mean([KYTE_DOOLITTLE[r] for r in three]))
        n_charged = sum(1 for r in three if r in POSITIVE or r in NEGATIVE)
        charge_pen = 1.0 - (n_charged / window_size)
        n_aromatic = sum(1 for r in three if r in AROMATIC)
        aromatic = n_aromatic / window_size
        zipper = zipper_compatibility_score(window, reference_motifs)

        window_starts.append(i + 1)  # 1-indexed residue numbering
        raw["beta_propensity"].append(beta)
        raw["hydrophobicity"].append(hydro)
        raw["charge_penalty"].append(charge_pen)
        raw["aromatic_bonus"].append(aromatic)
        raw["zipper_compatibility"].append(zipper)

    normalized = {k: _minmax(np.array(v)) for k, v in raw.items()}
    combined = sum(weights[k] * normalized[k] for k in weights)
    combined = _minmax(combined)  # final combined score also min-max normalized to [0,1]

    records = []
    for idx, start in enumerate(window_starts):
        end = start + window_size - 1
        records.append({
            "window_start": start, "window_end": end,
            "window_seq": sequence[start - 1:end],
            "beta_propensity_raw": round(raw["beta_propensity"][idx], 4),
            "hydrophobicity_raw": round(raw["hydrophobicity"][idx], 4),
            "charge_penalty_raw": round(raw["charge_penalty"][idx], 4),
            "aromatic_bonus_raw": round(raw["aromatic_bonus"][idx], 4),
            "zipper_compatibility_raw": round(raw["zipper_compatibility"][idx], 4),
            "beta_propensity_norm": round(float(normalized["beta_propensity"][idx]), 4),
            "hydrophobicity_norm": round(float(normalized["hydrophobicity"][idx]), 4),
            "charge_penalty_norm": round(float(normalized["charge_penalty"][idx]), 4),
            "aromatic_bonus_norm": round(float(normalized["aromatic_bonus"][idx]), 4),
            "zipper_compatibility_norm": round(float(normalized["zipper_compatibility"][idx]), 4),
            "combined_score": round(float(combined[idx]), 4),
        })

    records.sort(key=lambda r: r["combined_score"], reverse=True)
    for rank, r in enumerate(records, start=1):
        r["rank"] = rank
    records.sort(key=lambda r: r["window_start"])  # restore sequence order for the CSV
    return {"sequence_length": n, "window_size": window_size, "weights": weights, "records": records}
