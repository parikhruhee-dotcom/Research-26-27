"""THE STRAIN FINGERPRINT — Contribution #1's key deliverable.

Tau is the same protein (same sequence) in every tauopathy: the fibril
"strains" differ only in how that one sequence folds and packs, so the
selectivity handle cannot come from sequence differences — it has to come
from *differential conformational exposure*. For every residue in the shared
fibril core (306-378), we compute how exposed it is at the AD templating tip
versus the mean exposure of the equivalent residue (same numbering — tau
folds are all built from the same chain, so residue-by-residue comparison at
identical sequence positions is directly meaningful, no structural-alignment-
based re-indexing needed) across the other seven folds. A large positive
value means: "this residue is uniquely exposed and available to a binder in
the AD fold, but buried/inaccessible in most other folds" — the selectivity
handle.
"""
from __future__ import annotations

from sentinel.atlas.physicochemical import residue_physchem
from sentinel.utils.config import load_config


def compute_fingerprint(per_strain_tip_sasa: dict[str, dict[int, dict]],
                          reference_strain: str, boundary_exclude_resids: set[int] | None = None) -> dict:
    """per_strain_tip_sasa: {strain_name: {res_id: {res_name, rel_sasa}}}

    boundary_exclude_resids: residues within a few positions of the AD chain's
    OWN modeled N-/C-terminus. A truncated single-chain model always shows
    inflated apparent SASA right at its own cut ends (no neighboring residue
    there to occlude it) — that is a modeling artifact of this specific chain,
    not a real fold property, so those positions are excluded from the AD
    hotspot ranking (they are NOT excluded from the "other folds" side of the
    comparison, since for most other folds those same sequence positions are
    interior, not boundary, residues).
    """
    cfg = load_config()
    if reference_strain not in per_strain_tip_sasa:
        raise ValueError(f"reference strain {reference_strain!r} missing from atlas results")
    boundary_exclude_resids = boundary_exclude_resids or set()

    ref = per_strain_tip_sasa[reference_strain]
    others = {s: d for s, d in per_strain_tip_sasa.items() if s != reference_strain}

    all_resids = sorted(ref.keys())
    hotspots = []
    excluded = []
    for res_id in all_resids:
        ref_rec = ref[res_id]
        if res_id in boundary_exclude_resids:
            excluded.append(res_id)
            continue
        ref_rsasa = ref_rec["rel_sasa"]
        if ref_rsasa is None:
            continue
        other_vals = [others[s][res_id]["rel_sasa"] for s in others
                       if res_id in others[s] and others[s][res_id]["rel_sasa"] is not None]
        if not other_vals:
            continue
        mean_other = sum(other_vals) / len(other_vals)
        differential = ref_rsasa - mean_other
        physchem = residue_physchem(ref_rec["res_name"])
        hotspots.append({
            "res_id": res_id, "res_name": ref_rec["res_name"], "one_letter": physchem["one_letter"],
            "ad_rel_sasa": round(ref_rsasa, 4), "mean_other_folds_rel_sasa": round(mean_other, 4),
            "n_other_folds_compared": len(other_vals),
            "differential_exposure": round(differential, 4),
            "hydrophobicity_kd": physchem["hydrophobicity_kd"],
            "charge_class": physchem["charge_class"], "is_aromatic": physchem["is_aromatic"],
        })

    hotspots.sort(key=lambda h: h["differential_exposure"], reverse=True)
    top_n = cfg["atlas"]["fingerprint_top_n_hotspots"]
    for rank, h in enumerate(hotspots, start=1):
        h["rank"] = rank

    return {
        "reference_strain": reference_strain,
        "compared_against": sorted(others.keys()),
        "method": "per-residue relative-SASA differential at the fibril templating tip: "
                  "ad_rel_sasa - mean(other_folds_rel_sasa), same tau sequence numbering "
                  "throughout so no structural re-indexing is needed for this comparison.",
        "boundary_residues_excluded_from_ad_ranking": sorted(excluded),
        "boundary_exclusion_reason": "positions within a few residues of the AD single-chain "
                  "model's own modeled N-/C-terminus show inflated apparent SASA due to chain "
                  "truncation, not real fold geometry; excluded from the AD side of the ranking only.",
        "all_residues": hotspots,
        "top_hotspots": hotspots[:top_n],
    }
