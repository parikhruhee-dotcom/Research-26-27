"""M8 — Conformation-sensitive biosensor concept (diagnostic spin-off).

Computationally proposes a split-reporter biosensor architecture using the
same AD-selective binder scaffold designed in M6, and generates a schematic
figure. This is a concrete, buildable design PROPOSAL — per the brief, it
does not need experimental validation this year (that is Year 3+).

Run: python -m sentinel.biosensor.run_biosensor
"""
from __future__ import annotations

import json

from sentinel.utils.config import repo_path
from sentinel.utils.logging import append_progress_log, get_logger

logger = get_logger(__name__)


def main() -> dict:
    design_dir = repo_path("results", "design")
    fingerprint = json.load(open(repo_path("results", "atlas", "ad_strain_fingerprint.json")))
    top_hotspots = fingerprint["top_hotspots"][:5]

    try:
        leads = list(open(design_dir / "leads.fasta")) if (design_dir / "leads.fasta").exists() else []
        n_leads = sum(1 for line in leads if line.startswith(">"))
    except FileNotFoundError:
        n_leads = 0

    spec = {
        "concept_name": "Split-NanoLuc conformation-sensitive AD-tau biosensor",
        "rationale": (
            "Cryo-EM shows the AD fold packs as a stack of IDENTICAL monomer layers along the "
            "fibril axis (>=2 copies of the same protofilament chain within a few Angstroms of "
            "each other — verified directly in this project's own M1d protofilament clustering, "
            "e.g. 5O3L layer spacing 4.75 A). Two copies of the SAME AD-selective binder, each "
            "fused to a complementary split-reporter half, will therefore be brought into close "
            "proximity (within the reporter's reconstitution distance, typically <10 nm for "
            "NanoBiT/split-luciferase) ONLY when they both dock onto adjacent layers of an actual "
            "AD-fold fibril — not on monomeric tau, not on the other 7 folds (whose templating "
            "tips the binder was explicitly negative-designed against in M6e), and not on a single "
            "isolated binder with no fibril present."
        ),
        "architecture": {
            "reporter_system": "Split NanoLuc (NanoBiT: LgBiT + SmBiT), chosen over split-GFP/FRET "
                                "for its low background and high dynamic range at low analyte "
                                "concentration (relevant for CSF/plasma biomarker detection later "
                                "in the roadmap).",
            "fusion_design": "AD-selective binder (from results/design/leads.fasta) fused at its "
                              "solvent-exposed terminus (the end pointing AWAY from the fibril tip "
                              "in the docked pose, so the fusion does not clash with the binding "
                              "interface) to LgBiT via a flexible (GGGGS)x3 linker; a second copy "
                              "fused to SmBiT via the same linker chemistry.",
            "signal_logic": "Two-binder-bridging AND-gate: luminescence requires BOTH binder copies "
                             "engaging the SAME fibril (adjacent layers) simultaneously. This is a "
                             "built-in specificity check beyond the binder's own AD-selectivity — "
                             "monomeric tau or a single off-target fold cannot satisfy the AND-gate "
                             "even in the unlikely event ONE binder copy binds nonspecifically.",
            "top_hotspot_residues_engaged": [f"{h['res_name']}{h['res_id']}" for h in top_hotspots],
        },
        "n_candidate_binder_scaffolds_available": n_leads,
        "predicted_use_case": "Cell-free or CSF-based rapid AD-strain-specific aggregate detection; "
                                "a diagnostic spin-off distinct from the therapeutic degrader "
                                "program, reusing the same M2 fingerprint + M6 binder without new "
                                "design work.",
        "status": "computational design proposal only — NOT experimentally validated this year "
                   "(see roadmap: Year 3+ wet-lab validation, Part 6 of the brief).",
    }

    out_dir = repo_path("results", "design")
    out_path = out_dir / "biosensor_concept.json"
    with open(out_path, "w") as fh:
        json.dump(spec, fh, indent=2)

    logger.info(f"biosensor concept written: {spec['concept_name']}, engaging hotspots "
                f"{spec['architecture']['top_hotspot_residues_engaged']}")
    append_progress_log(
        "M8",
        "Proposed a split-NanoLuc (NanoBiT) conformation-sensitive biosensor: two copies of the "
        "same M6 AD-selective binder, each fused to a complementary split-luciferase half, "
        "reconstitute signal only when both dock onto adjacent layers of an actual AD fibril — an "
        "AND-gate that adds specificity beyond the binder's own negative design. A concrete, "
        "buildable proposal; not experimentally validated (Year 3+ per the roadmap).",
    )
    return spec


if __name__ == "__main__":
    main()
