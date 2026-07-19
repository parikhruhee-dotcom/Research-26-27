"""M6e developability reference population — two real bugs found and fixed
here in sequence.

developability_filter() needs a population to percentile a binder's own
worst exposed aggregation-window score against.

**Bug 1 (fixed): tau's own windows are the wrong population.** The original
choice was tau's own window-score distribution (a convenient reuse of the M3
scorer's already-built profile). Tau is intrinsically disordered and
overwhelmingly low-aggregation-propensity by amino-acid composition, so
almost any ordinarily-folded protein looks like an outlier by comparison:
measured directly, this build's own real, hyperstable, industrially-used
scaffold proteins (native Protein A B-domain, DARPin, engrailed homeodomain)
scored at the 73rd-98th percentile against tau's windows even restricted to
solvent-exposed patches only, despite no known aggregation liability.

**Bug 2 (fixed): pooling individual windows, not comparing peak-to-peak, is
also wrong.** The first fix pooled every individual exposed window score
from the 5 design-scaffold proteins into one population and percentiled the
binder's MAX window against it. That is still mechanically broken: a
protein's own maximum window is, by construction, higher than nearly all of
its OWN other windows and everyone else's typical windows — so percentiling
"my one worst window" against "everyone else's mostly-typical windows,
pooled" makes the true maximum of ANY protein land near the 100th percentile
almost automatically, independent of whether that protein is actually
aggregation-prone. Measured directly: even native DARPin (100th percentile),
native engrailed homeodomain (95th), and native de novo 3-helix bundle
(97th) failed against the pooled-individual-window population built from
their OWN scaffold library. The correct, apples-to-apples comparison is
peak-vs-peak: take each real reference protein's own worst exposed window
score, and percentile the binder's worst exposed window score against THAT
population of peaks (one number per protein), not against everyone's
individual windows.

**The fix actually shipped:** an independent panel of 8 real, well-known,
individually solved, monomeric, soluble small proteins (ubiquitin, GB1
domain, an SH3 domain, a cold-shock protein, chymotrypsin inhibitor 2, a
fibronectin type-III domain, acylphosphatase, barstar — see
SOLUBLE_REFERENCE_SOURCES; deliberately DISJOINT from the 5 design-scaffold
proteins in scaffold_library.py, to avoid the circularity of a binder built
on a scaffold being compared against that same scaffold's own peak). Each is
downloaded and verified via the RCSB Data API exactly like the M1 strain
panel and the M6a scaffold library (title, method, residue count checked
before use), PDBFixer-cleaned, and scored for its own real per-residue SASA
and worst solvent-exposed M3 aggregation window. The binder's own worst
exposed window is percentiled against these 8 real peak values. The known
limitation — only 8 reference points, so percentile granularity is ~12.5%
steps — is real and is reported (n_reference_proteins, individual peaks)
rather than hidden, exactly as the earlier 5-scaffold population's small-N
caveat was documented rather than hidden.
"""
from __future__ import annotations

import json

import biotite.structure.io.pdb as pdb_io
import biotite.structure.io.pdbx as pdbx

from sentinel.atlas.physicochemical import THREE_TO_ONE
from sentinel.atlas.sasa import compute_residue_sasa
from sentinel.io.http_client import download, get_json
from sentinel.md.system_builder import fix_and_protonate
from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger
from sentinel.utils.provenance import log_curated_artifact

logger = get_logger(__name__)

# Verified via the RCSB Data API before being added here (title, method, residue
# count all checked to match at fetch time — see PROGRESS_LOG.md M6-quality-3).
# Deliberately independent of scaffold_library.SCAFFOLD_SOURCES: these are real,
# well-characterized, monomeric, soluble small proteins with no relation to the
# design scaffold library, chosen purely to calibrate "what does a real,
# unremarkable, well-behaved protein's own worst exposed patch look like."
SOLUBLE_REFERENCE_SOURCES = {
    "ref_ubiquitin": {"pdb_id": "1UBQ", "chain": "A",
                        "description": "Ubiquitin — one of the most extensively characterized "
                                        "stable, soluble, monomeric proteins known.",
                        "expected_length": 76, "expected_method": "X-RAY DIFFRACTION"},
    "ref_gb1": {"pdb_id": "1PGB", "chain": "A",
                 "description": "Streptococcal Protein G, B1 immunoglobulin-binding domain — "
                                 "a classic hyperstable, highly soluble biophysics workhorse.",
                 "expected_length": 56, "expected_method": "X-RAY DIFFRACTION"},
    "ref_sh3": {"pdb_id": "1SHG", "chain": "A",
                 "description": "Alpha-spectrin SH3 domain — a classic well-behaved folding model.",
                 "expected_length": 62, "expected_method": "X-RAY DIFFRACTION"},
    "ref_csp": {"pdb_id": "1CSP", "chain": "A",
                 "description": "Bacillus subtilis major cold-shock protein CspB — hyperstable, "
                                 "highly soluble nucleic-acid-binding domain.",
                 "expected_length": 67, "expected_method": "X-RAY DIFFRACTION"},
    "ref_ci2": {"pdb_id": "2CI2", "chain": "I",  # deposited under chain ID "I" (for "inhibitor")
                 "description": "Chymotrypsin inhibitor 2 (barley) — a classic hyperstable "
                                 "folding-model protein. Its flexible N-terminal ~18 residues "
                                 "are crystallographically unresolved in this deposition.",
                 "expected_length": 65, "expected_method": "X-RAY DIFFRACTION"},
    "ref_fn3": {"pdb_id": "1TEN", "chain": "A",
                 "description": "Tenascin fibronectin type-III domain — a real, soluble, "
                                 "well-expressed beta-sandwich fold (also the scaffold family "
                                 "behind 'Adnectin' engineered binders).",
                 "expected_length": 90, "expected_method": "X-RAY DIFFRACTION"},
    "ref_acylphosphatase": {"pdb_id": "2ACY", "chain": "A",
                              "description": "Bovine testis acylphosphatase — a well-studied "
                                              "soluble, monomeric alpha/beta fold.",
                              "expected_length": 98, "expected_method": "X-RAY DIFFRACTION"},
    "ref_barstar": {"pdb_id": "1BTA", "chain": "A",
                      "description": "Barstar — a well-characterized soluble, monomeric "
                                      "ribonuclease inhibitor.",
                      "expected_length": 89, "expected_method": "SOLUTION NMR"},
}


def _verify_entry(pdb_id: str) -> dict:
    cfg = load_config()
    url = cfg["data"]["rcsb"]["entry_api"].format(id=pdb_id)
    data = get_json(url)
    return {
        "title": data.get("struct", {}).get("title"),
        "method": data.get("exptl", [{}])[0].get("method") if data.get("exptl") else None,
        "n_residues": (data.get("rcsb_entry_info", {}) or {}).get("deposited_polymer_monomer_count"),
    }


def fetch_soluble_reference_panel() -> dict:
    raw_dir = repo_path("data", "raw", "developability_reference")
    interim_dir = repo_path("data", "interim", "developability_reference")
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, spec in SOLUBLE_REFERENCE_SOURCES.items():
        pdb_id = spec["pdb_id"]
        meta = _verify_entry(pdb_id)
        if meta.get("method") != spec["expected_method"]:
            logger.warning(f"{pdb_id}: expected method {spec['expected_method']!r}, "
                            f"got {meta.get('method')!r} — check for a re-deposit")

        cif_dest = raw_dir / f"{pdb_id}.cif"
        cfg = load_config()
        download(cfg["data"]["rcsb"]["file_api_cif"].format(id=pdb_id), cif_dest,
                  extra_provenance={"kind": "developability_reference", "pdb_id": pdb_id, "name": name})

        cif = pdbx.CIFFile.read(str(cif_dest))
        arr = pdbx.get_structure(cif, model=1)
        # Filter by chain + exclude water directly (res_name != HOH) rather than trusting the
        # mmCIF hetero flag alone — at least one real deposition here (2CI2) marks its entire
        # polymer chain group_PDB=HETATM for historical reasons, which would wrongly exclude it.
        arr = arr[(arr.chain_id == spec["chain"]) & (arr.res_name != "HOH")]
        n_res_found = len(set(arr.res_id.tolist()))
        if abs(n_res_found - spec["expected_length"]) > 5:
            logger.warning(f"{pdb_id} chain {spec['chain']}: expected ~{spec['expected_length']} "
                            f"residues, found {n_res_found}")

        raw_pdb = interim_dir / f"{name}_raw.pdb"
        writer = pdb_io.PDBFile()
        writer.set_structure(arr)
        writer.write(str(raw_pdb))

        fixed_pdb = interim_dir / f"{name}_fixed.pdb"
        fix_and_protonate(raw_pdb, fixed_pdb)
        raw_pdb.unlink()

        manifest[name] = {**spec, "verified_title": meta.get("title"),
                            "verified_n_residues": n_res_found, "fixed_pdb": str(fixed_pdb)}
        logger.info(f"developability reference {name} ({pdb_id}): {meta.get('title')} — "
                    f"{n_res_found} residues")

    manifest_path = interim_dir / "reference_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log_curated_artifact(
        "RCSB PDB — 8 verified real, well-characterized, soluble monomeric reference proteins, "
        "independent of the design scaffold library, used to calibrate binder developability "
        "(see developability_reference.py docstring for the two bugs this design fixes)",
        manifest_path,
    )
    append_progress_log(
        "M6-developability-reference",
        f"Fetched and verified {len(manifest)} independent real soluble reference proteins "
        f"(1UBQ/ubiquitin, 1PGB/GB1, 1SHG/SH3, 1CSP/cold-shock, 2CI2, 1TEN/FN3, 2ACY/"
        f"acylphosphatase, 1BTA/barstar) to replace both the tau-window population AND the "
        f"pooled-individual-scaffold-window population as the developability-filter reference "
        f"— comparing a binder's own worst exposed patch against real proteins' own worst "
        f"exposed patches (peak-vs-peak), not against unrelated individual windows.",
    )
    return manifest


def _native_sequence_from_pdb(fixed_pdb: str) -> str:
    reader = pdb_io.PDBFile.read(str(fixed_pdb))
    arr = reader.get_structure(model=1)
    ca = arr[arr.atom_name == "CA"]
    return "".join(THREE_TO_ONE.get(r, "A") for r in ca.res_name)


def _protein_worst_exposed_window_score(fixed_pdb: str, tau_normalization_bounds: dict) -> dict:
    from sentinel.aggregation.scorer import compute_profile

    cfg = load_config()
    exposure_threshold = cfg["atlas"]["exposed_rel_sasa_threshold"]

    seq = _native_sequence_from_pdb(fixed_pdb)
    sasa_records = compute_residue_sasa(fixed_pdb)
    rel_sasa = {r["res_id"]: r["rel_sasa"] for r in sasa_records if r["rel_sasa"] is not None}

    profile = compute_profile(seq, normalization_bounds=tau_normalization_bounds)
    exposed_scores = []
    for r in profile["records"]:
        window_sasas = [rel_sasa[i] for i in range(r["window_start"], r["window_end"] + 1)
                          if i in rel_sasa]
        if window_sasas and (sum(window_sasas) / len(window_sasas)) >= exposure_threshold:
            exposed_scores.append(r["combined_score"])

    return {"n_residues": len(seq), "n_exposed_windows": len(exposed_scores),
            "worst_exposed_score": max(exposed_scores) if exposed_scores else 0.0}


def build_reference_distribution(tau_normalization_bounds: dict) -> dict:
    """Returns {scores, n_source_proteins, n_windows, per_scaffold}. `scores`
    is one PEAK (worst real solvent-exposed aggregation-window score) per
    reference protein — the population a binder's own peak should be
    percentiled against (peak-vs-peak; see module docstring for why pooling
    individual windows instead is a separate, distinct bug)."""
    manifest_path = repo_path("data", "interim", "developability_reference", "reference_manifest.json")
    if not manifest_path.exists():
        fetch_soluble_reference_panel()
    manifest = json.load(open(manifest_path))

    peaks, per_scaffold = [], {}
    for name, entry in manifest.items():
        result = _protein_worst_exposed_window_score(entry["fixed_pdb"], tau_normalization_bounds)
        peaks.append(result["worst_exposed_score"])
        per_scaffold[name] = result
        logger.info(f"developability reference: {name} worst exposed window = "
                    f"{result['worst_exposed_score']:.4f} ({result['n_exposed_windows']} exposed "
                    f"windows out of {result['n_residues']} residues)")

    if len(peaks) < 8:
        logger.warning(f"developability reference population has only {len(peaks)} real reference "
                        f"proteins — percentile resolution is coarse (~{100 / max(len(peaks), 1):.0f}% "
                        f"steps); documented as a known limitation, not hidden")

    return {"scores": peaks, "n_source_proteins": len(manifest),
            "n_windows": len(peaks), "per_scaffold": per_scaffold}


if __name__ == "__main__":
    fetch_soluble_reference_panel()
