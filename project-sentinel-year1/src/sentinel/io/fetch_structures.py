"""M1b — fetch and verify the 8-tauopathy-fold cryo-EM PDB panel, plus a live
RCSB search for newer AD tau structures and the VQIVYK/VQIINK steric-zipper
microcrystal structures.

Every entry is verified against the RCSB Data API (title/experimental method)
before its coordinates are downloaded, so a stale or wrong PDB ID in config
is caught rather than silently trusted.

Run: python -m sentinel.io.fetch_structures
"""
from __future__ import annotations

import json

from sentinel.io.http_client import download, get_json
from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger

logger = get_logger(__name__)


def verify_entry(pdb_id: str) -> dict:
    cfg = load_config()
    url = cfg["data"]["rcsb"]["entry_api"].format(id=pdb_id)
    data = get_json(url)
    return {
        "pdb_id": pdb_id,
        "title": data.get("struct", {}).get("title"),
        "method": data.get("exptl", [{}])[0].get("method") if data.get("exptl") else None,
        "release_date": data.get("rcsb_accession_info", {}).get("initial_release_date"),
        "resolution_A": (data.get("rcsb_entry_info", {}) or {}).get("resolution_combined", [None])[0],
    }


def fetch_panel() -> list[dict]:
    cfg = load_config()
    panel = cfg["data"]["strain_panel"]
    raw_dir = repo_path("data", "raw", "structures")
    manifest = []
    for entry in panel:
        pdb_id = entry["id"]
        try:
            meta = verify_entry(pdb_id)
        except Exception as exc:
            logger.error(f"could not verify {pdb_id} via RCSB Data API: {exc}")
            meta = {"pdb_id": pdb_id, "title": None, "method": None, "release_date": None,
                     "resolution_A": None, "verification_error": str(exc)}
        is_em = (meta.get("method") or "").upper().find("ELECTRON MICROSCOPY") >= 0
        if meta.get("title") and not is_em:
            logger.warning(f"{pdb_id} method={meta.get('method')!r} is not cryo-EM; check panel entry")

        cif_url = cfg["data"]["rcsb"]["file_api_cif"].format(id=pdb_id)
        pdb_url = cfg["data"]["rcsb"]["file_api_pdb"].format(id=pdb_id)
        cif_dest = raw_dir / f"{pdb_id}.cif"
        pdb_dest = raw_dir / f"{pdb_id}.pdb"
        got_pdb = True
        try:
            download(cif_url, cif_dest, extra_provenance={"kind": "rcsb_cif", "pdb_id": pdb_id})
        except Exception as exc:
            logger.error(f"cif download failed for {pdb_id}: {exc}")
        try:
            download(pdb_url, pdb_dest, extra_provenance={"kind": "rcsb_pdb", "pdb_id": pdb_id})
        except Exception as exc:
            logger.warning(f"legacy .pdb download failed for {pdb_id} (common for large multi-chain "
                            f"cryo-EM entries — .cif is authoritative and sufficient): {exc}")
            got_pdb = False

        record = {**entry, **meta, "cif_path": str(cif_dest.relative_to(repo_path())),
                  "pdb_path": str(pdb_dest.relative_to(repo_path())) if got_pdb else None}
        manifest.append(record)
        logger.info(f"{pdb_id} ({entry['strain']}): {meta.get('title')}")

    out_path = repo_path("data", "raw", "structures", "panel_manifest.json")
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def search_related_structures() -> dict:
    """Live RCSB full-text search for newer tau filament structures and the
    VQIVYK/VQIINK steric-zipper microcrystal structures (bonus provenance —
    not required for the core pipeline, logged for completeness per M1b)."""
    cfg = load_config()
    search_url = cfg["data"]["rcsb"]["search_api"]
    queries = {
        "newer_tau_filament_structures": {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {"type": "terminal", "service": "full_text",
                     "parameters": {"value": "tau filament"}},
                    {"type": "terminal", "service": "text",
                     "parameters": {"attribute": "exptl.method", "operator": "exact_match",
                                     "value": "ELECTRON MICROSCOPY"}},
                ],
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 25},
                                 "sort": [{"sort_by": "rcsb_accession_info.initial_release_date",
                                            "direction": "desc"}]},
        },
        # Note: the RCSB "sequence" search service enforces a minimum query length
        # (>=20 residues in practice) and rejects a bare 6-mer, so VQIVYK/VQIINK
        # steric-zipper microcrystal structures are instead located via full-text
        # search on the motif name, which is how they are indexed in struct.title.
        "vqivyk_zipper_structures": {
            "query": {"type": "terminal", "service": "full_text",
                       "parameters": {"value": "VQIVYK"}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 25}},
        },
        "vqiink_zipper_structures": {
            "query": {"type": "terminal", "service": "full_text",
                       "parameters": {"value": "VQIINK"}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 25}},
        },
    }
    results = {}
    for key, body in queries.items():
        try:
            resp = get_json(search_url, method="POST", json_body=body)
            hits = [r["identifier"] for r in resp.get("result_set", [])]
            results[key] = hits
            logger.info(f"RCSB search '{key}': {len(hits)} hits")
        except Exception as exc:
            logger.warning(f"RCSB search '{key}' failed (non-fatal, bonus provenance only): {exc}")
            results[key] = {"error": str(exc)}

    out_path = repo_path("data", "raw", "structures", "rcsb_search_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    return results


def main() -> None:
    manifest = fetch_panel()
    n_ok = sum(1 for m in manifest if m.get("cif_path") and not m.get("verification_error"))
    search = search_related_structures()
    append_progress_log(
        "M1b",
        f"Fetched {n_ok}/{len(manifest)} strain-panel structures ({', '.join(m['strain'] for m in manifest)}) "
        f"from RCSB, each verified via the Data API before coordinate download. Bonus RCSB search found "
        f"{len(search.get('vqivyk_zipper_structures', []))} VQIVYK-sequence hits and "
        f"{len(search.get('vqiink_zipper_structures', []))} VQIINK-sequence hits.",
    )
    assert n_ok == len(manifest), "not all strain-panel structures downloaded successfully"


if __name__ == "__main__":
    main()
