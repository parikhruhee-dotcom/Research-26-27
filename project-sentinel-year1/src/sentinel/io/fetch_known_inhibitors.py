"""M1c — validate and log provenance for the curated known-tau-inhibitors table.

Unlike M1a/M1b this is not a single-URL download: it is a small table hand
curated from a web literature search (WebSearch was used to confirm the
Seidler 2018 Nature Chemistry findings before this table was written — see
PROGRESS_LOG.md M1c entry). This script validates the table's schema and
logs it into results/PROVENANCE.json as a literature-curated artifact with an
access date, exactly like a download, so it is equally auditable.

Run: python -m sentinel.io.fetch_known_inhibitors
"""
from __future__ import annotations

import csv

from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger
from sentinel.utils.provenance import log_curated_artifact

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"name", "class", "target_segment", "mechanism", "evidence_level",
                    "citation", "notes"}


def main() -> list[dict]:
    cfg = load_config()
    path = repo_path(cfg["data"]["known_inhibitors_table"])
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    assert not missing, f"known_tau_inhibitors.csv missing columns: {missing}"
    assert len(rows) >= 3, "expected at least 3 curated inhibitor/control entries"

    segments = {r["target_segment"] for r in rows}
    logger.info(f"loaded {len(rows)} curated inhibitor/control entries; target segments: {segments}")

    log_curated_artifact(
        "literature_curation (WebSearch-verified: Seidler et al. 2018 Nature Chemistry "
        "'Structure-based inhibitors of tau aggregation', PMID 29359764)",
        path,
        extra={"n_entries": len(rows)},
    )

    append_progress_log(
        "M1c",
        f"Curated {len(rows)}-entry known-tau-inhibitors table from a literature search "
        f"(Seidler 2018 Nature Chemistry confirmed via WebSearch: VQIINK is the dominant "
        f"nucleator and VQIINK-targeted structure-based inhibitors block full-length-tau seeding "
        f"more effectively than VQIVYK-targeted ones). Exact designed-peptide sequences from the "
        f"primary reference's supplement were deliberately NOT transcribed (avoids mis-transcription "
        f"risk); the table records target segment + mechanism, which is what M9's segment-recovery "
        f"benchmark checks. Broad-spectrum compounds (methylene blue, EGCG) included as negative "
        f"controls for segment-level specificity.",
    )
    return rows


if __name__ == "__main__":
    main()
