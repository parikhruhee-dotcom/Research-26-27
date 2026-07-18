"""Secondary structure via the real DSSP binary (mkdssp, installed via
conda-forge — see PROGRESS_LOG.md M0). Uses biotite's DSSP wrapper, which
shells out to mkdssp and parses its output.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# DSSP 8-state codes -> simplified 3-state
DSSP_TO_SIMPLE = {
    "H": "helix", "G": "helix", "I": "helix",
    "E": "strand", "B": "strand",
    "T": "turn", "S": "turn", "P": "turn", "-": "coil", " ": "coil",
}


def run_dssp(pdb_path: str) -> list[dict]:
    mkdssp = shutil.which("mkdssp")
    if mkdssp is None:
        raise RuntimeError("mkdssp not found on PATH")

    with tempfile.NamedTemporaryFile(suffix=".dssp", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [mkdssp, str(pdb_path), str(out_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"mkdssp failed on {pdb_path}: {result.stderr[:500]}")
        return _parse_dssp(out_path)
    finally:
        out_path.unlink(missing_ok=True)


def _parse_dssp(dssp_path: Path) -> list[dict]:
    lines = dssp_path.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("  #  RESIDUE"):
            start = i + 1
            break
    if start is None:
        raise RuntimeError(f"could not find DSSP data table in {dssp_path}")

    records = []
    for line in lines[start:]:
        if len(line) < 17 or line[13] == "!":
            continue
        res_id = line[5:10].strip()
        chain = line[11].strip()
        aa = line[13]
        ss = line[16] if line[16] != " " else "-"
        try:
            res_id_int = int(res_id)
        except ValueError:
            continue
        records.append({
            "chain": chain, "res_id": res_id_int, "res_aa": aa,
            "dssp_code": ss, "ss_simple": DSSP_TO_SIMPLE.get(ss, "coil"),
        })
    return records


def beta_strand_count(dssp_records: list[dict], chain: str | None = None) -> int:
    """Count contiguous beta-strand segments (a simple run-length count over
    ss_simple=='strand'), optionally restricted to one chain."""
    recs = [r for r in dssp_records if chain is None or r["chain"] == chain]
    recs.sort(key=lambda r: r["res_id"])
    count, in_strand = 0, False
    for r in recs:
        is_strand = r["ss_simple"] == "strand"
        if is_strand and not in_strand:
            count += 1
        in_strand = is_strand
    return count
