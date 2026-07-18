"""M1a — fetch canonical human tau (2N4R, UniProt P10636) and derive landmark
residue records (PHF6*, PHF6, R1-R4, K18, K19) from config.

Run: python -m sentinel.io.fetch_sequence
"""
from __future__ import annotations

import json

from sentinel.io.http_client import download
from sentinel.utils.config import load_config, repo_path
from sentinel.utils.logging import append_progress_log, get_logger

logger = get_logger(__name__)


def parse_fasta(path) -> tuple[str, str]:
    header, seq_lines = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                header = line[1:]
            elif line:
                seq_lines.append(line)
    return header, "".join(seq_lines)


def build_aggregation_constructs(full_seq: str, landmarks: dict) -> dict[str, str]:
    """K18 = the 4R microtubule-binding domain (R1-R4, 244-372); K19 = the same
    span with R2 excised (3R construct), the two standard in vitro tau
    aggregation constructs used throughout the literature."""
    k18_spec = landmarks["K18"]
    k18_seq = full_seq[k18_spec["start"] - 1:k18_spec["end"]]

    k19_spec = landmarks["K19"]
    excl_start, excl_end = k19_spec["exclude"]
    k19_seq = (full_seq[k19_spec["start"] - 1:excl_start - 1] +
               full_seq[excl_end:k19_spec["end"]])
    return {"K18_4R": k18_seq, "K19_3R": k19_seq}


def main() -> dict:
    cfg = load_config()
    uni = cfg["data"]["uniprot"]
    dest = repo_path("data", "raw", "tau_P10636-8_2N4R.fasta")
    download(uni["fasta_url"], dest, extra_provenance={
        "kind": "uniprot_fasta", "accession": uni["isoform_accession"], "isoform": uni["isoform"],
    })

    header, seq = parse_fasta(dest)
    expected_len = uni["length_aa"]
    if len(seq) != expected_len:
        logger.warning(
            f"downloaded tau sequence length {len(seq)} != expected {expected_len} "
            f"(config may be stale — UniProt canonical entry can be re-annotated). Proceeding "
            f"with the downloaded sequence as ground truth."
        )

    landmarks = {}
    for name, spec in cfg["data"]["landmarks"].items():
        if "sequence" in spec:
            start, end = spec["start"], spec["end"]
            observed = seq[start - 1:end]
            match = observed == spec["sequence"]
            if not match:
                logger.error(
                    f"landmark {name} expected {spec['sequence']} at {start}-{end}, "
                    f"observed {observed} — numbering mismatch, must be fixed before proceeding"
                )
            landmarks[name] = {**spec, "observed_sequence": observed, "verified": match}
        else:
            landmarks[name] = dict(spec)

    out = {
        "uniprot_accession": uni["accession"],
        "isoform": uni["isoform"],
        "header": header,
        "length_aa": len(seq),
        "sequence": seq,
        "landmarks": landmarks,
    }
    out_path = repo_path("data", "interim", "tau_sequence.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    constructs = build_aggregation_constructs(seq, cfg["data"]["landmarks"])
    constructs_fasta = repo_path("data", "interim", "aggregation_constructs.fasta")
    with open(constructs_fasta, "w") as fh:
        for name, cseq in constructs.items():
            fh.write(f">{name} (derived in silico from P10636-8 2N4R, {len(cseq)} aa)\n{cseq}\n")
    logger.info(f"generated in silico constructs: " +
                ", ".join(f"{n}={len(s)}aa" for n, s in constructs.items()))

    verified_ok = all(v.get("verified", True) for v in landmarks.values())
    logger.info(f"tau 2N4R sequence fetched: {len(seq)} aa, landmarks verified={verified_ok}")
    append_progress_log(
        "M1a",
        f"Fetched tau P10636 ({len(seq)} aa) from UniProt REST API; "
        f"PHF6/PHF6* and repeat-region landmarks {'verified' if verified_ok else 'FAILED VERIFICATION'} "
        f"against the downloaded sequence.",
    )
    assert verified_ok, "landmark residues do not match downloaded sequence — fix config numbering"
    return out


if __name__ == "__main__":
    main()
