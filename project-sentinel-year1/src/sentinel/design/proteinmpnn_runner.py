"""Thin wrapper around the real, locally-installed ProteinMPNN (dauparas/
ProteinMPNN), invoked as a subprocess exactly as its own CLI intends
(protein_mpnn_run.py). Model weights ship inside the cloned repo (small,
~6.7 MB per checkpoint) — no separate download needed. Runs on CPU in a few
seconds for a mini-protein-length backbone (measured on this machine: ~1.2
s/sequence for a 106-residue backbone — see PROGRESS_LOG.md M6).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


def find_proteinmpnn_repo() -> Path:
    candidates = [Path("/tmp/ProteinMPNN"), Path.home() / "ProteinMPNN"]
    for c in candidates:
        if (c / "protein_mpnn_run.py").exists():
            return c
    raise FileNotFoundError(
        "ProteinMPNN repo not found. Clone it: git clone --depth 1 "
        "https://github.com/dauparas/ProteinMPNN.git /tmp/ProteinMPNN"
    )


def run_proteinmpnn(pdb_path: str, out_folder: str, num_sequences: int, temperature: float,
                      seed: int, chain: str = "A") -> list[dict]:
    repo = find_proteinmpnn_repo()
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", str(repo / "protein_mpnn_run.py"),
        "--pdb_path", str(pdb_path), "--pdb_path_chains", chain,
        "--out_folder", str(out_folder), "--num_seq_per_target", str(num_sequences),
        "--sampling_temp", str(temperature), "--seed", str(seed), "--batch_size", "1",
    ]
    result = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ProteinMPNN failed on {pdb_path}: {result.stderr[-2000:]}")

    stem = Path(pdb_path).stem
    fasta_path = out_folder / "seqs" / f"{stem}.fa"
    return _parse_mpnn_fasta(fasta_path)


def _parse_mpnn_fasta(fasta_path: Path) -> list[dict]:
    text = fasta_path.read_text()
    entries = text.strip().split(">")[1:]
    records = []
    for i, entry in enumerate(entries):
        if i == 0:
            continue  # first entry is the original/input sequence record, not a design
        header, seq = entry.split("\n", 1)
        seq = seq.strip()
        fields = {}
        for part in header.split(", "):
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.strip()] = v.strip()
        records.append({"sequence": seq, "mpnn_score": float(fields.get("score", "nan")),
                         "seq_recovery": float(fields.get("seq_recovery", "nan"))})
    return records
