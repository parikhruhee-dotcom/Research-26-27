"""Provenance ledger: every external artifact (download URL, PDB ID, checksum,
timestamp, tool version) and every documented tool substitution is logged to
results/PROVENANCE.json so a reader can audit exactly where every byte came from.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentinel.utils.config import repo_path

_PROVENANCE_PATH = repo_path("results", "PROVENANCE.json")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load() -> dict[str, Any]:
    if _PROVENANCE_PATH.exists():
        with open(_PROVENANCE_PATH) as fh:
            return json.load(fh)
    return {
        "created": _now(),
        "downloads": [],
        "tool_versions": {},
        "substitutions": [],
        "seeds": [],
    }


def _save(record: dict[str, Any]) -> None:
    _PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    record["updated"] = _now()
    with open(_PROVENANCE_PATH, "w") as fh:
        json.dump(record, fh, indent=2)


def sha256_of_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_download(url: str, dest_path: Path | str, extra: dict[str, Any] | None = None) -> str:
    """Record a completed download: URL, destination, sha256, timestamp."""
    dest_path = Path(dest_path)
    checksum = sha256_of_file(dest_path) if dest_path.exists() else None
    record = _load()
    entry = {
        "url": url,
        "dest": str(dest_path.relative_to(repo_path())) if dest_path.is_absolute() else str(dest_path),
        "sha256": checksum,
        "timestamp": _now(),
    }
    if extra:
        entry.update(extra)
    record["downloads"].append(entry)
    _save(record)
    return checksum or ""


def log_tool_version(tool: str, version: str) -> None:
    record = _load()
    record["tool_versions"][tool] = version
    _save(record)


def log_substitution(original_tool: str, substitute_tool: str, reason: str) -> None:
    """Document any deviation from the brief's specified tool (M0.3 rule)."""
    record = _load()
    record["substitutions"].append({
        "original": original_tool,
        "substitute": substitute_tool,
        "reason": reason,
        "timestamp": _now(),
    })
    _save(record)


def log_seed(module: str, seed: int) -> None:
    record = _load()
    record["seeds"].append({"module": module, "seed": seed, "timestamp": _now()})
    _save(record)


def log_environment_snapshot() -> None:
    record = _load()
    record["python_version"] = sys.version
    _save(record)
