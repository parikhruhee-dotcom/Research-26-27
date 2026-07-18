"""Central config loader. Code must read parameters from config/config.yaml
via this module rather than hardcoding values."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    with open(p) as fh:
        return yaml.safe_load(fh)


def repo_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return REPO_ROOT.joinpath(*parts)
