"""Structured logging to console + results/run.log, and PROGRESS_LOG.md appends."""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from sentinel.utils.config import repo_path

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        results_dir = repo_path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / "run.log"
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%dT%H:%M:%S%z"
        )
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(file_handler)
        root.addHandler(stream_handler)
        _CONFIGURED = True
    return logger


def append_progress_log(module: str, message: str) -> None:
    path = repo_path("PROGRESS_LOG.md")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"- **{timestamp}** `[{module}]` {message}\n"
    with open(path, "a") as fh:
        fh.write(line)
