"""Shared HTTP download helper with retry + provenance logging.

Used by every M1 fetch script. Never crashes the whole pipeline on a single
failed download: retries with backoff, then raises a clearly-labeled error
that the caller can catch to fall back to a documented alternative.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from sentinel.utils.logging import get_logger
from sentinel.utils.provenance import log_download

logger = get_logger(__name__)

USER_AGENT = "project-sentinel-year1/1.0 (autonomous research build; contact: n/a)"


def download(url: str, dest: Path | str, retries: int = 3, backoff_s: float = 2.0,
             extra_provenance: dict | None = None, timeout_s: float = 30.0) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            checksum = log_download(url, dest, extra=extra_provenance)
            logger.info(f"downloaded {url} -> {dest} sha256={checksum[:12]}...")
            return dest
        except Exception as exc:  # noqa: BLE001 - we deliberately catch broadly to retry
            last_exc = exc
            logger.warning(f"download attempt {attempt}/{retries} failed for {url}: {exc}")
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    raise RuntimeError(f"failed to download {url} after {retries} attempts") from last_exc


def get_json(url: str, retries: int = 3, backoff_s: float = 2.0, timeout_s: float = 30.0,
             method: str = "GET", json_body: dict | None = None) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if method == "POST":
                resp = requests.post(url, json=json_body, headers={"User-Agent": USER_AGENT},
                                      timeout=timeout_s)
            else:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(f"API attempt {attempt}/{retries} failed for {url}: {exc}")
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    raise RuntimeError(f"failed to query {url} after {retries} attempts") from last_exc
