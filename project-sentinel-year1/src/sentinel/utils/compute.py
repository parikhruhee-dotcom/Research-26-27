"""Compute-tier detection. Never assume a GPU exists; detect it and route work.

Writes results/compute_profile.json once per process on first call, so every
module can log honestly which tier produced its outputs.
"""
from __future__ import annotations

import json
import multiprocessing
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from sentinel.utils.config import repo_path


@dataclass
class ComputeProfile:
    tier: str                 # "GPU_LOCAL" or "CPU"
    cpu_cores: int
    ram_gb: float
    cuda_available: bool
    gpu_name: str | None
    platform: str
    python_version: str
    detected_at: str


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 2)
    except FileNotFoundError:
        pass
    return -1.0


def detect_compute_tier(write: bool = True) -> ComputeProfile:
    cuda_available = False
    gpu_name = None
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        # torch not installed: fall back to nvidia-smi presence as a weaker signal
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0 and out.stdout.strip():
                    cuda_available = True
                    gpu_name = out.stdout.strip().splitlines()[0]
            except Exception:
                pass

    profile = ComputeProfile(
        tier="GPU_LOCAL" if cuda_available else "CPU",
        cpu_cores=multiprocessing.cpu_count(),
        ram_gb=_ram_gb(),
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        platform=platform.platform(),
        python_version=platform.python_version(),
        detected_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if write:
        out_path = repo_path("results", "compute_profile.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(asdict(profile), fh, indent=2)
    return profile


TIER = detect_compute_tier(write=False).tier
