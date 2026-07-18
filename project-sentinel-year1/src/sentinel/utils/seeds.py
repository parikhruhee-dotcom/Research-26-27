"""Global determinism. Call set_global_seed() at the start of every entry point."""
from __future__ import annotations

import random

import numpy as np

from sentinel.utils.config import load_config

_SEEDED = False


def set_global_seed(seed: int | None = None) -> int:
    global _SEEDED
    if seed is None:
        seed = load_config()["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    _SEEDED = True
    return seed
