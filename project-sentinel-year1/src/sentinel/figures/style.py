"""Shared plotting style: colorblind-safe palette, consistent fonts/DPI."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from sentinel.utils.config import load_config, repo_path

STRAIN_ORDER = ["AD_PHF", "AD_SF", "CTE_I", "CTE_II", "PiD", "CBD", "PSP", "AGD", "GGT", "GPT"]


def apply_style() -> None:
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.labelsize": 11,
    })


def save_figure(fig, name: str, caption: str, captions_registry: list) -> None:
    cfg = load_config()
    figures_dir = repo_path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for fmt in cfg["figures"]["formats"]:
        fig.savefig(figures_dir / f"{name}.{fmt}", dpi=cfg["figures"]["dpi"], bbox_inches="tight")
    captions_registry.append((name, caption))
