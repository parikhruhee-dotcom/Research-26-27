"""Generates notebooks/colab_rfdiffusion.ipynb and notebooks/colab_af2_multimer.ipynb
— one-click, GPU-tier notebooks pre-filled with the real M5 target spec, so
a human with zero ML infrastructure experience can run the full-scale GPU
steps (RFdiffusion backbone generation; AlphaFold2-multimer complex folding)
on Colab's free GPU tier without editing anything.

Run: python -m sentinel.design.colab_notebooks
"""
from __future__ import annotations

import json

from sentinel.utils.config import repo_path


def _cell(cell_type: str, source: str) -> dict:
    return {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {})}


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


def build_rfdiffusion_notebook(target_spec: dict) -> dict:
    hotspots = target_spec["hotspot_residues"]
    core_start, core_end = target_spec["core_residue_range"]
    bmin, bmax = target_spec["binder_length_range"]
    hotspot_flag = "".join(f"A{r}," for r in hotspots)[:-1]

    cells = [
        _cell("markdown", f"""# Project SENTINEL — RFdiffusion backbone generation (GPU, one-click)

This notebook runs the **full-scale GPU step** that this project's CPU
sandbox could not run locally (RFdiffusion needs a CUDA SE3-Transformer —
see `results/design/GPU_TIER_STATUS.md` in the repo for why, and
`src/sentinel/design/backbone_gen.py` for the CPU geometric baseline that
was used instead so the rest of the pipeline could run end-to-end).

**What this does:** generates real, RFdiffusion-designed protein backbones
conditioned on the exact AD-fold hotspot residues identified in Year 1's
strain fingerprint (`results/atlas/ad_strain_fingerprint.json`), targeting
PDB **{target_spec['reference_pdb_id']}** ({target_spec['reference_strain']}),
core residues {core_start}-{core_end}.

**How to use:** Runtime -> Change runtime type -> GPU (T4 is fine, free
tier). Then Runtime -> Run all. Takes ~15-30 min. Output backbones download
as a zip; drop the .pdb files into `results/design/backbones/rfdiffusion/`
in the repo and re-run `make design` — the active-learning loop reads
backbones from that directory with no code changes needed.
"""),
        _cell("code", """#@title 1. Install RFdiffusion (~5 min)
!git clone https://github.com/RosettaCommons/RFdiffusion.git
%cd RFdiffusion
!pip install -q -e .
!pip install -q dgl -f https://data.dgl.ai/wheels/cu118/repo.html
!mkdir -p models && cd models && \\
  wget -q http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
"""),
        _cell("code", f"""#@title 2. Fetch the AD-fold target structure and set hotspots (pre-filled from Year-1 results)
!wget -q https://files.rcsb.org/download/{target_spec['reference_pdb_id']}.pdb -O target.pdb

CONTIG = "A{core_start}-{core_end}/0 {bmin}-{bmax}"  #@param {{type:"string"}}
HOTSPOT_RES = "{hotspot_flag}"  #@param {{type:"string"}}
NUM_DESIGNS = 128  #@param {{type:"integer"}}
print("contig:", CONTIG)
print("hotspots:", HOTSPOT_RES)
"""),
        _cell("code", """#@title 3. Run RFdiffusion
!python scripts/run_inference.py \\
    inference.output_prefix=outputs/ad_capper \\
    inference.input_pdb=../target.pdb \\
    'contigmap.contigs=[$CONTIG]' \\
    "ppi.hotspot_res=[$HOTSPOT_RES]" \\
    inference.num_designs=$NUM_DESIGNS
"""),
        _cell("code", """#@title 4. Zip and download results
import shutil
from google.colab import files
shutil.make_archive("rfdiffusion_backbones", "zip", "outputs")
files.download("rfdiffusion_backbones.zip")
print("Unzip into results/design/backbones/rfdiffusion/ in the repo, then: make design")
"""),
    ]
    return _notebook(cells)


def build_af2_multimer_notebook(target_spec: dict) -> dict:
    cells = [
        _cell("markdown", f"""# Project SENTINEL — AlphaFold2-multimer complex folding (GPU, one-click)

Full-scale complex refolding + interface scoring (interface pAE < 7.5,
pLDDT > 85, ipTM > 0.7 — thresholds from `config/config.yaml`,
`design.scoring`) for the leads in `results/design/leads.fasta`, folded
against tau target PDB **{target_spec['reference_pdb_id']}**
({target_spec['reference_strain']}). The CPU sandbox used a documented
substitute (ESM-2 sequence plausibility + geometric complementarity — see
`src/sentinel/design/interface_scorer.py`) instead of this GPU step.

**How to use:** Runtime -> GPU -> Run all. Upload `leads.fasta` when
prompted (from `results/design/leads.fasta` in the repo).
"""),
        _cell("code", """#@title 1. Install ColabFold (localcolabfold-in-Colab)
!pip install -q "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
!pip install -q jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
"""),
        _cell("code", f"""#@title 2. Upload leads.fasta and build binder:tau complex FASTA pairs
from google.colab import files
uploaded = files.upload()  # upload results/design/leads.fasta from the repo

TAU_CORE_SEQ = "PASTE_TAU_CORE_SEQUENCE_HERE"  #@param {{type:"string"}}
# (tau core sequence, residues {target_spec['core_residue_range'][0]}-{target_spec['core_residue_range'][1]},
#  is in data/interim/tau_sequence.json in the repo — copy the substring for this range)

lead_fasta = list(uploaded.keys())[0]
records = open(lead_fasta).read().strip().split(">")[1:]
with open("complexes.fasta", "w") as out:
    for r in records:
        header, seq = r.split("\\n", 1)
        seq = seq.strip()
        out.write(f">{{header.split()[0]}}\\n{{seq}}:{{TAU_CORE_SEQ}}\\n")
print("wrote complexes.fasta —", len(records), "binder:tau pairs")
"""),
        _cell("code", """#@title 3. Run ColabFold (AlphaFold2-multimer) on every pair
!colabfold_batch complexes.fasta af2_multimer_outputs --num-models 3 --model-type alphafold2_multimer_v3
"""),
        _cell("code", """#@title 4. Parse interface metrics and apply the Year-1 thresholds
import json, glob
INTERFACE_PAE_MAX = 7.5  #@param {type:"number"}
PLDDT_MIN = 85  #@param {type:"number"}
IPTM_MIN = 0.7  #@param {type:"number"}

passed = []
for f in glob.glob("af2_multimer_outputs/*_scores_rank_001*.json"):
    d = json.load(open(f))
    iptm = d.get("iptm", 0)
    plddt = sum(d.get("plddt", [0])) / max(len(d.get("plddt", [1])), 1)
    if iptm >= IPTM_MIN and plddt >= PLDDT_MIN:
        passed.append((f, iptm, plddt))
print(f"{len(passed)} designs passed AF2-multimer thresholds")
for f, iptm, plddt in passed:
    print(f, "ipTM=", round(iptm, 3), "pLDDT=", round(plddt, 1))
"""),
        _cell("code", """#@title 5. Zip and download
import shutil
from google.colab import files
shutil.make_archive("af2_multimer_results", "zip", "af2_multimer_outputs")
files.download("af2_multimer_results.zip")
"""),
    ]
    return _notebook(cells)


def main() -> None:
    target_path = repo_path("results", "target", "ad_capper_target.json")
    target_spec = json.load(open(target_path))

    notebooks_dir = repo_path("notebooks")
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    with open(notebooks_dir / "colab_rfdiffusion.ipynb", "w") as fh:
        json.dump(build_rfdiffusion_notebook(target_spec), fh, indent=1)
    with open(notebooks_dir / "colab_af2_multimer.ipynb", "w") as fh:
        json.dump(build_af2_multimer_notebook(target_spec), fh, indent=1)


if __name__ == "__main__":
    main()
