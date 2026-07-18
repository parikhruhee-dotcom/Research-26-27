"""Smoke tests: every pipeline stage's key output files exist and are
well-formed, once the pipeline has been run. Skips (not fails) if a stage
hasn't been run yet, so this file works both in CI-from-scratch and against
an already-built repo."""
import csv
import json

import pytest


def _skip_if_missing(path):
    if not path.exists():
        pytest.skip(f"{path} not present — run the relevant `make` target first")


def test_atlas_outputs(repo_root):
    path = repo_root / "results" / "atlas" / "ad_strain_fingerprint.json"
    _skip_if_missing(path)
    d = json.load(open(path))
    assert len(d["top_hotspots"]) > 0
    assert all("differential_exposure" in h for h in d["top_hotspots"])


def test_figures_generated(repo_root):
    fig_dir = repo_root / "figures"
    if not fig_dir.exists() or not any(fig_dir.glob("fig*.png")):
        pytest.skip("figures not generated yet — run `make figures` first")
    pngs = list(fig_dir.glob("fig*.png"))
    svgs = list(fig_dir.glob("fig*.svg"))
    pdfs = list(fig_dir.glob("fig*.pdf"))
    assert len(pngs) >= 11
    assert len(svgs) == len(pngs)
    assert len(pdfs) == len(pngs)
    assert (fig_dir / "CAPTIONS.md").exists()


def test_benchmarks_outputs(repo_root):
    path = repo_root / "results" / "benchmarks" / "aggregation_roc_pr.json"
    _skip_if_missing(path)
    d = json.load(open(path))
    assert 0.0 <= d["roc_auc"] <= 1.0
    assert d["roc_auc"] > 0.5, "aggregation predictor should beat random chance"


def test_design_leads_are_valid_sequences(repo_root):
    path = repo_root / "results" / "design" / "leads.fasta"
    _skip_if_missing(path)
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    text = open(path).read()
    seqs = [line.strip() for line in text.splitlines() if line and not line.startswith(">")]
    assert len(seqs) > 0
    for seq in seqs:
        assert set(seq) <= valid_aa, f"invalid residues in {seq}"


def test_provenance_ledger_populated(repo_root):
    path = repo_root / "results" / "PROVENANCE.json"
    _skip_if_missing(path)
    d = json.load(open(path))
    assert len(d["downloads"]) >= 10, "expected at least the 10-structure panel logged"
    assert all("sha256" in entry for entry in d["downloads"])
