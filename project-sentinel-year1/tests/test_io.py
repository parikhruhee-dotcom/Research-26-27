import pytest


def test_parse_fasta(tmp_path):
    from sentinel.io.fetch_sequence import parse_fasta
    f = tmp_path / "test.fasta"
    f.write_text(">sp|P10636-8|TAU_HUMAN test\nMAEPRQ\nEFEVM\n")
    header, seq = parse_fasta(f)
    assert header.startswith("sp|P10636-8")
    assert seq == "MAEPRQEFEVM"


def test_build_aggregation_constructs():
    from sentinel.io.fetch_sequence import build_aggregation_constructs
    full_seq = "X" * 243 + "A" * 129  # pad so indices 244-372 (1-indexed) match landmarks
    landmarks = {
        "K18": {"start": 244, "end": 372},
        "K19": {"start": 244, "end": 372, "exclude": [275, 305]},
    }
    constructs = build_aggregation_constructs(full_seq, landmarks)
    assert len(constructs["K18_4R"]) == 372 - 244 + 1
    assert len(constructs["K19_3R"]) == (372 - 244 + 1) - (305 - 275 + 1)


def test_panel_manifest_has_ten_verified_structures(repo_root):
    import json
    path = repo_root / "data" / "raw" / "structures" / "panel_manifest.json"
    if not path.exists():
        pytest.skip("panel_manifest.json not present — run `make data` first")
    manifest = json.load(open(path))
    assert len(manifest) == 10
    for entry in manifest:
        assert entry.get("cif_path"), f"{entry['pdb_id']} missing cif_path"
        assert "ELECTRON MICROSCOPY" in (entry.get("method") or "").upper()
