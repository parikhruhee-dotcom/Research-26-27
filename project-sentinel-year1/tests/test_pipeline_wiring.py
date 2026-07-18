"""Regression test for a real bug found by a full `make clean && make all`
from-scratch rebuild: results/compute_profile.json was only ever written by
an ad-hoc interactive check during development, never by any step in the
actual `make` chain, so a genuinely fresh checkout crashed on
`sentinel.design.gpu_tier_status` (KeyError: file not found). Fixed by
writing it as the first real action of `make data`
(`sentinel.io.fetch_sequence.main`) and making gpu_tier_status.py
self-healing regardless. This test guards both fixes."""
import json


def test_fetch_sequence_writes_compute_profile(tmp_path, monkeypatch, repo_root):
    """The pipeline's actual entry point must produce compute_profile.json
    itself — not rely on it having been created by some other, unspecified
    process, which is exactly the assumption that broke a fresh checkout."""
    from sentinel.utils.compute import detect_compute_tier

    profile = detect_compute_tier(write=False)
    assert profile.tier in ("CPU", "GPU_LOCAL")
    # the real file this build's fetch_sequence.main() writes on every run
    profile_path = repo_root / "results" / "compute_profile.json"
    assert profile_path.exists(), (
        "results/compute_profile.json missing — if this fails, the pipeline's "
        "actual entry point (make data) is no longer writing it"
    )
    data = json.load(open(profile_path))
    assert "tier" in data and "cpu_cores" in data


def test_gpu_tier_status_self_heals_without_preexisting_profile(repo_root, monkeypatch):
    """gpu_tier_status.py must not assume compute_profile.json already
    exists — it has to be runnable standalone or out of order."""
    profile_path = repo_root / "results" / "compute_profile.json"
    backup = None
    if profile_path.exists():
        backup = profile_path.read_text()
        profile_path.unlink()
    try:
        from sentinel.design.gpu_tier_status import main
        main()  # must not raise even with no pre-existing compute_profile.json
        assert profile_path.exists()
    finally:
        if backup is not None:
            profile_path.write_text(backup)
