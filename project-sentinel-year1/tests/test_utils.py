import numpy as np


def test_config_loads(config):
    assert config["project"]["seed"] == 42
    assert "strain_panel" in config["data"]
    assert len(config["data"]["strain_panel"]) == 10


def test_set_global_seed_reproducible():
    from sentinel.utils.seeds import set_global_seed
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_compute_tier_detection():
    from sentinel.utils.compute import detect_compute_tier
    profile = detect_compute_tier(write=False)
    assert profile.tier in ("CPU", "GPU_LOCAL")
    assert profile.cpu_cores >= 1


def test_provenance_sha256(tmp_path):
    from sentinel.utils.provenance import sha256_of_file
    f = tmp_path / "x.txt"
    f.write_text("hello world")
    h = sha256_of_file(f)
    import hashlib
    assert h == hashlib.sha256(b"hello world").hexdigest()
