import pytest

from sentinel.atlas.physicochemical import charge_class, residue_physchem
from sentinel.atlas.sasa import classify_burial


def test_charge_class():
    assert charge_class("ASP") == "negative"
    assert charge_class("LYS") == "positive"
    assert charge_class("ALA") == "neutral"


def test_residue_physchem_aromatic():
    assert residue_physchem("TYR")["is_aromatic"] is True
    assert residue_physchem("ALA")["is_aromatic"] is False


def test_classify_burial_thresholds(config):
    buried_t = config["atlas"]["buried_rel_sasa_threshold"]
    exposed_t = config["atlas"]["exposed_rel_sasa_threshold"]
    assert classify_burial(buried_t - 0.01) == "buried"
    assert classify_burial(exposed_t + 0.01) == "exposed"
    assert classify_burial((buried_t + exposed_t) / 2) == "intermediate"
    assert classify_burial(None) == "unknown"


def test_kabsch_rmsd_identical_structures_is_zero():
    import numpy as np
    from sentinel.atlas.alignment import kabsch_rmsd
    coords = np.random.RandomState(0).rand(10, 3) * 10
    assert kabsch_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-6)


def test_kabsch_rmsd_rotation_invariant():
    import numpy as np
    from sentinel.atlas.alignment import kabsch_rmsd
    rng = np.random.RandomState(0)
    coords = rng.rand(10, 3) * 10
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rotated = coords @ rot.T
    assert kabsch_rmsd(coords, rotated) == pytest.approx(0.0, abs=1e-4)
