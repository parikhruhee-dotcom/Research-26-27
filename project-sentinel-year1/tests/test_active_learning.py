"""Regression tests for the active-learning parameter encoding. A real bug
was found during the build: encoding the 4-way categorical topology choice
as a single ordinal index [0,3] gives the GP's RBF kernel a false distance
structure (topology 0 vs 3 treated as more different than 0 vs 1, purely
from dict ordering) — this measurably hurt learning within a small round
budget. Fixed with one-hot encoding; these tests guard against regression."""
import numpy as np

from sentinel.design.active_learning import (
    N_TOPOLOGIES, PARAM_BOUNDS, build_surrogate, decode_params, random_params,
)

TOPOLOGIES = ["helix_hairpin", "three_helix_bundle", "helix_strand_helix", "long_helix_capper"]


def test_param_bounds_has_onehot_topology_block():
    assert PARAM_BOUNDS.shape[0] == N_TOPOLOGIES + 3
    for i in range(N_TOPOLOGIES):
        assert tuple(PARAM_BOUNDS[i]) == (0.0, 1.0)


def test_decode_params_picks_argmax_topology():
    x = np.array([0.1, 0.9, 0.2, 0.05, 10.0, 0.15, 0.7])
    params = decode_params(x, TOPOLOGIES, [0.1, 0.15, 0.2])
    assert params["topology"] == "three_helix_bundle"  # index 1 has the highest one-hot value


def test_decode_params_all_topologies_reachable():
    seen = set()
    for i in range(N_TOPOLOGIES):
        x = np.zeros(N_TOPOLOGIES + 3)
        x[i] = 1.0
        x[N_TOPOLOGIES:] = [10.0, 0.15, 0.7]
        seen.add(decode_params(x, TOPOLOGIES, [0.1, 0.15, 0.2])["topology"])
    assert seen == set(TOPOLOGIES)


def test_random_params_shape():
    rng = np.random.default_rng(0)
    X = random_params(rng, 10)
    assert X.shape == (10, N_TOPOLOGIES + 3)
    assert np.all(X >= PARAM_BOUNDS[:, 0]) and np.all(X <= PARAM_BOUNDS[:, 1])


def test_surrogate_fits_and_predicts_on_new_param_space():
    rng = np.random.default_rng(0)
    X = random_params(rng, 8)
    y = rng.uniform(0, 1, size=8)
    surrogate = build_surrogate("gaussian_process")
    surrogate.fit(X, y)
    mu, sigma = surrogate.predict(X, return_std=True)
    assert mu.shape == (8,) and sigma.shape == (8,)
