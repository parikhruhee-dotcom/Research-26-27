"""Regression tests for the active-learning parameter encoding. A real bug
was found during the build: encoding the categorical topology choice as a
single ordinal index gives the GP's RBF kernel a false distance structure
(topology 0 vs N-1 treated as more different than 0 vs 1, purely from dict
ordering) — this measurably hurt learning within a small round budget.
Fixed with one-hot encoding; these tests guard against regression. Uses the
real ALL_TOPOLOGY_NAMES registry (4 idealized + 5 real scaffolds) so this
suite can't silently drift out of sync with production, the way a
hardcoded topology list would."""
import numpy as np

from sentinel.design.active_learning import (
    N_TOPOLOGIES, PARAM_BOUNDS, build_surrogate, decode_params, random_params,
)
from sentinel.design.backbone_gen import ALL_TOPOLOGY_NAMES


def test_param_bounds_has_onehot_topology_block():
    assert PARAM_BOUNDS.shape[0] == N_TOPOLOGIES + 3
    for i in range(N_TOPOLOGIES):
        assert tuple(PARAM_BOUNDS[i]) == (0.0, 1.0)


def test_n_topologies_matches_real_registry():
    """Guards against N_TOPOLOGIES silently drifting out of sync with the
    real topology/scaffold registry (a real risk once real scaffolds were
    added alongside the idealized ones)."""
    assert N_TOPOLOGIES == len(ALL_TOPOLOGY_NAMES)


def test_decode_params_picks_argmax_topology():
    x = np.zeros(N_TOPOLOGIES + 3)
    x[1] = 0.9  # second topology should win
    x[N_TOPOLOGIES:] = [10.0, 0.15, 0.7]
    params = decode_params(x, ALL_TOPOLOGY_NAMES, [0.1, 0.15, 0.2])
    assert params["topology"] == ALL_TOPOLOGY_NAMES[1]


def test_decode_params_all_topologies_reachable():
    seen = set()
    for i in range(N_TOPOLOGIES):
        x = np.zeros(N_TOPOLOGIES + 3)
        x[i] = 1.0
        x[N_TOPOLOGIES:] = [10.0, 0.15, 0.7]
        seen.add(decode_params(x, ALL_TOPOLOGY_NAMES, [0.1, 0.15, 0.2])["topology"])
    assert seen == set(ALL_TOPOLOGY_NAMES)


def test_random_params_shape():
    rng = np.random.default_rng(0)
    X = random_params(rng, 10)
    assert X.shape == (10, N_TOPOLOGIES + 3)
    assert np.all(X >= PARAM_BOUNDS[:, 0]) and np.all(X <= PARAM_BOUNDS[:, 1])


def test_random_params_topology_block_is_true_onehot_corner():
    """Regression test for a real bug: the topology block must be an exact
    one-hot corner (a single 1.0, the rest exactly 0.0), not independent
    continuous values that merely happen to be argmax-decodable. A soft
    encoding breaks the GP's categorical kernel — see random_params'
    docstring for the measured impact (active learning failing to beat
    random search)."""
    rng = np.random.default_rng(0)
    X = random_params(rng, 200)
    onehot_block = X[:, :N_TOPOLOGIES]
    assert np.all(np.isin(onehot_block, [0.0, 1.0]))
    assert np.all(onehot_block.sum(axis=1) == 1.0)


def test_random_params_topology_roughly_uniform():
    rng = np.random.default_rng(0)
    X = random_params(rng, 2000)
    counts = X[:, :N_TOPOLOGIES].sum(axis=0)
    # each topology should appear a roughly-fair share of draws (loose bound; this
    # is a sanity check against a broken/biased sampler, not a strict uniformity test)
    assert np.all(counts > 2000 / N_TOPOLOGIES * 0.5)


def test_surrogate_fits_and_predicts_on_new_param_space():
    rng = np.random.default_rng(0)
    X = random_params(rng, 8)
    y = rng.uniform(0, 1, size=8)
    surrogate = build_surrogate("gaussian_process")
    surrogate.fit(X, y)
    mu, sigma = surrogate.predict(X, return_std=True)
    assert mu.shape == (8,) and sigma.shape == (8,)
