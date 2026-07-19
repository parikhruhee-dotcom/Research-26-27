"""M6d — the closed-loop active-learning driver (the flagship novelty).

Design space per round (a small, real, continuous/discrete parameter vector,
NOT the sequence itself — the surrogate learns which *generation settings*
tend to produce good designs). Topology is an N-way CATEGORICAL choice (4
idealized geometric topologies + 5 real solved-structure scaffolds — see
backbone_gen.ALL_TOPOLOGY_NAMES), one-hot encoded as N separate [0,1]
dimensions (argmax picks the topology at decode time) rather than a single
ordinal index:
  x = [onehot_topology_0, ..., onehot_topology_{N-1},
       standoff_A (7-15), mpnn_temperature (0.05-0.3), hotspot_fraction (0.5-1.0)]

This one-hot encoding is not cosmetic: an RBF kernel over a single ordinal
topology index imposes a false, arbitrary distance structure (it treats
topology 0 vs N-1 as "more different" than 0 vs 1, purely because of dict
ordering, with no basis in reality) and measurably hurt the GP's ability to
learn which topology performs well within a small round budget — caught
during this build when a run's random-search baseline pulled ahead of
active learning by chance-favoring one topology early (see PROGRESS_LOG.md
M6). One-hot encoding lets the kernel treat topology identity as its own
independent dimension, which is the methodologically correct way to put a
categorical variable into a continuous-kernel GP. Letting the loop choose
freely between idealized and real-scaffold topologies also means the
closed loop itself empirically discovers which backbone source performs
better — an honest result, not an assumption.

Loop: propose params -> generate backbone(s) + ProteinMPNN sequences with
those params -> score every resulting design -> record the best score for
that x -> refit a Gaussian Process surrogate on all (x, best_score) pairs so
far -> propose the next round's x via expected-improvement (or randomly, for
the random-search baseline used in M9's ablation).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from sentinel.design.backbone_gen import ALL_TOPOLOGY_NAMES

N_TOPOLOGIES = len(ALL_TOPOLOGY_NAMES)
PARAM_BOUNDS = np.array(
    [[0.0, 1.0]] * N_TOPOLOGIES + [
        [7.0, 15.0],   # standoff_A
        [0.05, 0.30],  # mpnn_temperature
        [0.5, 1.0],    # hotspot_fraction
    ]
)


def random_params(rng: np.random.Generator, n: int) -> np.ndarray:
    """Topology is a true categorical, sampled as an exact one-hot corner
    (uniform pick over ALL_TOPOLOGY_NAMES) — not N independent continuous
    Uniform[0,1] draws. A real bug was found and fixed here: the original
    version drew every column, including the one-hot block, as independent
    continuous values. Two non-corner points can argmax to different
    topologies while sitting close together in raw x-space, and two points
    that decode to the SAME topology can sit far apart in that space — so
    the GP's RBF kernel, which measures smooth Euclidean distance in raw x,
    was fed a signal with no real relationship to the actual (discontinuous,
    argmax-decoded) categorical structure it needed to learn. Measured
    impact: with the soft encoding, active learning's mean candidate score
    (0.3286) did not beat an equal-budget random-search baseline (0.3311)
    over a full 10-round/320-design run — the surrogate could not reliably
    learn which topology tended to score well. Sampling true one-hot corners
    makes same-topology points identical in that block and different-
    topology points a constant, well-defined distance apart in it — the
    behavior one-hot + RBF is supposed to give a genuine categorical
    kernel."""
    n_continuous = PARAM_BOUNDS.shape[0] - N_TOPOLOGIES
    lo, hi = PARAM_BOUNDS[N_TOPOLOGIES:, 0], PARAM_BOUNDS[N_TOPOLOGIES:, 1]
    topo_idx = rng.integers(0, N_TOPOLOGIES, size=n)
    onehot = np.zeros((n, N_TOPOLOGIES))
    onehot[np.arange(n), topo_idx] = 1.0
    continuous = rng.uniform(lo, hi, size=(n, n_continuous))
    return np.concatenate([onehot, continuous], axis=1)


def build_surrogate(kind: str):
    if kind == "gaussian_process":
        n_dims = len(PARAM_BOUNDS)
        length_scale = [0.5] * N_TOPOLOGIES + [2.0, 0.1, 0.1]
        kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(1e-3)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                                          random_state=42)
    elif kind == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    raise ValueError(f"unknown surrogate kind: {kind}")


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_y - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-8] = 0.0
    return ei


def propose_next_params(surrogate, X_observed: np.ndarray, y_observed: np.ndarray,
                          n_propose: int, xi: float, rng: np.random.Generator,
                          n_candidates: int = 2000) -> np.ndarray:
    """Expected-improvement acquisition over a large random candidate pool
    (a standard, simple way to do EI-based BO without a separate continuous
    optimizer)."""
    candidates = random_params(rng, n_candidates)
    if hasattr(surrogate, "predict") and hasattr(surrogate, "kernel_"):
        mu, sigma = surrogate.predict(candidates, return_std=True)
    else:
        # RandomForestRegressor: use the ensemble's tree-to-tree spread as an uncertainty proxy
        tree_preds = np.array([tree.predict(candidates) for tree in surrogate.estimators_])
        mu, sigma = tree_preds.mean(axis=0), tree_preds.std(axis=0)

    best_y = float(y_observed.max())
    ei = expected_improvement(mu, sigma, best_y, xi)
    top_idx = np.argsort(ei)[::-1][:n_propose]
    return candidates[top_idx]


def decode_params(x: np.ndarray, topologies: list[str], mpnn_temp_choices: list[float]) -> dict:
    onehot = x[:N_TOPOLOGIES]
    topo_idx = int(np.argmax(onehot[:len(topologies)]))
    standoff_A = float(x[N_TOPOLOGIES])
    mpnn_temp = float(min(mpnn_temp_choices, key=lambda t: abs(t - x[N_TOPOLOGIES + 1])))
    hotspot_fraction = float(np.clip(x[N_TOPOLOGIES + 2], 0.5, 1.0))
    return {"topology": topologies[topo_idx], "standoff_A": standoff_A,
            "mpnn_temperature": mpnn_temp, "hotspot_fraction": hotspot_fraction}
