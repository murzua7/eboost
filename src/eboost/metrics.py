"""Scoring and compute-accounting formulas used by every arm.

Each function here is covered by a golden test on hand-computed values
(tests/test_metrics.py). Nothing here touches XGBoost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-15


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean negative log-likelihood.

    Binary: y in {0,1}, p = P(y=1) with shape (n,). Multiclass: y integer
    labels in 0..K-1, p with shape (n, K) and rows summing to 1.
    """
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    if p.ndim == 1:
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError(f"p has shape {p.shape}, y has length {len(y)}")
    return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(int)])))


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {yhat.shape}")
    return float(np.mean((y - yhat) ** 2))


def regret(loss_arm: float, loss_ref: float) -> float:
    """Relative excess loss, (L_arm - L_ref) / L_ref. Endpoints P1, P3, S6."""
    if loss_ref <= 0:
        raise ValueError(f"reference loss must be positive, got {loss_ref}")
    return (loss_arm - loss_ref) / loss_ref


def row_passes(rounds: int, rows_per_tree: float, n: int) -> float:
    """E = T * B / n, the boosting analogue of an epoch."""
    return rounds * rows_per_tree / n


def hist_proxy(histogram_builds: int, rows: float, features: int) -> float:
    """Hardware-free compute proxy: histogram builds x rows x features."""
    return float(histogram_builds) * float(rows) * float(features)


@dataclass(frozen=True)
class ComputeToMatch:
    cost: float  # compute the tuner spent before first reaching the target
    censored: bool  # True when the tuner never reached it (cost = full budget)


def compute_to_match(costs: np.ndarray, losses: np.ndarray, target: float,
                     full_budget: float) -> ComputeToMatch:
    """First cumulative cost at which the tuner's chosen configuration reaches
    `target` test loss (endpoint P2's numerator). `costs` must be increasing
    and `losses[i]` is the test loss of the configuration the tuner would pick
    after spending `costs[i]`. A tuner that never reaches the target is charged
    its full budget and flagged as censored, which makes the ratio a lower
    bound.
    """
    costs = np.asarray(costs, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    if costs.shape != losses.shape or costs.ndim != 1:
        raise ValueError("costs and losses must be 1-d and the same length")
    if np.any(np.diff(costs) <= 0):
        raise ValueError("costs must be strictly increasing")
    hit = np.flatnonzero(losses <= target)
    if hit.size == 0:
        return ComputeToMatch(cost=float(full_budget), censored=True)
    return ComputeToMatch(cost=float(costs[hit[0]]), censored=False)


@dataclass(frozen=True)
class Window:
    """One rolling origin. Train rows are [0, train_end); test rows are
    [test_start, test_end). Every index is explicit, never implied by a
    horizon."""

    train_end: int
    test_start: int
    test_end: int


def rolling_origins(n: int, n_origins: int, test_len: int,
                    min_train: int) -> list[Window]:
    """Expanding-window rolling origins over n time-ordered rows.

    The last origin's test window ends at row n. Origins step back by
    test_len, so test windows tile the end of the series without overlap.
    """
    if n_origins < 1 or test_len < 1:
        raise ValueError("need at least one origin and a positive test length")
    first_train_end = n - n_origins * test_len
    if first_train_end < min_train:
        raise ValueError(
            f"first origin has {first_train_end} train rows, below min_train={min_train}")
    out = []
    for k in range(n_origins):
        train_end = first_train_end + k * test_len
        out.append(Window(train_end=train_end, test_start=train_end,
                          test_end=train_end + test_len))
    return out
