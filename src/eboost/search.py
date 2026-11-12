"""Random search over TabArena's XGBoost space (A0-T) and the data splits.

The space copies tabarena/models/xgboost/hpo.py (autogluon/tabarena, saved
2026-11-12): learning_rate log-uniform [5e-3, 1e-1], max_depth log-int [4, 10],
min_child_weight log-uniform [1e-3, 5], subsample, colsample_bylevel and
colsample_bynode uniform [0.6, 1], reg_alpha and reg_lambda uniform [1e-4, 5],
grow_policy in {depthwise, lossguide}, max_leaves log-int [8, 1024].
max_cat_to_onehot is left out: categoricals are ordinal-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb

from eboost import metrics


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _log_int(rng, lo, hi):
    return int(np.clip(np.round(np.exp(rng.uniform(np.log(lo - 0.5), np.log(hi + 0.5)))), lo, hi))


def sample_configs(n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({
            "eta": _log_uniform(rng, 5e-3, 1e-1),
            "max_depth": _log_int(rng, 4, 10),
            "min_child_weight": _log_uniform(rng, 1e-3, 5.0),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bylevel": float(rng.uniform(0.6, 1.0)),
            "colsample_bynode": float(rng.uniform(0.6, 1.0)),
            "reg_alpha": float(rng.uniform(1e-4, 5.0)),
            "reg_lambda": float(rng.uniform(1e-4, 5.0)),
            "grow_policy": str(rng.choice(["depthwise", "lossguide"])),
            "max_leaves": _log_int(rng, 8, 1024),
        })
    return out


@dataclass(frozen=True)
class Split:
    train: np.ndarray   # outer training rows
    test: np.ndarray    # outer test rows, never seen by search or arms
    inner_train: np.ndarray
    inner_val: np.ndarray


def split(n: int, seed: int, test_frac: float = 0.2, val_frac: float = 0.2,
          strata: np.ndarray | None = None) -> Split:
    """Outer train/test, then inner train/validation carved from train only."""
    rng = np.random.default_rng(seed)

    def cut(idx, frac):
        if strata is None:
            perm = rng.permutation(idx)
            k = round(frac * len(idx))
            return np.sort(perm[k:]), np.sort(perm[:k])
        keep, hold = [], []
        for c in np.unique(strata[idx]):
            members = rng.permutation(idx[strata[idx] == c])
            k = round(frac * len(members))
            hold.append(members[:k])
            keep.append(members[k:])
        return np.sort(np.concatenate(keep)), np.sort(np.concatenate(hold))

    train, test = cut(np.arange(n), test_frac)
    inner_train, inner_val = cut(train, val_frac)
    return Split(train, test, inner_train, inner_val)


def loss_of(objective: str, y: np.ndarray, pred: np.ndarray) -> float:
    if objective == "reg:squarederror":
        return metrics.mse(y, pred)
    return metrics.log_loss(y, pred)


@dataclass
class Trial:
    config: dict[str, Any]
    best_rounds: int
    rounds_built: int
    val_loss: float
    test_loss: float  # of the inner-trained model; for the anytime curve only


def random_search(X, y, objective, sp: Split, configs, *, num_class=None, cap=4000,
                  patience=50, seed=0, nthread=1, device="cpu") -> list[Trial]:
    dtr = xgb.DMatrix(X[sp.inner_train], label=y[sp.inner_train], nthread=nthread)
    dva = xgb.DMatrix(X[sp.inner_val], label=y[sp.inner_val], nthread=nthread)
    dte = xgb.DMatrix(X[sp.test], nthread=nthread)
    trials = []
    for cfg in configs:
        params = {"objective": objective, "tree_method": "hist", "device": device,
                  "seed": seed, "nthread": nthread, **cfg}
        if num_class:
            params["num_class"] = num_class
        bst = xgb.train(params, dtr, cap, evals=[(dva, "val")], early_stopping_rounds=patience,
                        verbose_eval=False)
        it = (0, bst.best_iteration + 1)
        trials.append(Trial(cfg, bst.best_iteration + 1, bst.num_boosted_rounds(),
                            loss_of(objective, y[sp.inner_val],
                                    bst.predict(dva, iteration_range=it)),
                            loss_of(objective, y[sp.test], bst.predict(dte, iteration_range=it))))
    return trials
