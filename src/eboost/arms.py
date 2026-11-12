"""The five arms (pre-registration, Method). Each adds one component:

    A0  constant step, library defaults, early stopping on inner validation
    A1  cosine-annealed step over a fixed T, no early stopping
    A2  A1 plus a joint leaf refit (full or damped)
    A3  A2 grown on row samples of size B (subsample = B / n)
    A4  A3 with every setting read from the fitted laws (laws.py, Stage 2)

A0-T is A0 with settings from the random search (search.py). Every arm calls
xgb.train, so all trees come from XGBoost's own histogram kernels.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xgboost as xgb

from eboost import metrics, refit, schedule


@dataclass(frozen=True)
class ArmConfig:
    name: str
    eta0: float = 0.3
    rounds: int = 100                  # T, or the cap when early stopping
    floor: float | None = None         # None: constant step. Else cosine with this floor
    early_stopping: int | None = None  # patience in rounds, needs validation data
    refit: str | None = None           # None, "full" or "damped"
    lam_ref: float = 1.0
    subsample: float = 1.0             # B / n
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitResult:
    booster: xgb.Booster
    rounds_built: int
    rounds_kept: int
    row_passes: float
    hist_proxy: float
    seconds_boost: float
    seconds_refit: float


def xgb_params(cfg: ArmConfig, objective: str, *, seed: int, nthread: int, device: str,
               num_class: int | None) -> dict[str, Any]:
    p: dict[str, Any] = {"objective": objective, "eta": cfg.eta0, "subsample": cfg.subsample,
                         "tree_method": "hist", "device": device, "seed": seed,
                         "nthread": nthread, **cfg.params}
    if num_class is not None:
        p["num_class"] = num_class
    return p


def internal_nodes(bst: xgb.Booster) -> int:
    trees = json.loads(bst.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"]
    return sum(int(np.sum(np.asarray(t["left_children"]) != -1)) for t in trees)


def fit(cfg: ArmConfig, X: np.ndarray, y: np.ndarray, objective: str, *,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
        num_class: int | None = None, seed: int = 0, nthread: int = 1,
        device: str = "cpu") -> FitResult:
    if cfg.floor is not None and cfg.early_stopping is not None:
        raise ValueError("an annealed arm fixes T in advance and cannot stop early")
    if cfg.early_stopping is not None and X_val is None:
        raise ValueError("early stopping needs inner validation rows")
    params = xgb_params(cfg, objective, seed=seed, nthread=nthread, device=device,
                        num_class=num_class)
    dtrain = xgb.DMatrix(X, label=y, nthread=nthread)
    callbacks = [schedule.callback(cfg.rounds, cfg.eta0, cfg.floor)] if cfg.floor is not None else []
    evals = [(xgb.DMatrix(X_val, label=y_val, nthread=nthread), "val")] if X_val is not None else []

    t0 = time.perf_counter()
    bst = xgb.train(params, dtrain, cfg.rounds, evals=evals, verbose_eval=False,
                    early_stopping_rounds=cfg.early_stopping, callbacks=callbacks)
    seconds_boost = time.perf_counter() - t0

    rounds_built = bst.num_boosted_rounds()
    proxy = metrics.hist_proxy(internal_nodes(bst), cfg.subsample * len(X), X.shape[1])
    if cfg.early_stopping is not None:
        bst = bst[: bst.best_iteration + 1]
    rounds_kept = bst.num_boosted_rounds()

    seconds_refit = 0.0
    if cfg.refit is not None:
        t1 = time.perf_counter()
        design = refit.leaf_design(bst, dtrain)
        kind = refit.loss_kind(bst)
        if cfg.refit == "full":
            w = refit.full_refit(design, y, kind, cfg.lam_ref)
        elif cfg.refit == "damped":
            last = (schedule.cosine_eta(cfg.rounds - 1, cfg.rounds, cfg.eta0, cfg.floor)
                    if cfg.floor is not None else cfg.eta0)
            w = refit.damped_refit(design, y, kind, cfg.lam_ref, step=last)
        else:
            raise ValueError(f"unknown refit {cfg.refit}")
        bst = refit.write_back(bst, design, w)
        seconds_refit = time.perf_counter() - t1

    return FitResult(booster=bst, rounds_built=rounds_built, rounds_kept=rounds_kept,
                     row_passes=metrics.row_passes(rounds_built, cfg.subsample * len(X), len(X)),
                     hist_proxy=proxy, seconds_boost=seconds_boost, seconds_refit=seconds_refit)
