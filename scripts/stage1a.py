"""Stage 1a pilot on F, one (task, seed) per call, one JSONL file out.

Development data only (F). Budgets here are PILOT budgets, smaller than the
registered ones: 50 search configurations on one inner holdout instead of 200
with 5-fold CV. Nothing in this file touches an H dataset.

Arms, all on the outer training rows and scored on the outer test rows:
  A0        library defaults, early stopping on the inner holdout
  A0-T      best searched config, constant eta*, T* rounds, its own subsample
  C1        A0-T with all rows (subsample 1): the constant-step control
  A1r       cosine from eta*, T* rounds (matched rounds), all rows
  A1e       cosine from eta*, T_e rounds with sum of steps = eta* T* (matched eta.T)
  A2-*      A1r plus full or damped leaf refit, lambda_ref on a grid
  A3-s*     cosine from eta*, row sample s, ceil(T*/s) rounds (matched row-passes), no refit
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from eboost import arms, data, runlog, schedule, search

REFIT_NNZ_CAP = 100_000_000   # n_train x T above this: refit skipped and logged
ROUND_CAP = 40_000


def matched_sum_rounds(T: int, floor: float) -> int:
    """Smallest T_e whose cosine steps sum to at least T (in units of eta0)."""
    T_e = T
    while sum(schedule.schedule(T_e, 1.0, floor)) < T:
        T_e += 1
    return T_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nthread", type=int, default=4)
    a = ap.parse_args()

    d = data.load(a.data, a.task)
    X, y, obj = d.X, d.y, d.objective
    sp = search.split(len(y), a.seed, strata=None if d.kind == "regression" else y)
    out = a.out / f"task{a.task}_seed{a.seed}.jsonl"
    common = {"task": a.task, "dataset": d.name, "kind": d.kind, "seed": a.seed,
              "n_train": len(sp.train), "n_test": len(sp.test), "p": X.shape[1],
              "data_hash": runlog.data_hash(X, y), "stage": "1a-pilot"}
    kw = {"num_class": d.num_class, "seed": a.seed, "nthread": a.nthread, "device": a.device}

    t0 = time.perf_counter()
    trials = search.random_search(X, y, obj, sp, search.sample_configs(a.configs, a.seed),
                                  num_class=d.num_class, seed=a.seed, nthread=a.nthread,
                                  device=a.device)
    for i, t in enumerate(trials):
        runlog.append(out, {**common, "record": "trial", "i": i, "config": t.config,
                            "best_rounds": t.best_rounds, "rounds_built": t.rounds_built,
                            "val_loss": t.val_loss, "test_loss_inner_model": t.test_loss})
    best = min(trials, key=lambda t: t.val_loss)
    eta, T = best.config["eta"], best.best_rounds
    tuned = {k: v for k, v in best.config.items() if k != "eta"}
    all_rows = {**tuned, "subsample": 1.0}
    runlog.append(out, {**common, "record": "search_done", "seconds": time.perf_counter() - t0,
                        "search_rounds_built": sum(t.rounds_built for t in trials),
                        "eta_star": eta, "T_star": T})

    Xtr, ytr, Xte, yte = X[sp.train], y[sp.train], X[sp.test], y[sp.test]
    T_e = matched_sum_rounds(T, schedule.DEFAULT_FLOOR)
    cfgs = [
        arms.ArmConfig("C1", eta0=eta, rounds=T, params=all_rows),
        arms.ArmConfig("A0-T", eta0=eta, rounds=T, subsample=tuned["subsample"], params=tuned),
        arms.ArmConfig("A1r", eta0=eta, rounds=T, floor=0.1, params=all_rows),
        arms.ArmConfig("A1e", eta0=eta, rounds=T_e, floor=0.1, params=all_rows),
    ]
    nnz = len(ytr) * T * (d.num_class or 1)  # one indicator per row per tree
    refit_ok = nnz <= REFIT_NNZ_CAP
    for lam in (0.1, 1.0, 10.0, 100.0):
        if refit_ok:
            cfgs.append(arms.ArmConfig(f"A2-full-l{lam:g}", eta0=eta, rounds=T, floor=0.1,
                                       refit="full", lam_ref=lam, params=all_rows))
    for lam in (1.0, 10.0):
        if refit_ok:
            cfgs.append(arms.ArmConfig(f"A2-damped-l{lam:g}", eta0=eta, rounds=T, floor=0.1,
                                       refit="damped", lam_ref=lam, params=all_rows))
    if not refit_ok:
        runlog.append(out, {**common, "record": "refit_skipped", "nnz": nnz})
    for s in (0.05, 0.1, 0.2, 0.4, 0.7):
        rounds = min(ROUND_CAP, math.ceil(T / s))
        cfgs.append(arms.ArmConfig(f"A3-s{s:g}", eta0=eta, rounds=rounds, floor=0.1,
                                   subsample=s, params={**all_rows, "subsample": s}))

    cfgs.insert(0, arms.ArmConfig("A0", eta0=0.3, rounds=4000, early_stopping=50))
    dte = xgb.DMatrix(Xte, nthread=a.nthread)
    dtr = xgb.DMatrix(Xtr, nthread=a.nthread)
    for cfg in cfgs:
        if cfg.name == "A0":
            res = arms.fit(cfg, X[sp.inner_train], y[sp.inner_train], obj,
                           X_val=X[sp.inner_val], y_val=y[sp.inner_val], **kw)
        else:
            res = arms.fit(cfg, Xtr, ytr, obj, **kw)
        runlog.append(out, {
            **common, "record": "arm", "arm": cfg.name,
            "config": {"eta0": cfg.eta0, "rounds": cfg.rounds, "floor": cfg.floor,
                       "refit": cfg.refit, "lam_ref": cfg.lam_ref, "subsample": cfg.subsample,
                       "params": cfg.params},
            "test_loss": search.loss_of(obj, yte, res.booster.predict(dte)),
            "train_loss": search.loss_of(obj, ytr, res.booster.predict(dtr)),
            "rounds_built": res.rounds_built, "rounds_kept": res.rounds_kept,
            "row_passes": res.row_passes, "hist_proxy": res.hist_proxy,
            "seconds_boost": res.seconds_boost, "seconds_refit": res.seconds_refit,
            "step_sum": float(np.sum(schedule.schedule(cfg.rounds, cfg.eta0, cfg.floor)))
            if cfg.floor is not None else cfg.eta0 * res.rounds_built})
    runlog.append(out, {**common, "record": "done", "seconds": time.perf_counter() - t0})


if __name__ == "__main__":
    main()
