"""Stage 0 gate and nesting tests: each arm at its simple limit reproduces
the arm below it exactly."""

import numpy as np
import xgboost as xgb

from eboost import arms


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(size=n) > 0).astype(float)
    return X[:400], y[:400], X[400:500], y[400:500], X[500:], y[500:]


OBJ = "binary:logistic"


def test_A0_reproduces_stock_xgboost_exactly():
    """Stage 0 gate: A0 is stock XGBoost with early stopping, nothing added."""
    Xtr, ytr, Xva, yva, Xte, _ = _data()
    cfg = arms.ArmConfig("A0", eta0=0.3, rounds=500, early_stopping=20)
    res = arms.fit(cfg, Xtr, ytr, OBJ, X_val=Xva, y_val=yva)
    stock = xgb.train({"objective": OBJ, "eta": 0.3, "tree_method": "hist", "seed": 0,
                       "nthread": 1}, xgb.DMatrix(Xtr, label=ytr), 500,
                      evals=[(xgb.DMatrix(Xva, label=yva), "val")],
                      early_stopping_rounds=20, verbose_eval=False)
    dte = xgb.DMatrix(Xte)
    np.testing.assert_array_equal(
        res.booster.predict(dte),
        stock.predict(dte, iteration_range=(0, stock.best_iteration + 1)))
    assert res.rounds_kept == stock.best_iteration + 1
    assert res.rounds_built == stock.num_boosted_rounds()


def test_A1_with_floor_one_reproduces_constant_step_exactly():
    Xtr, ytr, _, _, Xte, _ = _data()
    flat = arms.fit(arms.ArmConfig("A1", eta0=0.1, rounds=40, floor=1.0), Xtr, ytr, OBJ)
    const = arms.fit(arms.ArmConfig("A0", eta0=0.1, rounds=40), Xtr, ytr, OBJ)
    dte = xgb.DMatrix(Xte)
    np.testing.assert_array_equal(flat.booster.predict(dte), const.booster.predict(dte))


def test_A2_damped_with_zero_step_reproduces_A1_exactly():
    Xtr, ytr, _, _, Xte, _ = _data()
    a1 = arms.fit(arms.ArmConfig("A1", eta0=0.1, rounds=30, floor=0.1), Xtr, ytr, OBJ)
    cfg = arms.ArmConfig("A2", eta0=0.1, rounds=30, floor=0.1, refit="damped", lam_ref=1.0)
    a2 = arms.fit(cfg, Xtr, ytr, OBJ)
    # damped step is eta(T-1) > 0, so A2 differs; with step 0 it must not
    from eboost import refit
    d = xgb.DMatrix(Xtr, label=ytr)
    design = refit.leaf_design(a1.booster, d)
    zero = refit.write_back(a1.booster, design,
                            refit.damped_refit(design, ytr, "logistic", 1.0, step=0.0))
    dte = xgb.DMatrix(Xte)
    np.testing.assert_array_equal(zero.predict(dte), a1.booster.predict(dte))
    assert not np.array_equal(a2.booster.predict(dte), a1.booster.predict(dte))


def test_A3_with_all_rows_reproduces_A2_exactly():
    Xtr, ytr, _, _, Xte, _ = _data()
    base = {"eta0": 0.1, "rounds": 30, "floor": 0.1, "refit": "full", "lam_ref": 1.0}
    a2 = arms.fit(arms.ArmConfig("A2", **base), Xtr, ytr, OBJ)
    a3 = arms.fit(arms.ArmConfig("A3", subsample=1.0, **base), Xtr, ytr, OBJ)
    dte = xgb.DMatrix(Xte)
    np.testing.assert_array_equal(a3.booster.predict(dte), a2.booster.predict(dte))


def test_full_refit_lowers_training_loss_on_an_annealed_run():
    Xtr, ytr, _, _, _, _ = _data()
    from eboost import metrics
    a1 = arms.fit(arms.ArmConfig("A1", eta0=0.1, rounds=30, floor=0.1), Xtr, ytr, OBJ)
    a2 = arms.fit(arms.ArmConfig("A2", eta0=0.1, rounds=30, floor=0.1, refit="full",
                                 lam_ref=1e-3), Xtr, ytr, OBJ)
    dtr = xgb.DMatrix(Xtr)
    assert metrics.log_loss(ytr, a2.booster.predict(dtr)) < metrics.log_loss(
        ytr, a1.booster.predict(dtr))


def test_compute_accounting():
    Xtr, ytr, _, _, _, _ = _data()
    res = arms.fit(arms.ArmConfig("A3", eta0=0.1, rounds=20, floor=0.1, subsample=0.5),
                   Xtr, ytr, OBJ)
    assert res.rounds_built == 20
    assert res.row_passes == 20 * 0.5  # T * B / n with B = n / 2
    assert res.hist_proxy == arms.internal_nodes(res.booster) * 200 * 8
