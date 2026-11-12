import math

import numpy as np
import pytest
import xgboost as xgb
from hypothesis import given, settings
from hypothesis import strategies as st

from eboost import schedule


def test_cosine_golden():
    # eta0 = 0.2, eps = 0.1, T = 4:
    # m=0: 0.2;  m=2: 0.2*(0.1 + 0.9*0.5) = 0.11;  m=1: 0.2*(0.1 + 0.9*0.5*(1+cos(pi/4)))
    assert schedule.cosine_eta(0, 4, 0.2, 0.1) == pytest.approx(0.2)
    assert schedule.cosine_eta(2, 4, 0.2, 0.1) == pytest.approx(0.11)
    assert schedule.cosine_eta(1, 4, 0.2, 0.1) == pytest.approx(
        0.2 * (0.1 + 0.45 * (1 + math.sqrt(0.5))))
    with pytest.raises(ValueError):
        schedule.cosine_eta(4, 4, 0.2, 0.1)


@given(T=st.integers(1, 500), eta0=st.floats(1e-4, 1.0), eps=st.floats(0.0, 1.0))
@settings(max_examples=200, deadline=None)
def test_schedule_is_nonincreasing_and_bounded(T, eta0, eps):
    s = np.array(schedule.schedule(T, eta0, eps))
    assert s[0] == pytest.approx(eta0)
    assert np.all(np.diff(s) <= 1e-15)
    assert np.all(s >= eps * eta0 - 1e-15)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = X[:, 0] - 0.5 * X[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    return X, y


def test_callback_sets_each_rounds_step_inside_xgboost():
    """Training with the callback equals training round by round, resetting
    eta by hand before each round. This checks XGBoost really used eta(m)."""
    X, y = _data()
    d = xgb.DMatrix(X, label=y)
    base = {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": 3,
            "seed": 0, "nthread": 1}
    T, eta0, eps = 4, 0.25, 0.1
    a = xgb.train(base, d, T, callbacks=[schedule.callback(T, eta0, eps)])
    np.testing.assert_array_equal(a.predict(d), _by_hand(base, d, schedule.schedule(T, eta0, eps)))


def _by_hand(base, d, rates):
    b = None
    for eta in rates:
        b = xgb.train({**base, "eta": eta}, d, 1, xgb_model=b)
    return b.predict(d)


def test_xgboost_builtin_scheduler_is_one_round_late():
    """Documents why schedule.py does not use xgb.callback.LearningRateScheduler:
    it trains with [params eta, rate[0], ..., rate[T-2]]."""
    X, y = _data()
    d = xgb.DMatrix(X, label=y)
    base = {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": 3,
            "seed": 0, "nthread": 1}
    rates = [0.25, 0.15, 0.08, 0.02]
    builtin = xgb.train(base, d, 4, callbacks=[xgb.callback.LearningRateScheduler(rates)])
    np.testing.assert_array_equal(builtin.predict(d), _by_hand(base, d, [0.3] + rates[:-1]))
    assert not np.array_equal(builtin.predict(d), _by_hand(base, d, rates))
