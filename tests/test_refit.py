import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import xgboost as xgb
from hypothesis import given, settings
from hypothesis import strategies as st

from eboost import refit

CASES = {
    "reg:squarederror": None,
    "binary:logistic": None,
    "multi:softprob": 3,
}


def _fit(objective, seed=0, n=300, rounds=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    if objective == "reg:squarederror":
        y = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.3 * rng.normal(size=n)
    elif objective == "binary:logistic":
        y = (X[:, 0] + X[:, 1] ** 2 + rng.normal(size=n) > 1).astype(float)
    else:
        y = np.digitize(X[:, 0] + 0.5 * rng.normal(size=n), [-0.5, 0.5]).astype(float)
    params = {"objective": objective, "max_depth": 3, "eta": 0.3, "seed": seed, "nthread": 1}
    if CASES[objective]:
        params["num_class"] = CASES[objective]
    d = xgb.DMatrix(X, label=y)
    return xgb.train(params, d, rounds), d, y


@pytest.mark.parametrize("objective", CASES)
def test_write_back_of_current_leaves_is_bit_identical(objective):
    bst, d, _ = _fit(objective)
    design = refit.leaf_design(bst, d)
    again = refit.write_back(bst, design, design.w0)
    np.testing.assert_array_equal(bst.predict(d, output_margin=True),
                                  again.predict(d, output_margin=True))


@pytest.mark.parametrize("objective", CASES)
def test_xgboost_predictions_match_offset_plus_Zw(objective):
    """Gate on the scored artifact: after writing new leaves, XGBoost's own
    margin equals offset + Z w to float32 precision."""
    bst, d, _ = _fit(objective)
    design = refit.leaf_design(bst, d)
    w = design.w0 + np.random.default_rng(1).normal(scale=0.1, size=design.w0.shape)
    new = refit.write_back(bst, design, w)
    got = new.predict(d, output_margin=True).reshape(len(design.offset), -1)
    np.testing.assert_allclose(got, refit.margin(design, w), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("objective", CASES)
def test_gradient_and_hessian_vector_match_finite_differences(objective):
    bst, d, y = _fit(objective)
    design = refit.leaf_design(bst, d)
    prob = refit.Problem(design, y, refit.loss_kind(bst), lam=0.7)
    rng = np.random.default_rng(2)
    w = design.w0 + rng.normal(scale=0.05, size=design.w0.shape)
    v = rng.normal(size=w.shape)
    v /= np.linalg.norm(v)
    h = 1e-5
    fd_dir = (prob.value(w + h * v) - prob.value(w - h * v)) / (2 * h)
    assert prob.grad(w) @ v == pytest.approx(fd_dir, rel=1e-6)
    fd_hv = (prob.grad(w + h * v) - prob.grad(w - h * v)) / (2 * h)
    np.testing.assert_allclose(prob.hessp(w, v), fd_hv, rtol=1e-5, atol=1e-6)


def test_full_refit_squared_matches_ridge_closed_form():
    """Manufactured solution: for squared loss the problem is ridge regression."""
    bst, d, y = _fit("reg:squarederror")
    design = refit.leaf_design(bst, d)
    lam = 2.0
    A = (design.Z.T @ design.Z + lam * sp.eye(design.Z.shape[1])).tocsc()
    w_star = spl.spsolve(A, design.Z.T @ (y - design.offset[:, 0]))
    w = refit.full_refit(design, y, "squared", lam)
    np.testing.assert_allclose(w, w_star, rtol=1e-6, atol=1e-8)


def test_damped_step_one_on_squared_loss_is_the_exact_solution():
    """One Newton step solves a quadratic exactly, so damped(step=1) = full."""
    bst, d, y = _fit("reg:squarederror")
    design = refit.leaf_design(bst, d)
    full = refit.full_refit(design, y, "squared", 1.5)
    damped = refit.damped_refit(design, y, "squared", 1.5, step=1.0)
    np.testing.assert_allclose(damped, full, rtol=1e-6, atol=1e-8)


@given(seed=st.integers(0, 50), lam=st.floats(0.01, 50.0),
       objective=st.sampled_from(sorted(CASES)))
@settings(max_examples=25, deadline=None)
def test_full_refit_never_raises_the_penalized_training_objective(seed, lam, objective):
    bst, d, y = _fit(objective, seed=seed, n=200, rounds=4)
    design = refit.leaf_design(bst, d)
    kind = refit.loss_kind(bst)
    prob = refit.Problem(design, y, kind, lam)
    w = refit.full_refit(design, y, kind, lam)
    assert prob.value(w) <= prob.value(design.w0) + 1e-9 * abs(prob.value(design.w0))


def test_huge_penalty_shrinks_leaves_to_zero():
    bst, d, y = _fit("binary:logistic")
    design = refit.leaf_design(bst, d)
    w = refit.full_refit(design, y, "logistic", lam=1e9)
    assert np.max(np.abs(w)) < 1e-6


def _cuda_or_skip():
    try:
        xgb.train({"device": "cuda", "tree_method": "hist"},
                  xgb.DMatrix(np.zeros((4, 1)), label=np.zeros(4)), 1)
    except xgb.core.XGBoostError:
        pytest.skip("no CUDA device")


def test_leaf_design_is_right_for_a_gpu_model_with_missing_values():
    """XGBoost 3.4.1's GPU pred_leaf returns the wrong leaf for some rows with
    missing values (found on adult in the Stage 1a pilot). leaf_design must
    build its design from a traversal whose leaves add up to the margin."""
    _cuda_or_skip()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6))
    X[:, :3] = np.round(X[:, :3] * 2)
    X[rng.random(size=X.shape) < 0.3] = np.nan
    y = (np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 3]) + rng.normal(size=2000) > 0).astype(float)
    d = xgb.DMatrix(X, label=y)
    bst = xgb.train({"objective": "binary:logistic", "device": "cuda", "max_depth": 5,
                     "eta": 0.1, "seed": 0, "reg_alpha": 1.0}, d, 100)
    gpu_leaf = bst.predict(d, pred_leaf=True)
    cpu = bst.copy()
    cpu.set_param({"device": "cpu"})
    assert (gpu_leaf != cpu.predict(d, pred_leaf=True)).any(), "the bug this guards no longer shows"
    design = refit.leaf_design(bst, d)
    again = refit.write_back(bst, design, design.w0)
    np.testing.assert_allclose(again.predict(d, output_margin=True),
                               bst.predict(d, output_margin=True), rtol=0, atol=1e-5)
