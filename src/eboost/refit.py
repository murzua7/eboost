"""Finalization: hold every tree's structure fixed and re-solve all leaf values
jointly (pre-registration, Method):

    min_w  sum_i loss(y_i, offset_i + sum_m w[m, leaf_m(x_i)]) + lam/2 * ||w||^2

The design matrix Z has one indicator column per leaf. Leaf values are read
from, and written back into, the booster's own JSON, and `write_back` is
gated by comparing XGBoost's predictions with offset + Z w.

Losses follow XGBoost's conventions: squared error is 0.5 * (y - f)^2,
logistic and softmax are the negative log-likelihood, summed over rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import scipy.optimize as so
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import xgboost as xgb

LOSSES = {"reg:squarederror": "squared", "binary:logistic": "logistic",
          "multi:softprob": "softmax", "multi:softmax": "softmax"}


@dataclass
class LeafDesign:
    Z: sp.csr_matrix          # n x L indicators, one column per leaf
    w0: np.ndarray            # current leaf values, length L
    col_class: np.ndarray     # output class of each column (0 unless multiclass)
    offset: np.ndarray        # n x K margin not carried by the leaves (base score)
    leaf_index: list[tuple[int, int]]  # (tree, node id) for each column
    n_class: int


def _objective_name(bst: xgb.Booster) -> str:
    cfg = json.loads(bst.save_config())
    return cfg["learner"]["objective"]["name"]


def loss_kind(bst: xgb.Booster) -> str:
    name = _objective_name(bst)
    if name not in LOSSES:
        raise ValueError(f"refit does not support objective {name}")
    return LOSSES[name]


def _on_cpu(bst: xgb.Booster) -> xgb.Booster:
    """A copy that predicts on the processor. XGBoost 3.4.1's GPU leaf lookup
    (pred_leaf on device cuda) returns the wrong leaf for some rows with
    missing values, while its GPU margin is right (adult, 5 of 39,074 rows;
    tests/test_refit.py reproduces it). The processor's traversal is the one
    whose leaf lookup and margin agree, so the design is built from it."""
    cpu = bst.copy()
    cpu.set_param({"device": "cpu"})
    return cpu


def leaf_design(bst: xgb.Booster, dmat: xgb.DMatrix) -> LeafDesign:
    bst = _on_cpu(bst)
    model = json.loads(bst.save_raw("json"))["learner"]["gradient_booster"]["model"]
    trees, tree_class = model["trees"], np.asarray(model["tree_info"])
    n_class = int(tree_class.max()) + 1
    leaf = bst.predict(dmat, pred_leaf=True).astype(np.int64)
    leaf = leaf.reshape(leaf.shape[0], -1)
    if leaf.shape[1] != len(trees):
        raise ValueError(f"pred_leaf gives {leaf.shape[1]} trees, model has {len(trees)}")
    n = leaf.shape[0]

    cols, w0, col_class, leaf_index = [], [], [], []
    start = 0
    for t, tree in enumerate(trees):
        left = np.asarray(tree["left_children"])
        node_ids = np.flatnonzero(left == -1)
        to_col = np.full(len(left), -1, dtype=np.int64)
        to_col[node_ids] = start + np.arange(len(node_ids))
        c = to_col[leaf[:, t]]
        if np.any(c < 0):
            raise ValueError(f"tree {t}: pred_leaf points at a non-leaf node")
        cols.append(c)
        w0.extend(np.asarray(tree["split_conditions"], dtype=np.float64)[node_ids])
        col_class.extend([tree_class[t]] * len(node_ids))
        leaf_index.extend((t, int(i)) for i in node_ids)
        start += len(node_ids)

    rows = np.repeat(np.arange(n), len(trees))
    Z = sp.csr_matrix((np.ones(n * len(trees)), (rows, np.stack(cols, axis=1).ravel())),
                      shape=(n, start))
    w0 = np.asarray(w0)
    col_class = np.asarray(col_class)

    margin = bst.predict(dmat, output_margin=True).astype(np.float64).reshape(n, -1)
    offset = margin - _leaf_margin(Z, w0, col_class, n_class)
    spread = np.ptp(offset, axis=0)
    if np.any(spread > 1e-4 * (1.0 + np.abs(offset).max())):
        raise ValueError(f"offset is not constant across rows (spread {spread}); the margin "
                         "carries something other than base score plus leaves")
    offset = np.broadcast_to(np.median(offset, axis=0), offset.shape).copy()
    return LeafDesign(Z, w0, col_class, offset, leaf_index, n_class)


def _leaf_margin(Z: sp.csr_matrix, w: np.ndarray, col_class: np.ndarray,
                 n_class: int) -> np.ndarray:
    out = np.empty((Z.shape[0], n_class))
    for k in range(n_class):
        out[:, k] = Z @ np.where(col_class == k, w, 0.0)
    return out


def margin(design: LeafDesign, w: np.ndarray) -> np.ndarray:
    return design.offset + _leaf_margin(design.Z, w, design.col_class, design.n_class)


def _row_grad_hess(kind: str, y: np.ndarray, f: np.ndarray):
    """Per-row loss, gradient and (for softmax) probabilities, given margins f (n x K)."""
    if kind == "squared":
        r = f[:, 0] - y
        return 0.5 * r**2, r[:, None], None
    if kind == "logistic":
        z = f[:, 0]
        p = 0.5 * (1.0 + np.tanh(0.5 * z))  # stable sigmoid
        loss = np.logaddexp(0.0, z) - y * z
        return loss, (p - y)[:, None], p[:, None]
    if kind == "softmax":
        z = f - f.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        yi = y.astype(int)
        loss = -np.log(np.clip(p[np.arange(len(y)), yi], 1e-300, None))
        g = p.copy()
        g[np.arange(len(y)), yi] -= 1.0
        return loss, g, p
    raise ValueError(kind)


class Problem:
    """Penalized refit objective with exact gradient and Hessian-vector product."""

    def __init__(self, design: LeafDesign, y: np.ndarray, kind: str, lam: float):
        self.d, self.y, self.kind, self.lam = design, np.asarray(y, dtype=np.float64), kind, lam
        self.masks = [design.col_class == k for k in range(design.n_class)]
        self.ZT = design.Z.T.tocsr()

    def _back(self, g_rows: np.ndarray) -> np.ndarray:
        """Z^T applied per class: maps an n x K row quantity to a length-L vector."""
        out = np.zeros(self.d.Z.shape[1])
        for k, m in enumerate(self.masks):
            out[m] = (self.ZT @ g_rows[:, k])[m]
        return out

    def value(self, w: np.ndarray) -> float:
        loss, _, _ = _row_grad_hess(self.kind, self.y, margin(self.d, w))
        return float(loss.sum() + 0.5 * self.lam * w @ w)

    def grad(self, w: np.ndarray) -> np.ndarray:
        _, g, _ = _row_grad_hess(self.kind, self.y, margin(self.d, w))
        return self._back(g) + self.lam * w

    def hessp(self, w: np.ndarray, v: np.ndarray) -> np.ndarray:
        _, _, p = _row_grad_hess(self.kind, self.y, margin(self.d, w))
        u = _leaf_margin(self.d.Z, v, self.d.col_class, self.d.n_class)
        if self.kind == "squared":
            hu = u
        elif self.kind == "logistic":
            hu = p * (1.0 - p) * u
        else:  # softmax: (diag(p) - p p^T) u, row by row
            hu = p * u - p * (p * u).sum(axis=1, keepdims=True)
        return self._back(hu) + self.lam * v


def full_refit(design: LeafDesign, y: np.ndarray, kind: str, lam: float,
               tol: float = 1e-8, maxiter: int = 200) -> np.ndarray:
    """Solve the penalized problem to tolerance, starting from the current leaves.
    Fails loud if the result raises the training objective, which the
    pre-registration says cannot happen."""
    prob = Problem(design, y, kind, lam)
    res = so.minimize(prob.value, design.w0, jac=prob.grad, hessp=prob.hessp,
                      method="trust-ncg", options={"gtol": tol, "maxiter": maxiter})
    if not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"full refit returned non-finite leaves ({res.message})")
    f0, f1 = prob.value(design.w0), prob.value(res.x)
    if f1 > f0 + 1e-9 * abs(f0):
        raise RuntimeError(f"full refit raised the penalized objective: {f0} -> {f1}")
    return res.x


def damped_refit(design: LeafDesign, y: np.ndarray, kind: str, lam: float,
                 step: float, cg_tol: float = 1e-10) -> np.ndarray:
    """One Newton step from the current leaves, scaled by `step` (the final
    step size eta(T-1) in the registered variant)."""
    prob = Problem(design, y, kind, lam)
    w0 = design.w0
    g = prob.grad(w0)
    H = spl.LinearOperator((len(w0), len(w0)), matvec=lambda v: prob.hessp(w0, v))
    d, info = spl.cg(H, -g, rtol=cg_tol, maxiter=10 * len(w0))
    if info != 0:
        raise RuntimeError(f"conjugate gradient did not converge (info={info})")
    return w0 + step * d


def write_back(bst: xgb.Booster, design: LeafDesign, w: np.ndarray) -> xgb.Booster:
    """A new booster whose leaf values are w; tree structures unchanged."""
    raw = json.loads(bst.save_raw("json"))
    trees = raw["learner"]["gradient_booster"]["model"]["trees"]
    for (t, node), value in zip(design.leaf_index, w, strict=True):
        trees[t]["split_conditions"][node] = float(value)
        trees[t]["base_weights"][node] = float(value)
    out = xgb.Booster()
    out.load_model(bytearray(json.dumps(raw).encode()))
    return out
