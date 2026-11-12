"""Golden tests: every expected value below was computed by hand."""

import json
import math

import numpy as np
import pytest

from eboost import metrics as M
from eboost import runlog


def test_log_loss_binary_golden():
    # -(ln 0.8 + ln 0.7) / 2
    expected = (0.2231435513142097 + 0.35667494393873245) / 2
    assert M.log_loss(np.array([1, 0]), np.array([0.8, 0.3])) == pytest.approx(expected, rel=1e-12)


def test_log_loss_multiclass_golden():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    expected = -(math.log(0.7) + math.log(0.8)) / 2
    assert M.log_loss(np.array([0, 2]), p) == pytest.approx(expected, rel=1e-12)


def test_mse_golden():
    assert M.mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])) == pytest.approx(5 / 3)


def test_regret_golden():
    assert M.regret(1.05, 1.0) == pytest.approx(0.05)
    assert M.regret(0.9, 1.2) == pytest.approx(-0.25)
    with pytest.raises(ValueError):
        M.regret(1.0, 0.0)


def test_row_passes_and_proxy_golden():
    assert M.row_passes(rounds=100, rows_per_tree=5000, n=10000) == 50.0
    assert M.hist_proxy(histogram_builds=3, rows=1000, features=7) == 21000.0


def test_compute_to_match_golden():
    costs = np.array([10.0, 20.0, 40.0, 80.0])
    losses = np.array([0.50, 0.42, 0.40, 0.39])
    hit = M.compute_to_match(costs, losses, target=0.41, full_budget=100.0)
    assert hit == M.ComputeToMatch(cost=40.0, censored=False)
    miss = M.compute_to_match(costs, losses, target=0.30, full_budget=100.0)
    assert miss == M.ComputeToMatch(cost=100.0, censored=True)


def test_rolling_origins_golden():
    # 20 rows, 3 origins, test windows of 2 rows: tests are [14,16), [16,18), [18,20)
    w = M.rolling_origins(n=20, n_origins=3, test_len=2, min_train=10)
    assert [(x.train_end, x.test_start, x.test_end) for x in w] == [(14, 14, 16), (16, 16, 18),
                                                                   (18, 18, 20)]
    with pytest.raises(ValueError):
        M.rolling_origins(n=20, n_origins=6, test_len=2, min_train=10)


def test_runlog_is_append_only_and_hash_ignores_key_order(tmp_path):
    assert runlog.config_hash({"a": 1, "b": 2}) == runlog.config_hash({"b": 2, "a": 1})
    log = tmp_path / "runs.jsonl"
    runlog.append(log, {"config": {"arm": "A0"}, "seed": 0})
    runlog.append(log, {"config": {"arm": "A1"}, "seed": 1})
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["seed"] == 1
    assert {"git", "hardware", "versions", "config_hash"} <= set(json.loads(lines[0]))
