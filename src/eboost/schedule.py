"""Cosine step-size schedule with a floor (pre-registration, Method):

    eta(m) = eta0 * (eps + (1 - eps) * 0.5 * (1 + cos(pi * m / T))),  m = 0..T-1

It runs inside XGBoost through a training callback, so every tree is grown by
XGBoost's own histogram kernels.

XGBoost's own LearningRateScheduler is NOT used: it sets the rate in
after_iteration, so round 0 runs at the params' eta and round m at rate[m-1]
(xgboost 3.4.1; shown in tests/test_schedule.py). StepBeforeRound sets
rate[m] before round m is grown.
"""

from __future__ import annotations

import math

import xgboost as xgb

DEFAULT_FLOOR = 0.1


def cosine_eta(m: int, T: int, eta0: float, eps: float = DEFAULT_FLOOR) -> float:
    if not 0 <= m < T:
        raise ValueError(f"round {m} outside 0..{T - 1}")
    return eta0 * (eps + (1.0 - eps) * 0.5 * (1.0 + math.cos(math.pi * m / T)))


def schedule(T: int, eta0: float, eps: float = DEFAULT_FLOOR) -> list[float]:
    return [cosine_eta(m, T, eta0, eps) for m in range(T)]


class StepBeforeRound(xgb.callback.TrainingCallback):
    """Sets learning_rate = rates[m] before round m is grown."""

    def __init__(self, rates: list[float]):
        super().__init__()
        self.rates = rates

    def before_iteration(self, model, epoch, evals_log) -> bool:
        model.set_param("learning_rate", self.rates[epoch])
        return False


def callback(T: int, eta0: float, eps: float = DEFAULT_FLOOR) -> xgb.callback.TrainingCallback:
    return StepBeforeRound(schedule(T, eta0, eps))
