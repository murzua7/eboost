"""Deliberate-mutant switch, so a broken run can be shown to exit nonzero.

EBOOST_MUTANT=log_loss   scales log_loss by 1.01
EBOOST_MUTANT=write_back writes every leaf value back 1% too large
EBOOST_MUTANT=schedule   drops the cosine floor to zero
"""

import os

import pytest

MUTANT = os.environ.get("EBOOST_MUTANT", "")


@pytest.fixture(autouse=True)
def _mutant(monkeypatch):
    if MUTANT == "log_loss":
        from eboost import metrics
        real = metrics.log_loss
        monkeypatch.setattr(metrics, "log_loss", lambda y, p: 1.01 * real(y, p))
    elif MUTANT == "write_back":
        from eboost import refit
        real = refit.write_back
        monkeypatch.setattr(refit, "write_back", lambda b, d, w: real(b, d, 1.01 * w))
    elif MUTANT == "schedule":
        from eboost import schedule
        monkeypatch.setattr(schedule, "DEFAULT_FLOOR", 0.0)
        real = schedule.cosine_eta
        monkeypatch.setattr(schedule, "cosine_eta",
                            lambda m, T, eta0, eps=0.0: real(m, T, eta0, 0.0))
    yield
