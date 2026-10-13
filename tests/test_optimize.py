"""optimize/portfolio.py (skfolio sizing wrapper) and the hard limits in forge/risk.py it must obey."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from skfolio.optimization import MeanRisk

from forge.risk import check_limits
from optimize.portfolio import IC_CLAUDE, IC_KRONOS, alpha_vector, optimize_weights
from tests.synth_small import make_returns

RET = make_returns(n_days=300, n_tickers=12, seed=3)     # columns T00..T11, daily simple returns
NAMES = list(RET.columns)


def _kronos(scores: dict[str, float]) -> pd.DataFrame:
    tick = list(RET.columns)
    return pd.DataFrame({
        "ticker": tick,
        "exp_ret": [0.0] * len(tick),
        "p_up": [0.5] * len(tick),
        "dispersion": [0.1] * len(tick),
        "score": [scores.get(t, 0.0) for t in tick],
    })


def _assert_limits(weights: dict[str, float], equity: float | None = None):
    assert check_limits(weights, equity=equity) == []
    gross = sum(weights.values())
    assert gross <= 0.80 + 1e-9
    assert all(w <= 0.25 + 1e-9 for w in weights.values())
    assert len(weights) <= 10
    assert 1.0 - gross >= 0.10 - 1e-9


def test_output_obeys_hard_limits_with_more_candidates_than_slots():
    cw = {NAMES[i]: 0.09 - 0.002 * i for i in range(12)}      # 12 candidates, only 10 may be held
    res = optimize_weights(cw, RET)
    assert res.method == "mean_risk"
    _assert_limits(res.weights)
    assert set(res.weights) <= set(NAMES[:10])                 # top-10 by Claude weight only
    assert res.budget == pytest.approx(0.80)


def test_budget_follows_claude_invested_fraction():
    cw = {"T00": 0.15, "T01": 0.10, "T02": 0.10, "T03": 0.05}   # sums to 0.40 -> 60% cash requested
    res = optimize_weights(cw, RET)
    assert res.budget == pytest.approx(0.40)
    assert sum(res.weights.values()) <= 0.40 + 1e-6
    _assert_limits(res.weights)


def test_fewer_than_four_candidates_cannot_exceed_n_times_single_cap():
    cw = {"T00": 0.5, "T01": 0.5}                             # two names, each capped at 25%
    res = optimize_weights(cw, RET)
    assert sum(res.weights.values()) <= 2 * 0.25 + 1e-6
    _assert_limits(res.weights)


def test_unknown_tickers_are_dropped_and_noted():
    cw = {"NOPE": 0.30, "T00": 0.20, "T01": 0.20, "T02": 0.20, "T03": 0.10}
    res = optimize_weights(cw, RET)
    assert set(res.weights) <= set(NAMES)
    assert any("dropped" in n and "NOPE" in n for n in res.notes)
    _assert_limits(res.weights)


def test_insufficient_history_is_dropped_below_min_obs():
    ret = RET.copy()
    ret.iloc[:-50, ret.columns.get_loc("T11")] = np.nan       # only 50 observations (< MIN_OBS=120)
    cw = {"T11": 0.30, "T00": 0.20, "T01": 0.20}
    res = optimize_weights(cw, ret)
    assert "T11" not in res.weights
    assert any("T11" in n for n in res.notes)


def test_all_unknown_tickers_give_empty_method():
    res = optimize_weights({"XXX": 0.5, "YYY": 0.3}, RET)
    assert res.method == "empty"
    assert res.weights == {}
    assert any("no usable candidates" in n for n in res.notes)


def test_strong_positive_kronos_score_tilts_weight_toward_that_name():
    cw = {f"T{i:02d}": 0.10 for i in range(6)}                # equal Claude conviction -> z_claude == 0
    base = optimize_weights(cw, RET, kronos=None)
    tilted = optimize_weights(cw, RET, kronos=_kronos({"T00": 3.0, "T01": -3.0}))
    assert base.method == tilted.method == "mean_risk"
    assert tilted.weights.get("T00", 0.0) > base.weights.get("T00", 0.0)
    assert tilted.weights.get("T01", 0.0) < base.weights.get("T01", 0.0)
    _assert_limits(tilted.weights)


def test_solver_failure_falls_back_to_inverse_vol(monkeypatch):
    def boom(self, *args, **kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(MeanRisk, "fit", boom)
    cw = {"T00": 0.20, "T01": 0.15, "T02": 0.15, "T03": 0.10, "T04": 0.10}
    res = optimize_weights(cw, RET, kronos=_kronos({"T00": 1.0}))
    assert res.method == "fallback_invvol"
    assert any("skfolio failed" in n for n in res.notes)
    assert res.weights                                        # never empty when candidates exist
    _assert_limits(res.weights)
    assert sum(res.weights.values()) <= res.budget + 1e-6


def test_alpha_vector_kronos_sign_and_scale():
    sigma = pd.Series({"A": 0.010, "B": 0.020, "C": 0.030})
    flat_claude = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2})   # z_claude == 0 -> alpha from Kronos only
    kron = pd.Series({"A": 2.0, "B": -1.0, "C": 0.0})
    alpha = alpha_vector(sigma, flat_claude, kron)
    assert alpha["A"] > 0 and alpha["B"] < 0 and alpha["C"] == pytest.approx(0.0)
    assert alpha["A"] == pytest.approx(0.010 * IC_KRONOS * 2.0)
    assert alpha["B"] == pytest.approx(0.020 * IC_KRONOS * -1.0)


def test_alpha_vector_scales_with_volatility():
    sigma = pd.Series({"A": 0.010, "B": 0.020, "C": 0.030})
    claude = pd.Series({"A": 0.30, "B": 0.10, "C": 0.20})
    kron = pd.Series({"A": 1.0, "B": -0.5, "C": 0.25})
    a1 = alpha_vector(sigma, claude, kron)
    a2 = alpha_vector(2.0 * sigma, claude, kron)
    pd.testing.assert_series_equal(a2, 2.0 * a1)


def test_alpha_vector_claude_only_is_relative_conviction():
    sigma = pd.Series({"A": 0.010, "B": 0.010, "C": 0.010})
    claude = pd.Series({"A": 0.30, "B": 0.10, "C": 0.20})     # mean 0.2, population sd sqrt(2/300)
    alpha = alpha_vector(sigma, claude, None)
    z_a = 0.10 / math.sqrt(0.02 / 3)
    assert alpha["A"] == pytest.approx(0.010 * IC_CLAUDE * z_a, rel=1e-9)
    assert alpha["B"] == pytest.approx(-0.010 * IC_CLAUDE * z_a, rel=1e-9)
    assert alpha["C"] == pytest.approx(0.0, abs=1e-12)


def test_min_notional_drops_tiny_positions_for_small_equity():
    cw = {"T00": 0.22, "T01": 0.22, "T02": 0.22, "T03": 0.22, "T04": 0.06}
    big = optimize_weights(cw, RET, equity=None)
    assert big.weights["T02"] < 0.25                          # the solver sized these below the notional floor

    # equity=2000: min_notional=500 means only names at >= 25% survive; with this book none do
    small = optimize_weights(cw, RET, equity=2000.0)
    assert small.weights == {t: w for t, w in big.weights.items() if w * 2000.0 >= 500.0}
    assert check_limits(small.weights, equity=2000.0) == []

    # equity=4000: names below 12.5% are dropped, larger ones are kept untouched
    mid = optimize_weights(cw, RET, equity=4000.0)
    assert mid.weights == {t: w for t, w in big.weights.items() if w * 4000.0 >= 500.0}
    assert "T00" in mid.weights and "T02" not in mid.weights
    assert check_limits(mid.weights, equity=4000.0) == []
    assert all(w * 4000.0 >= 500.0 - 1e-6 for w in mid.weights.values())
