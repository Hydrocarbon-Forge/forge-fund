"""brain/features.py: indicator maths, minimum-history filter, 12-point path, no look-ahead."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from brain.features import PATH_POINTS, compute_features, index_context
from tests.synth_small import make_close, make_volume

FULL = ["AAA", "BBB", "CCC", "DDD"]
SHORT = "SHORT"
COLS = ["r1", "r5", "r20", "r60", "vol20", "rsi14", "d50", "d200", "p52", "volx", "path", "last"]


@pytest.fixture
def panel():
    close = make_close(FULL + [SHORT], n_days=300, seed=7)
    close.loc[close.index[:-40], SHORT] = np.nan      # SHORT has only 40 bars of history
    volume = make_volume(close, seed=8)
    return close, volume


def test_columns_and_universe_filter(panel):
    close, volume = panel
    f = compute_features(close, volume, close.index[-1])
    assert list(f.columns) == COLS
    assert sorted(f.index) == sorted(FULL)


def test_ticker_with_too_little_history_is_excluded(panel):
    close, volume = panel
    f = compute_features(close, volume, close.index[-1])
    assert SHORT not in f.index


def test_min_history_boundary_is_61_rows():
    close = make_close(["SIXTY", "SIXTYONE"], n_days=300, seed=2)
    close.loc[close.index[:-60], "SIXTY"] = np.nan        # 60 valid rows -> excluded
    close.loc[close.index[:-61], "SIXTYONE"] = np.nan     # 61 valid rows -> included
    volume = make_volume(close, seed=2)
    f = compute_features(close, volume, close.index[-1])
    assert list(f.index) == ["SIXTYONE"]


def test_returns_match_hand_computation(panel):
    close, volume = panel
    f = compute_features(close, volume, close.index[-1])
    s = close["BBB"]
    row = f.loc["BBB"]
    assert row["r1"] == pytest.approx(s.iloc[-1] / s.iloc[-2] - 1, rel=1e-12)
    assert row["r5"] == pytest.approx(s.iloc[-1] / s.iloc[-6] - 1, rel=1e-12)
    assert row["r20"] == pytest.approx(s.iloc[-1] / s.iloc[-21] - 1, rel=1e-12)
    assert row["r60"] == pytest.approx(s.iloc[-1] / s.iloc[-61] - 1, rel=1e-12)
    assert row["last"] == pytest.approx(s.iloc[-1], rel=1e-12)


def test_trend_volatility_and_volume_match_hand_computation(panel):
    close, volume = panel
    f = compute_features(close, volume, close.index[-1])
    s, v = close["BBB"], volume["BBB"]
    row = f.loc["BBB"]
    # 20-day sample std of simple returns, annualised with sqrt(252)
    assert row["vol20"] == pytest.approx(s.pct_change().iloc[-20:].std(ddof=1) * math.sqrt(252), rel=1e-12)
    assert row["d50"] == pytest.approx(s.iloc[-1] / s.iloc[-50:].mean() - 1, rel=1e-12)
    assert row["d200"] == pytest.approx(s.iloc[-1] / s.iloc[-200:].mean() - 1, rel=1e-12)
    last252 = s.iloc[-252:]
    assert row["p52"] == pytest.approx((s.iloc[-1] - last252.min()) / (last252.max() - last252.min()), rel=1e-12)
    assert row["volx"] == pytest.approx(v.iloc[-20:].mean() / v.iloc[-60:].mean(), rel=1e-12)


def test_path_has_12_integer_points_sampled_every_5_days(panel):
    close, volume = panel
    f = compute_features(close, volume, close.index[-1])
    assert PATH_POINTS == 12
    for t in FULL:
        path = f.loc[t, "path"]
        assert len(path) == 12
        assert all(type(p) is int for p in path)
        s = close[t]
        expected = [round((s.iloc[-61 + 5 * k] / s.iloc[-61] - 1) * 100) for k in range(1, 13)]
        assert path == expected
        # the final path point is the 60-day change (rounded to whole percent)
        assert path[-1] == round(f.loc[t, "r60"] * 100)


def test_p52_and_rsi_ranges(panel):
    close, volume = panel
    f = compute_features(close, volume, close.index[-1])
    assert ((f["p52"] >= 0) & (f["p52"] <= 1)).all()
    assert ((f["rsi14"] >= 0) & (f["rsi14"] <= 100)).all()
    assert np.isfinite(f[["r1", "r5", "r20", "r60", "vol20", "rsi14", "d50", "d200", "p52", "volx"]].to_numpy()).all()


def test_rsi_of_pure_downtrend_is_zero():
    idx = pd.bdate_range("2024-01-02", periods=120)
    close = pd.DataFrame({"DN": 400.0 - np.arange(120.0)}, index=idx)
    volume = pd.DataFrame({"DN": 1e6}, index=idx)
    f = compute_features(close, volume, idx[-1])
    assert f.loc["DN", "rsi14"] == pytest.approx(0.0)


def test_rsi_of_pure_uptrend_is_100():
    idx = pd.bdate_range("2024-01-02", periods=120)
    close = pd.DataFrame({"UP": 100.0 + np.arange(120.0)}, index=idx)
    volume = pd.DataFrame({"UP": 1e6}, index=idx)
    f = compute_features(close, volume, idx[-1])
    assert f.loc["UP", "rsi14"] == pytest.approx(100.0)


def test_no_lookahead_features_ignore_data_after_asof(panel):
    close, volume = panel
    asof = close.index[200]
    before = compute_features(close, volume, asof)

    rng = np.random.default_rng(99)
    after = close.index > asof
    corrupted_close = close.copy()
    corrupted_volume = volume.copy()
    corrupted_close.loc[after, :] = rng.uniform(1.0, 1000.0, size=(int(after.sum()), close.shape[1]))
    corrupted_volume.loc[after, :] = rng.uniform(1e9, 1e12, size=(int(after.sum()), volume.shape[1]))

    after_f = compute_features(corrupted_close, corrupted_volume, asof)
    pd.testing.assert_frame_equal(before, after_f)


def test_index_context_keys_and_drawdown(panel):
    close, _ = panel
    asof = close.index[-1]
    ctx = index_context(close["AAA"], asof)
    assert set(ctx) == {"r1", "r5", "r20", "r60", "vol20", "p52", "dd_from_high"}
    assert ctx["dd_from_high"] <= 0
    s = close["AAA"]
    assert ctx["dd_from_high"] == pytest.approx(s.iloc[-1] / s.iloc[-252:].max() - 1, rel=1e-12)
    assert 0.0 <= ctx["p52"] <= 1.0


def test_index_context_no_lookahead(panel):
    close, _ = panel
    asof = close.index[200]
    before = index_context(close["AAA"], asof)
    corrupted = close["AAA"].copy()
    corrupted.loc[corrupted.index > asof] = 1e6
    assert index_context(corrupted, asof) == before


def test_index_context_too_short_returns_empty():
    idx = pd.bdate_range("2024-01-02", periods=30)
    s = pd.Series(np.linspace(100, 110, 30), index=idx)
    assert index_context(s, idx[-1]) == {}
