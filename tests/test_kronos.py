"""Kronos wrapper: cache semantics + a real CPU-model smoke test on synthetic bars."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forge.config import KRONOS, get_paths
from kronos.forecast import (KronosForecaster, cfg_key, forecast_path, forecast_with_cache, load_cached,
                             save_cached)

ASOF = pd.Timestamp("2025-03-14")


def table(tickers=("AAA", "BBB")):
    return pd.DataFrame({"ticker": list(tickers), "exp_ret": [0.01, -0.01], "p_up": [0.7, 0.3],
                         "dispersion": [0.01, 0.02], "score": [1.0, -1.0], "asof": ASOF, "model": "Kronos-mini"})


def test_cfg_key_depends_on_settings():
    assert cfg_key(KRONOS) == cfg_key(replace(KRONOS))
    assert cfg_key(KRONOS) != cfg_key(replace(KRONOS, lookback=64))
    assert cfg_key(KRONOS) != cfg_key(replace(KRONOS, sample_count=9))


def test_cache_roundtrip_and_invalidation():
    assert load_cached(ASOF) is None
    p = save_cached(table(), ASOF)
    assert p == forecast_path(ASOF) and p.name == "forecasts_20250314.parquet"
    got = load_cached(ASOF)
    assert list(got["ticker"]) == ["AAA", "BBB"]
    assert load_cached(ASOF, replace(KRONOS, lookback=64)) is None            # config changed -> stale
    p.write_bytes(b"corrupt")
    assert load_cached(ASOF) is None                                          # corrupt -> recompute


class FakeForecaster:
    s = KRONOS

    def __init__(self):
        self.calls = 0

    def forecast(self, histories, asof):
        self.calls += 1
        return table(tuple(histories))


def hist(n=KRONOS.lookback + 5):
    idx = pd.bdate_range(end=ASOF, periods=n)
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx)


def test_forecast_with_cache_computes_once():
    f = FakeForecaster()
    h = {"AAA": hist(), "BBB": hist()}
    forecast_with_cache(f, h, ASOF)
    forecast_with_cache(f, h, ASOF)
    assert f.calls == 1


def test_forecast_with_cache_waits_for_precompute_process(monkeypatch):
    import kronos.forecast as kf
    f = FakeForecaster()
    h = {"AAA": hist(), "BBB": hist()}
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        save_cached(table(), ASOF)                       # the "other process" publishes the file while we wait

    monkeypatch.setattr(kf.time, "sleep", fake_sleep)
    out = forecast_with_cache(f, h, ASOF, wait_s=60)
    assert f.calls == 0 and len(sleeps) == 1 and len(out) == 2


def _models_present():
    p = get_paths()
    return (p.vendor / "Kronos" / "model" / "kronos.py").exists() and \
        (p.models / KRONOS.model_name / "model.safetensors").exists() and \
        (p.models / KRONOS.tokenizer_name / "model.safetensors").exists()


@pytest.mark.skipif(not _models_present(), reason="Kronos weights/source not installed (run scripts/setup.sh)")
def test_real_model_smoke_cpu():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end=ASOF, periods=KRONOS.lookback)

    def bars(drift):
        c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(idx))))
        return pd.DataFrame({"open": c * 0.999, "high": c * 1.005, "low": c * 0.995, "close": c,
                             "volume": rng.integers(1e6, 3e6, len(idx))}, index=idx)

    h = {"UP": bars(0.002), "FLAT": bars(0.0), "DOWN": bars(-0.002), "SHORT": bars(0.0).iloc[:50]}
    kf = KronosForecaster(replace(KRONOS, sample_count=2))
    out = kf.forecast(h, ASOF)
    assert set(out.ticker) == {"UP", "FLAT", "DOWN"}                          # short history skipped
    assert out["p_up"].between(0, 1).all() and (out["dispersion"] >= 0).all()
    assert abs(out["score"].mean()) < 1e-6 and out["exp_ret"].abs().max() < 0.5
    out2 = KronosForecaster(replace(KRONOS, sample_count=2)).forecast(h, ASOF)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), out2.reset_index(drop=True))   # seeded => deterministic
