"""Synthetic market data for tests (no network): seeded GBM bars in the store's long format."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forge.universe import SP100


def make_long(tickers: list[str], n_days: int = 420, seed: int = 1, start: str = "2021-01-04") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    frames = []
    for i, t in enumerate(tickers):
        drift = 0.0003 + 0.0001 * (i % 5)
        vol = 0.010 + 0.002 * (i % 4)
        ret = rng.normal(drift, vol, n_days)
        close = 50.0 * (1 + i % 7) * np.exp(np.cumsum(ret))
        gap = rng.normal(0, vol / 3, n_days)
        open_ = np.r_[close[0], close[:-1]] * (1 + gap)
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 4, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 4, n_days)))
        vols = rng.integers(1_000_000, 5_000_000, n_days)
        frames.append(pd.DataFrame({"date": dates, "ticker": t, "open": open_, "high": high, "low": low,
                                    "close": close, "volume": vols}))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["date"].astype("datetime64[ns]")
    return df


def universe_names(n: int = 12) -> list[str]:
    return SP100[:n]


def write_store(root, n_names: int = 12, n_days: int = 420, seed: int = 1) -> list[str]:
    """Write ohlcv_daily.parquet under ``root/data`` containing n SP100 names + SPY."""
    names = universe_names(n_names)
    df = make_long(names + ["SPY"], n_days, seed)
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "ohlcv_daily.parquet", index=False)
    return names


def panels_from_long(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {f: df.pivot(index="date", columns="ticker", values=f).sort_index() for f in ("open", "high", "low", "close", "volume")}
