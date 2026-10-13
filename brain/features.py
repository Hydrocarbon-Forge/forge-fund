"""Per-ticker market snapshot (indicators over the last ~60-260 daily bars) for the brain.

Everything is computed from bars up to and including ``asof`` only (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PATH_POINTS = 12   # 60 trading days sampled every 5 days


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    up_l, dn_l = up.iloc[-1], dn.iloc[-1]
    rsi = 100 - 100 / (1 + up_l / dn_l.replace(0, np.nan))
    rsi = rsi.where(~((dn_l == 0) & (up_l > 0)), 100.0)          # only gains -> 100 (not "no data")
    return rsi.fillna(50.0)                                     # no moves / not enough history -> neutral


def compute_features(close: pd.DataFrame, volume: pd.DataFrame, asof: pd.Timestamp,
                     min_rows: int = 61) -> pd.DataFrame:
    """Return one row per ticker with return/volatility/trend/volume indicators + a 12-point path."""
    c = close.loc[:asof].tail(260)
    v = volume.reindex(c.index).fillna(0)
    ok = c.notna().sum() >= min_rows
    c = c.loc[:, ok[ok].index]
    if c.empty:
        return pd.DataFrame()
    v = v[c.columns]
    last = c.iloc[-1]
    ret = c.pct_change()
    f = pd.DataFrame(index=c.columns)
    f["r1"] = last / c.iloc[-2] - 1
    f["r5"] = last / c.iloc[-6] - 1
    f["r20"] = last / c.iloc[-21] - 1
    f["r60"] = last / c.iloc[-61] - 1
    f["vol20"] = ret.tail(20).std() * np.sqrt(252)
    f["rsi14"] = _rsi(c)
    f["d50"] = last / c.tail(50).mean() - 1
    f["d200"] = (last / c.tail(200).mean() - 1) if len(c) >= 200 else np.nan
    hi, lo = c.tail(252).max(), c.tail(252).min()
    f["p52"] = ((last - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)
    f["volx"] = (v.tail(20).mean() / v.tail(60).mean().replace(0, np.nan)).fillna(1.0)
    base = c.iloc[-61]
    path = (c.iloc[-61::5].iloc[1:] / base - 1.0)       # 12 points
    f["path"] = [[round(float(x) * 100) for x in path[t].to_numpy()] for t in c.columns]
    f["last"] = last
    return f.replace([np.inf, -np.inf], np.nan)


def index_context(close: pd.Series, asof: pd.Timestamp) -> dict[str, float]:
    c = close.loc[:asof].dropna().tail(260)
    if len(c) < 61:
        return {}
    ret = c.pct_change()
    return {
        "r1": c.iloc[-1] / c.iloc[-2] - 1, "r5": c.iloc[-1] / c.iloc[-6] - 1,
        "r20": c.iloc[-1] / c.iloc[-21] - 1, "r60": c.iloc[-1] / c.iloc[-61] - 1,
        "vol20": float(ret.tail(20).std() * np.sqrt(252)),
        "p52": float((c.iloc[-1] - c.tail(252).min()) / max(c.tail(252).max() - c.tail(252).min(), 1e-12)),
        "dd_from_high": float(c.iloc[-1] / c.tail(252).max() - 1),
        "d50": float(c.iloc[-1] / c.tail(50).mean() - 1),
        "d200": float(c.iloc[-1] / c.tail(200).mean() - 1) if len(c) >= 200 else float("nan"),
    }


def universe_regime(features: pd.DataFrame) -> dict[str, float]:
    """Breadth/dispersion statistics over the universe table - computed by code so the LLM never has to count."""
    if features.empty:
        return {}
    f = features
    return {
        "n": int(len(f)),
        "above50": float((f["d50"] > 0).mean()), "above200": float((f["d200"].dropna() > 0).mean()) if f["d200"].notna().any() else float("nan"),
        "up20": float((f["r20"] > 0).mean()), "up60": float((f["r60"] > 0).mean()),
        "med_r20": float(f["r20"].median()), "med_r60": float(f["r60"].median()),
        "disp_r20": float(f["r20"].std(ddof=0)), "med_vol20": float(f["vol20"].median()),
        "near_high": float((f["p52"] > 0.9).mean()),
    }
