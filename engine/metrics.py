"""Performance metrics for backtest / paper equity curves. All numbers are computed from the actual
simulated equity curve - nothing is hard-coded or estimated.

Conventions: daily simple returns, 252 trading days/year, risk-free rate 0 (so Sharpe == mean/vol*sqrt(252)).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TD = 252


def _dd(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def curve_stats(eq: pd.Series) -> dict:
    """Stats for an equity curve (index = dates, values = equity)."""
    eq = eq.dropna()
    if len(eq) < 3:
        return {}
    r = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std(ddof=1) * math.sqrt(TD)
    dn = r[r < 0]
    dvol = math.sqrt((dn ** 2).sum() / max(len(r), 1)) * math.sqrt(TD)
    dd = _dd(eq)
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(TD) if r.std(ddof=1) > 0 else float("nan")
    return {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()), "days": int(len(eq)),
        "total_return": float(total), "cagr": float(cagr), "ann_vol": float(vol), "sharpe": float(sharpe),
        "sortino": float(r.mean() * TD / dvol) if dvol > 0 else float("nan"),
        "max_drawdown": float(dd.min()), "max_dd_date": str(dd.idxmin().date()),
        "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else float("nan"),
        "day_win_rate": float((r > 0).sum() / max((r != 0).sum(), 1)),
        "best_day": float(r.max()), "worst_day": float(r.min()),
    }


def regress(fund: pd.Series, bench: pd.Series) -> dict:
    """Daily-return CAPM regression of fund on benchmark (annualised alpha, beta, corr, IR)."""
    r = pd.concat([fund.pct_change(), bench.pct_change()], axis=1).dropna()
    r.columns = ["f", "b"]
    if len(r) < 20 or r["b"].var() == 0:
        return {}
    beta = r["f"].cov(r["b"]) / r["b"].var()
    alpha_d = r["f"].mean() - beta * r["b"].mean()
    resid = r["f"] - beta * r["b"] - alpha_d
    n = len(r)
    se_alpha = resid.std(ddof=2) / math.sqrt(n)
    active = r["f"] - r["b"]
    return {
        "beta": float(beta), "alpha_ann": float(alpha_d * TD),
        "alpha_tstat": float(alpha_d / se_alpha) if se_alpha > 0 else float("nan"),
        "correlation": float(r["f"].corr(r["b"])),
        "information_ratio": float(active.mean() / active.std(ddof=1) * math.sqrt(TD)) if active.std(ddof=1) > 0 else float("nan"),
        "tracking_error": float(active.std(ddof=1) * math.sqrt(TD)),
    }


def bootstrap_excess_ci(fund: pd.Series, bench: pd.Series, n: int = 2000, block: int = 10, seed: int = 11) -> dict:
    """Stationary-ish block bootstrap of the annualised mean daily excess return vs the benchmark."""
    r = pd.concat([fund.pct_change(), bench.pct_change()], axis=1).dropna()
    if len(r) < 60:
        return {}
    act = (r.iloc[:, 0] - r.iloc[:, 1]).to_numpy()
    rng = np.random.default_rng(seed)
    T = len(act)
    nb = int(np.ceil(T / block))
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T - block + 1, nb)
        means[i] = np.concatenate([act[s:s + block] for s in starts])[:T].mean() * TD
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"excess_ann_mean": float(act.mean() * TD), "ci95_low": float(lo), "ci95_high": float(hi),
            "p_excess_le_0": float((means <= 0).mean())}


def trade_stats(fills: pd.DataFrame, closed: pd.DataFrame, marks: pd.DataFrame) -> dict:
    out = {"n_fills": int(len(fills)), "n_round_trips": int(len(closed))}
    if len(closed):
        out["trade_win_rate"] = float((closed["pnl"] > 0).mean())
        wins, losses = closed.loc[closed.pnl > 0, "pnl"], closed.loc[closed.pnl <= 0, "pnl"]
        out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
        out["profit_factor"] = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("nan")
        out["avg_trade_return"] = float(closed["ret"].mean())
    if len(fills) and len(marks):
        years = max((pd.Timestamp(marks.date.iloc[-1]) - pd.Timestamp(marks.date.iloc[0])).days / 365.25, 1e-9)
        avg_eq = marks["equity"].mean()
        out["annual_turnover"] = float(fills["notional"].sum() / 2 / avg_eq / years)   # one-way
        out["total_fees"] = float(fills["fee"].sum())
        out["fees_pct_of_start"] = float(fills["fee"].sum() / marks["equity"].iloc[0])
    if len(marks):
        out["avg_exposure"] = float((marks["positions_value"] / marks["equity"]).mean())
        out["avg_positions"] = float(marks["n_positions"].mean())
    return out


def rank_ic(pred: pd.Series, realized: pd.Series) -> float:
    d = pd.concat([pred, realized], axis=1).dropna()
    if len(d) < 10:
        return float("nan")
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))
