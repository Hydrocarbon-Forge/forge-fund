"""skfolio wrapper: Claude's picks + Kronos signal + return history -> position weights.

Division of labour
------------------
* Claude decides WHAT to hold and how much cash to keep (``claude_weights``; sum = 1 - cash).
* Kronos + Claude conviction become a modest alpha vector (Grinold-style: ``mu_i = sigma_i *
  (IC_k * z_kronos_i + IC_c * z_claude_i)`` with deliberately small assumed ICs, so noisy
  forecasts can tilt but never dominate).
* skfolio ``MeanRisk`` (max utility, variance risk, Ledoit-Wolf covariance, long-only, per-name cap,
  budget = invested fraction) turns that into sized weights.
* ``forge.risk.clamp_weights`` then re-imposes every hard limit - the optimizer is not trusted.

Fallbacks (never raise into the pipeline): solver failure -> inverse-vol weights over the same
candidates; no usable candidates -> empty weights (caller holds current portfolio).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn.utils.validation as skv

from forge.config import LIMITS, Limits
from forge.risk import check_limits, clamp_weights, max_investable

log = logging.getLogger("forge.optimize")

IC_KRONOS = 0.03      # assumed (small) information coefficient for the Kronos z-score
IC_CLAUDE = 0.04      # assumed information coefficient for Claude's relative conviction
RISK_AVERSION = 30.0  # utility = mu'w - gamma * w'Sigma w  (daily units)
WINDOW = 252
MIN_OBS = 120


def _fixed_mu_cls():
    from skfolio.moments import BaseMu

    class FixedMu(BaseMu):
        """Expected-return 'estimator' that returns a caller-supplied vector."""

        def __init__(self, mu=None):
            self.mu = mu

        def fit(self, X, y=None):
            X = skv.validate_data(self, X)
            mu = np.asarray(self.mu, dtype=float)
            if mu.shape != (X.shape[1],):
                raise ValueError(f"mu has shape {mu.shape}, expected ({X.shape[1]},)")
            self.mu_ = mu
            return self

    return FixedMu


@dataclass
class OptResult:
    weights: dict[str, float]
    method: str                      # mean_risk | fallback_invvol | empty
    budget: float = 0.0
    mu: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return s * 0.0
    return ((s - s.mean()) / sd).clip(-3, 3)


def alpha_vector(sigma: pd.Series, claude_w: pd.Series, kronos_score: pd.Series | None,
                 ic_kronos: float = IC_KRONOS, ic_claude: float = IC_CLAUDE) -> pd.Series:
    """Daily expected-return vector (Grinold: alpha = IC * sigma * z)."""
    z_c = _zscore(claude_w.reindex(sigma.index).fillna(0.0))
    z_k = pd.Series(0.0, index=sigma.index)
    if kronos_score is not None and len(kronos_score):
        z_k = kronos_score.reindex(sigma.index).fillna(0.0).clip(-3, 3)
    return sigma * (ic_kronos * z_k + ic_claude * z_c)


def optimize_weights(claude_weights: dict[str, float], returns: pd.DataFrame,
                     kronos: pd.DataFrame | None = None, limits: Limits = LIMITS,
                     equity: float | None = None, risk_aversion: float = RISK_AVERSION) -> OptResult:
    """Size Claude's picks. ``returns``: daily simple returns (index=date, cols=tickers), as-of data only."""
    notes: list[str] = []
    cw = {t.upper(): float(w) for t, w in claude_weights.items() if w and w > 0}
    cands = [t for t in sorted(cw, key=cw.get, reverse=True) if t in returns.columns][: limits.max_positions]
    window = returns.tail(WINDOW)
    cands = [t for t in cands if window[t].notna().sum() >= MIN_OBS]
    dropped = sorted(set(cw) - set(cands))
    if dropped:
        notes.append(f"dropped (no data/not in universe): {dropped}")
    if not cands:
        return OptResult({}, "empty", 0.0, notes=notes + ["no usable candidates"])

    X = window[cands].fillna(0.0)
    claude_s = pd.Series({t: cw[t] for t in cands})
    budget = min(float(claude_s.sum()), max_investable(limits), len(cands) * limits.max_single)
    if budget <= 0.0:
        return OptResult({}, "empty", 0.0, notes=notes + ["zero budget"])
    sigma = X.std(ddof=0)
    kscore = None
    if kronos is not None and len(kronos):
        kscore = kronos.drop_duplicates("ticker").set_index("ticker")["score"]
    mu = alpha_vector(sigma, claude_s, kscore)

    weights: dict[str, float] = {}
    method = "mean_risk"
    try:
        from skfolio import RiskMeasure
        from skfolio.optimization import MeanRisk, ObjectiveFunction
        from skfolio.moments import LedoitWolf
        from skfolio.prior import EmpiricalPrior

        model = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            risk_measure=RiskMeasure.VARIANCE,
            risk_aversion=risk_aversion,
            min_weights=0.0,
            max_weights=limits.max_single,
            budget=budget,
            prior_estimator=EmpiricalPrior(mu_estimator=_fixed_mu_cls()(mu=mu.to_numpy()),
                                           covariance_estimator=LedoitWolf()),
            raise_on_failure=True,
        )
        model.fit(X)
        w = pd.Series(model.weights_, index=cands)
        if not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("non-finite or empty optimizer weights")
        weights = {t: float(v) for t, v in w.items() if v > 1e-6}
    except Exception as e:  # noqa: BLE001 - optimizer must never take the pipeline down
        method = "fallback_invvol"
        notes.append(f"skfolio failed ({type(e).__name__}: {str(e)[:120]}); using inverse-vol fallback")
        inv = 1.0 / sigma.replace(0, np.nan).fillna(sigma.median() or 1.0)
        weights = (inv / inv.sum() * budget).to_dict()

    final = clamp_weights(weights, limits, equity)
    bad = check_limits(final, limits, equity)
    if bad:                                  # defence in depth: should be impossible
        notes.append(f"limit violations after clamp: {bad}")
        final = {}
        method = "empty"
    return OptResult(final, method, budget, {t: float(v) for t, v in mu.items()}, notes)
