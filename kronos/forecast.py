"""Kronos (NeoQuasar/Kronos-mini, CPU) forecasting wrapper -> cross-sectional signals.

Runs on a schedule (nightly / once per rebalance), NEVER per tick. Output per ticker:
``exp_ret`` (mean cumulative return over the horizon), ``p_up`` (share of sampled paths that
end up), ``dispersion`` (std of sampled cumulative returns) and ``score`` (cross-sectional
z-score of ``exp_ret``). Forecast tables are cached as ``data/forecasts_YYYYMMDD.parquet`` so
backtests are resumable and deterministic (seeded per as-of date).

The upstream ``KronosPredictor`` averages the sampled paths internally; we need the individual
samples for ``p_up``/``dispersion``, so ``_SamplingPredictor`` re-implements the generation loop
(adapted from shiyu-coder/Kronos ``auto_regressive_inference``, MIT) without the final mean.
Vendored source is cloned to ``vendor/Kronos`` (pinned commit, see scripts/setup.sh).
"""
from __future__ import annotations

import hashlib
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from forge.config import KRONOS, KronosSettings, get_paths

log = logging.getLogger("forge.kronos")

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def cfg_key(s: KronosSettings = KRONOS) -> str:
    """Short hash of everything that changes forecast values (cache invalidation key)."""
    return hashlib.sha256(repr(sorted(asdict(s).items())).encode()).hexdigest()[:10]


def forecast_path(asof: pd.Timestamp, data_dir: Path | None = None) -> Path:
    data_dir = data_dir or get_paths().data
    return data_dir / f"forecasts_{pd.Timestamp(asof):%Y%m%d}.parquet"


def _import_vendor():
    vendor = get_paths().vendor / "Kronos"
    if not (vendor / "model" / "kronos.py").exists():
        raise RuntimeError(f"Kronos source missing at {vendor}; run scripts/setup.sh")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    import model as kmodel  # noqa: PLC0415  (vendored top-level package named 'model')
    from model import kronos as kmod  # noqa: PLC0415
    return kmodel, kmod


class _SamplingPredictor:
    """KronosPredictor subclass returning un-averaged sample paths. Built lazily (needs vendor)."""

    @staticmethod
    def build(model, tokenizer, max_context: int, device: str = "cpu"):
        kmodel, kmod = _import_vendor()
        import torch

        class Sampling(kmodel.KronosPredictor):
            last_samples: np.ndarray | None = None

            def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
                xt = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
                xs = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
                ys = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)
                z = _ar_samples(kmod, torch, self.tokenizer, self.model, xt, xs, ys, self.max_context,
                                pred_len, self.clip, T, top_k, top_p, sample_count)
                z = z[:, :, -pred_len:, :]            # (B, S, pred_len, F)
                self.last_samples = z
                return z.mean(axis=1)

        return Sampling(model, tokenizer, device=device, max_context=max_context)


def _ar_samples(kmod, torch, tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len,
                clip, T, top_k, top_p, sample_count) -> np.ndarray:
    """Adapted from Kronos ``auto_regressive_inference``; returns (B, S, L, F) without averaging."""
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device
        rep = lambda t: t.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, t.size(1), t.size(2)).to(device)  # noqa: E731
        x, x_stamp, y_stamp = rep(x), rep(x_stamp), rep(y_stamp)
        x_token = tokenizer.encode(x, half=True)
        init_len = x.size(1)
        bsz = x_token[0].size(0)
        total = init_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
        gen_pre = x_token[0].new_empty(bsz, pred_len)
        gen_post = x_token[1].new_empty(bsz, pred_len)
        pre_buf = x_token[0].new_zeros(bsz, max_context)
        post_buf = x_token[1].new_zeros(bsz, max_context)
        buf_len = min(init_len, max_context)
        start = max(0, init_len - max_context)
        pre_buf[:, :buf_len] = x_token[0][:, start:start + buf_len]
        post_buf[:, :buf_len] = x_token[1][:, start:start + buf_len]
        for i in range(pred_len):
            cur = init_len + i
            wl = min(cur, max_context)
            inp = [pre_buf[:, :wl], post_buf[:, :wl]] if cur <= max_context else [pre_buf, post_buf]
            stamp = full_stamp[:, max(0, cur - max_context):cur, :].contiguous()
            s1_logits, ctx = model.decode_s1(inp[0], inp[1], stamp)
            s_pre = kmod.sample_from_logits(s1_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
            s2_logits = model.decode_s2(ctx, s_pre)
            s_post = kmod.sample_from_logits(s2_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
            gen_pre[:, i] = s_pre.squeeze(-1)
            gen_post[:, i] = s_post.squeeze(-1)
            if cur < max_context:
                pre_buf[:, cur] = s_pre.squeeze(-1)
                post_buf[:, cur] = s_post.squeeze(-1)
            else:
                pre_buf.copy_(torch.roll(pre_buf, shifts=-1, dims=1))
                post_buf.copy_(torch.roll(post_buf, shifts=-1, dims=1))
                pre_buf[:, -1] = s_pre.squeeze(-1)
                post_buf[:, -1] = s_post.squeeze(-1)
        full_pre = torch.cat([x_token[0], gen_pre], dim=1)
        full_post = torch.cat([x_token[1], gen_post], dim=1)
        cs = max(0, total - max_context)
        z = tokenizer.decode([full_pre[:, cs:total].contiguous(), full_post[:, cs:total].contiguous()], half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        return z.cpu().numpy()


class KronosForecaster:
    """Loads the CPU model once; ``forecast()`` turns OHLCV histories into signal rows."""

    def __init__(self, settings: KronosSettings = KRONOS, models_dir: Path | None = None):
        self.s = settings
        self.models_dir = models_dir or get_paths().models
        self._predictor = None

    # -- model loading -----------------------------------------------------------------------
    def _load(self):
        if self._predictor is not None:
            return self._predictor
        import os
        import torch
        torch.set_num_threads(os.cpu_count() or 4)
        kmodel, _ = _import_vendor()
        tok_dir = self.models_dir / self.s.tokenizer_name
        mdl_dir = self.models_dir / self.s.model_name
        for d in (tok_dir, mdl_dir):
            if not (d / "model.safetensors").exists():
                raise RuntimeError(f"Kronos weights missing: {d} (run scripts/setup.sh)")
        tokenizer = kmodel.KronosTokenizer.from_pretrained(str(tok_dir))
        model = kmodel.Kronos.from_pretrained(str(mdl_dir))
        model.eval()
        tokenizer.eval()
        self._predictor = _SamplingPredictor.build(model, tokenizer, self.s.max_context, device="cpu")
        return self._predictor

    # -- forecasting -------------------------------------------------------------------------
    def forecast(self, histories: dict[str, pd.DataFrame], asof: pd.Timestamp) -> pd.DataFrame:
        """``histories``: ticker -> OHLCV frame indexed by date, last row == ``asof``.

        Returns one row per forecastable ticker (>= lookback clean bars):
        ticker, exp_ret, p_up, dispersion, score, asof, model.
        """
        import torch
        asof = pd.Timestamp(asof).normalize()
        lb, h = self.s.lookback, self.s.pred_len
        usable: list[tuple[str, pd.DataFrame]] = []
        for t, df in histories.items():
            d = df.loc[:asof].tail(lb)
            if len(d) < lb or d[["open", "high", "low", "close", "volume"]].isna().any().any():
                continue
            usable.append((t, d))
        if not usable:
            return pd.DataFrame(columns=["ticker", "exp_ret", "p_up", "dispersion", "score", "asof", "model"])

        predictor = self._load()
        y_ts = pd.Series(pd.bdate_range(asof + pd.Timedelta(days=1), periods=h))
        rows = []
        t0 = time.time()
        torch.manual_seed(self.s.seed + int(asof.strftime("%Y%m%d")))
        for i in range(0, len(usable), self.s.batch_size):
            chunk = usable[i:i + self.s.batch_size]
            dfs, xs, ys = [], [], []
            for t, d in chunk:
                x = d[["open", "high", "low", "close", "volume"]].astype("float64").copy()
                x["amount"] = x["close"] * x["volume"]
                dfs.append(x.reset_index(drop=True))
                xs.append(pd.Series(pd.to_datetime(d.index)))
                ys.append(y_ts)
            predictor.predict_batch(dfs, xs, ys, pred_len=h, T=self.s.temperature, top_p=self.s.top_p,
                                    sample_count=self.s.sample_count, verbose=False)
            samples = predictor.last_samples            # (B, S, H, F) normalised
            for j, (t, d) in enumerate(chunk):
                raw = dfs[j][FEATURES].to_numpy(dtype="float64")
                mean, std = raw.mean(axis=0), raw.std(axis=0)
                paths = samples[j].astype("float64") * (std + 1e-5) + mean      # (S, H, F)
                last_close = float(d["close"].iloc[-1])
                end_close = paths[:, -1, 3]
                rets = end_close / last_close - 1.0
                rows.append({"ticker": t, "exp_ret": float(rets.mean()),
                             "p_up": float((rets > 0).mean()), "dispersion": float(rets.std())})
        out = pd.DataFrame(rows)
        sd = out["exp_ret"].std(ddof=0)
        out["score"] = (out["exp_ret"] - out["exp_ret"].mean()) / (sd if sd > 1e-12 else 1.0)
        out["asof"] = asof
        out["model"] = self.s.model_name
        log.info("kronos: %d tickers forecast in %.1fs (asof %s)", len(out), time.time() - t0, asof.date())
        return out


# ---------------------------------------------------------------------------- caching
def load_cached(asof: pd.Timestamp, settings: KronosSettings = KRONOS, data_dir: Path | None = None) -> pd.DataFrame | None:
    p = forecast_path(asof, data_dir)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:  # noqa: BLE001 - corrupt cache => recompute
        return None
    if "cfg" not in df.columns or df["cfg"].iloc[0] != cfg_key(settings):
        return None
    return df


def save_cached(df: pd.DataFrame, asof: pd.Timestamp, settings: KronosSettings = KRONOS,
                data_dir: Path | None = None) -> Path:
    p = forecast_path(asof, data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["cfg"] = cfg_key(settings)
    tmp = p.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(p)
    return p


def forecast_with_cache(forecaster: KronosForecaster, histories: dict[str, pd.DataFrame],
                        asof: pd.Timestamp, data_dir: Path | None = None, wait_s: float = 0.0) -> pd.DataFrame:
    """Cached forecast for ``asof``. With ``wait_s`` > 0, first wait for a separate precompute process
    (``run_backtest --phase kronos``) to publish the file, then fall back to computing it here."""
    deadline = time.time() + wait_s
    while True:
        cached = load_cached(asof, forecaster.s, data_dir)
        if cached is not None:
            want = [t for t in histories if len(histories[t].loc[:asof]) >= forecaster.s.lookback]
            if set(cached["ticker"]) <= set(histories) and len(cached) >= 0.9 * len(want):
                return cached.drop(columns=["cfg"])
        if time.time() >= deadline:
            break
        time.sleep(5)
    df = forecaster.forecast(histories, asof)
    save_cached(df, asof, forecaster.s, data_dir)
    return df
