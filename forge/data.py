"""Daily OHLCV data: yfinance download, cleaning, parquet store, wide panels and latest prices.

The store is one long-format parquet file (``get_paths().data / ohlcv_daily.parquet``) plus a
``meta.json`` sidecar. Writes go to a temp file first and are swapped in with ``os.replace``.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from forge.config import get_paths
from forge.universe import BENCHMARK, SP100

log = logging.getLogger("forge.data")

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
OHLCV = [*PRICE_COLS, "volume"]
DEFAULT_START = "2020-12-01"
STORE_NAME = "ohlcv_daily.parquet"
META_NAME = "meta.json"
MAX_FFILL = 3                       # trading days a missing bar may be forward-filled
LIVE_MIN_COVERAGE = 0.8             # share of tickers that must have a live 1m price
LIVE_MAX_AGE = pd.Timedelta(minutes=20)
EXCHANGE_TZ = "America/New_York"    # naive intraday timestamps from yfinance are exchange-local


# ---------------------------------------------------------------- download & clean
def fetch_history(tickers: Sequence[str], start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV for ``tickers`` and return cleaned long-format data.

    Tickers with no usable rows are absent from the result (``update_store`` reports them).
    """
    symbols = list(dict.fromkeys(tickers))
    if not symbols:
        return _empty_long()
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False,
                      threads=True, group_by="column")
    frames = []
    for symbol, bars in _split_by_ticker(raw, symbols).items():
        part = bars[OHLCV].set_axis(_naive_ns(bars.index)).rename_axis("date").reset_index()
        frames.append(part.assign(ticker=symbol))
    if not frames:
        return _empty_long()
    return _clean(pd.concat(frames, ignore_index=True))


def _split_by_ticker(raw: pd.DataFrame, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Split a yfinance download into one frame per ticker with lower-case field columns."""
    if raw is None or raw.empty:
        return {}
    if isinstance(raw.columns, pd.MultiIndex):  # group_by="column": (field, ticker)
        present = set(raw.columns.get_level_values(1))
        parts = {t: raw.xs(t, axis=1, level=1) for t in symbols if t in present}
    else:  # defensive: flat columns only come back for a single symbol
        parts = {symbols[0]: raw} if len(symbols) == 1 else {}
    return {t: frame.rename(columns=str.lower) for t, frame in parts.items()}


def _naive_ns(index: pd.Index) -> pd.DatetimeIndex:
    """Tz-naive, nanosecond-resolution DatetimeIndex (the store's ``date`` dtype)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.as_unit("ns")


def _clean(long: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN/non-positive bars, enforce OHLC consistency, dedupe, sort by (ticker, date)."""
    df = long.astype({c: "float64" for c in OHLCV})
    df = df.dropna(subset=PRICE_COLS)
    df = df[(df[PRICE_COLS] > 0).all(axis=1)].copy()
    df["high"] = pd.concat([df["high"], df["open"], df["close"]], axis=1).max(axis=1)
    df["low"] = pd.concat([df["low"], df["open"], df["close"]], axis=1).min(axis=1)
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    return df.sort_values(["ticker", "date"], ignore_index=True)[COLUMNS]


def _empty_long() -> pd.DataFrame:
    dtypes = {"date": "datetime64[ns]", "ticker": "str", **{c: "float64" for c in PRICE_COLS}, "volume": "int64"}
    return pd.DataFrame(columns=COLUMNS).astype(dtypes)


# ------------------------------------------------------------------------- store
def update_store(tickers: Sequence[str] | None = None, start: str = DEFAULT_START,
                 store_dir: Path | str | None = None) -> dict:
    """Re-download history and atomically replace the parquet store and ``meta.json``.

    ``tickers`` defaults to ``SP100 + [BENCHMARK]``; if given, it is used as-is (include SPY yourself).
    If nothing usable is downloaded, raises ``RuntimeError`` and leaves the existing store untouched.
    Returns the meta dict that was written.
    """
    symbols = list(dict.fromkeys([*SP100, BENCHMARK] if tickers is None else tickers))
    target = Path(store_dir) if store_dir is not None else get_paths().data
    target.mkdir(parents=True, exist_ok=True)
    try:
        df = fetch_history(symbols, start=start)
    except Exception as exc:
        raise RuntimeError(f"history download failed: {exc}") from exc
    if df.empty:
        raise RuntimeError("download returned no usable rows; existing store left untouched")

    present = set(df["ticker"])
    dropped = [t for t in symbols if t not in present]
    meta = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": start,
        "last_date": df["date"].max().strftime("%Y-%m-%d"),
        "n_tickers": len(present),
        "dropped": dropped,
    }
    _atomic_write(target / STORE_NAME, lambda p: df.to_parquet(p, index=False))
    _atomic_write(target / META_NAME, lambda p: p.write_text(json.dumps(meta, indent=2)))
    log.info("store updated: %d rows, %d tickers, last_date=%s, dropped=%s",
             len(df), len(present), meta["last_date"], dropped)
    return meta


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """Write via a sibling temp file, then ``os.replace`` so readers never see a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _store_path() -> Path:
    return get_paths().data / STORE_NAME


def load_store() -> pd.DataFrame:
    """Full long-format store. Raises ``FileNotFoundError`` if ``update_store`` never ran."""
    path = _store_path()
    if not path.exists():
        raise FileNotFoundError(f"no price store at {path}; run update_store() first")
    return pd.read_parquet(path)


def last_store_date() -> pd.Timestamp | None:
    """Latest date in the store, or ``None`` when there is no store yet."""
    path = _store_path()
    if not path.exists():
        return None
    dates = pd.read_parquet(path, columns=["date"])["date"]
    return None if dates.empty else pd.Timestamp(dates.max())


def load_panel(field: str = "close", tickers: Sequence[str] | None = None,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Wide panel (index=date, columns=ticker) of one field.

    Forward-filled up to ``MAX_FFILL`` rows, never backfilled. Requested tickers missing from the
    store come back as all-NaN columns; columns follow the requested order. ``start``/``end`` slice
    the index inclusively.
    """
    if field not in OHLCV:
        raise ValueError(f"unknown field {field!r}; expected one of {OHLCV}")
    symbols = None if tickers is None else list(dict.fromkeys(tickers))
    df = load_store()
    if symbols is not None:
        df = df[df["ticker"].isin(symbols)]
    wide = df.pivot(index="date", columns="ticker", values=field).sort_index()
    if symbols is not None:
        wide = wide.reindex(columns=symbols)
    return wide.ffill(limit=MAX_FFILL).loc[start:end]


# ------------------------------------------------------------------ latest prices
def latest_prices(tickers: Sequence[str]) -> tuple[dict[str, float], str]:
    """Latest price per ticker, preferring today's 1-minute bars.

    Returns ``(prices, source)``. ``source`` is ``"live_1m"`` when at least 80% of tickers have a
    live bar and the newest bar is under 20 minutes old (any missing ticker is filled from the store).
    Otherwise it is ``"last_close"`` and every ticker uses its stored close. A live-data outage never
    raises; ``FileNotFoundError`` is raised only when the store is actually needed and absent.
    """
    symbols = list(dict.fromkeys(tickers))
    if not symbols:
        return {}, "last_close"
    live, newest = _live_closes(symbols)
    fresh = newest is not None and pd.Timestamp.now(tz="UTC") - newest < LIVE_MAX_AGE
    if fresh and len(live) >= LIVE_MIN_COVERAGE * len(symbols):
        missing = [t for t in symbols if t not in live]
        if missing:
            log.warning("no live price for %s; using last stored close", missing)
            live.update(_last_stored_closes(missing))
        return live, "live_1m"
    log.info("live 1m prices unavailable or stale; using last stored closes")
    return _last_stored_closes(symbols), "last_close"


def _live_closes(symbols: list[str]) -> tuple[dict[str, float], pd.Timestamp | None]:
    """Last non-NaN 1-minute close per ticker from the current session, plus the newest bar (UTC)."""
    try:
        raw = yf.download(symbols, period="1d", interval="1m", progress=False, auto_adjust=True)
    except Exception as exc:  # live prices are best-effort; never break the caller
        log.warning("live price download failed (%s); falling back to stored closes", type(exc).__name__)
        return {}, None
    closes = pd.DataFrame({t: bars["close"] for t, bars in _split_by_ticker(raw, symbols).items()
                           if "close" in bars.columns})
    closes = closes.dropna(how="all")
    if closes.empty:
        return {}, None
    last = closes.ffill().iloc[-1].dropna()
    prices = {str(t): float(v) for t, v in last.items() if v > 0}
    newest = pd.Timestamp(closes.index.max())
    if newest.tzinfo is None:
        newest = newest.tz_localize(EXCHANGE_TZ)
    return prices, newest.tz_convert("UTC")


def _last_stored_closes(symbols: list[str]) -> dict[str, float]:
    """Most recent stored close per ticker; tickers absent from the store are omitted."""
    df = load_store()
    last = df[df["ticker"].isin(symbols)].sort_values("date").groupby("ticker")["close"].last()
    prices = {str(t): float(v) for t, v in last.items() if pd.notna(v)}
    absent = [t for t in symbols if t not in prices]
    if absent:
        log.warning("no stored close for %s", absent)
    return prices
