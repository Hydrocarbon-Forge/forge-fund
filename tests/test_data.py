"""Tests for forge.data. yfinance is faked everywhere except the opt-in ``network`` smoke test."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from forge import data
from forge.config import get_paths
from forge.universe import BENCHMARK, CRYPTO, SP100, UNIVERSE

DAYS = pd.bdate_range("2026-08-03", periods=30).as_unit("s")  # tz-naive, like yfinance
REQUESTED = ["AAA", "BBB", "CCC", "DDD", "EEE"]
ALL_NAMES = [*REQUESTED, "NOPE"]  # NOPE never returns data
STORED_LAST = {"AAA": 114.5, "BBB": 64.5, "CCC": 84.5, "DDD": 44.5, "EEE": 54.5}  # close on DAYS[-1]
NOW = pd.Timestamp.now(tz="UTC").floor("min")


def _bars(base: float) -> pd.DataFrame:
    """Clean synthetic daily bars: close drifts up 0.5 per day from ``base``."""
    close = base + 0.5 * np.arange(len(DAYS))
    return pd.DataFrame({"open": close - 0.2, "high": close + 1.0, "low": close - 1.0,
                         "close": close, "volume": np.full(len(DAYS), 1_000_000.0)}, index=DAYS)


def _damaged_bbb() -> pd.DataFrame:
    bars = _bars(50.0)
    bars.loc[DAYS[10], "close"] = np.nan                            # NaN bar -> dropped
    bars.loc[DAYS[5], "high"] = bars.loc[DAYS[5], "close"] - 1.0    # high below close -> clipped up
    bars.loc[DAYS[7], "volume"] = np.nan                            # missing volume -> 0
    return bars


def _damaged_ccc() -> pd.DataFrame:
    bars = _bars(70.0)
    bars.loc[DAYS[12], "low"] = -1.0                                # non-positive price -> dropped
    return bars


def _raw(frames: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """yfinance-shaped download: (Price, Ticker) columns, NaN where a ticker has no data."""
    cols = {}
    for ticker, bars in frames.items():
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            src = pd.Series(np.nan, index=DAYS) if bars is None else bars[field.lower()]
            cols[(field, ticker)] = src.reindex(DAYS)
    raw = pd.DataFrame(cols, index=DAYS)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])
    return raw


def _live(quotes: dict[str, tuple[float, float | None]], last: pd.Timestamp) -> pd.DataFrame:
    """yfinance-shaped 1-minute frame (NY-tz index): two bars ending at ``last``; None = no print."""
    idx = pd.DatetimeIndex([last - pd.Timedelta(minutes=1), last]).tz_convert("America/New_York")
    cols = {("Close", t): pd.Series([prev, cur], index=idx, dtype="float64") for t, (prev, cur) in quotes.items()}
    frame = pd.DataFrame(cols, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Price", "Ticker"])
    return frame


class FakeYahoo:
    """Stands in for ``yf.download``. Daily calls serve ``frames``; 1m calls serve ``live``."""

    def __init__(self) -> None:
        self.frames: dict[str, pd.DataFrame | None] = {
            "AAA": _bars(100.0), "BBB": _damaged_bbb(), "CCC": _damaged_ccc(),
            "DDD": _bars(30.0), "EEE": _bars(40.0),  # NOPE is absent -> returned all-NaN
        }
        self.live: pd.DataFrame | Exception | None = None
        self.exc: Exception | None = None
        self.calls: list[dict] = []

    def __call__(self, tickers, **kwargs):
        self.calls.append({"tickers": list(tickers), **kwargs})
        if kwargs.get("interval") == "1m":
            if isinstance(self.live, Exception):
                raise self.live
            return self.live if self.live is not None else pd.DataFrame()
        if self.exc is not None:
            raise self.exc
        return _raw({t: self.frames.get(t) for t in tickers})


@pytest.fixture
def fake_yahoo(monkeypatch: pytest.MonkeyPatch) -> FakeYahoo:
    fake = FakeYahoo()
    monkeypatch.setattr(yf, "download", fake)
    return fake


@pytest.fixture
def stored(fake_yahoo: FakeYahoo) -> FakeYahoo:
    """Fake yfinance with a store already written for REQUESTED."""
    data.update_store(tickers=REQUESTED)
    return fake_yahoo


# ------------------------------------------------------------------ fetch / clean
def test_fetch_history_long_format_and_dtypes(fake_yahoo: FakeYahoo) -> None:
    df = data.fetch_history(ALL_NAMES)
    assert list(df.columns) == data.COLUMNS
    assert df["date"].dtype == "datetime64[ns]"
    assert pd.api.types.is_string_dtype(df["ticker"])
    assert all(df[c].dtype == "float64" for c in ["open", "high", "low", "close"])
    assert df["volume"].dtype == "int64"
    assert set(df["ticker"]) == set(REQUESTED)  # NOPE had no rows
    keys = list(zip(df["ticker"], df["date"]))
    assert keys == sorted(keys)


def test_cleaning_rules(fake_yahoo: FakeYahoo) -> None:
    df = data.fetch_history(ALL_NAMES)
    px = df[["open", "high", "low", "close"]]
    assert (px > 0).all().all()
    assert (df["high"] >= px[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= px[["open", "close"]].min(axis=1)).all()
    assert not df.duplicated(["ticker", "date"]).any()
    assert len(df) == 30 * 5 - 2  # BBB lost its NaN bar, CCC its non-positive bar
    bbb = df[df["ticker"] == "BBB"].set_index("date")
    assert pd.Timestamp(DAYS[10]) not in bbb.index
    assert bbb.loc[DAYS[5], "high"] == bbb.loc[DAYS[5], "close"]  # clipped up to close
    assert bbb.loc[DAYS[7], "volume"] == 0


def test_single_ticker_with_flat_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    flat = _bars(100.0)
    flat.columns = [c.title() for c in flat.columns]
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: flat)
    df = data.fetch_history(["AAA"])
    assert len(df) == len(DAYS)
    assert set(df["ticker"]) == {"AAA"}


# ------------------------------------------------------------------------ store
def test_update_store_writes_parquet_and_meta(fake_yahoo: FakeYahoo) -> None:
    meta = data.update_store(tickers=ALL_NAMES)
    root = get_paths().data
    assert (root / data.STORE_NAME).exists()
    assert not list(root.glob("*.tmp"))
    assert meta["dropped"] == ["NOPE"]
    assert meta["n_tickers"] == len(REQUESTED)
    assert meta["start"] == data.DEFAULT_START
    assert meta["last_date"] == DAYS[-1].strftime("%Y-%m-%d")
    assert json.loads((root / data.META_NAME).read_text()) == meta
    assert len(pd.read_parquet(root / data.STORE_NAME)) == 30 * 5 - 2
    assert fake_yahoo.calls[0]["auto_adjust"] is True


def test_update_store_defaults_to_universe_plus_benchmark(fake_yahoo: FakeYahoo) -> None:
    fake_yahoo.frames["SPY"] = _bars(400.0)
    meta = data.update_store()
    assert set(fake_yahoo.calls[0]["tickers"]) == {*UNIVERSE, BENCHMARK}
    assert BENCHMARK not in meta["dropped"]


def test_crypto_in_store_and_panel(fake_yahoo: FakeYahoo) -> None:
    fake_yahoo.frames["BTC-USD"] = _bars(50000.0)
    meta = data.update_store(tickers=["AAA", "BTC-USD", BENCHMARK])
    assert "BTC-USD" not in meta["dropped"]
    panel = data.load_panel("close", tickers=["AAA", "BTC-USD"])
    assert "BTC-USD" in panel.columns
    assert panel["BTC-USD"].iloc[-1] == pytest.approx(50000.0 + 0.5 * (len(DAYS) - 1))



@pytest.mark.parametrize("failure", ["empty", "raises"])
def test_total_failure_keeps_old_store(stored: FakeYahoo, failure: str) -> None:
    store = get_paths().data / data.STORE_NAME
    meta_path = get_paths().data / data.META_NAME
    before = (store.read_bytes(), meta_path.read_bytes())
    stored.frames = {}  # every ticker comes back all-NaN
    if failure == "raises":
        stored.exc = ConnectionError("network unreachable")
    with pytest.raises(RuntimeError):
        data.update_store(tickers=REQUESTED)
    assert (store.read_bytes(), meta_path.read_bytes()) == before


def test_last_store_date(fake_yahoo: FakeYahoo) -> None:
    assert data.last_store_date() is None
    data.update_store(tickers=REQUESTED)
    assert data.last_store_date() == DAYS[-1]


def test_load_store_without_store_raises(fake_yahoo: FakeYahoo) -> None:
    with pytest.raises(FileNotFoundError):
        data.load_store()


# ------------------------------------------------------------------------ panel
def test_load_panel_ffill_limit_and_no_backfill(fake_yahoo: FakeYahoo) -> None:
    fake_yahoo.frames = {
        "AAA": _bars(100.0),
        "BBB": _bars(50.0).iloc[[0, 5, 6, 7, 8, 9]],  # four missing trading days after day 0
        "CCC": _bars(70.0).iloc[3:],                  # no bars before day 3
    }
    data.update_store(tickers=["AAA", "BBB", "CCC"])
    wide = data.load_panel("close")
    assert list(wide.columns) == ["AAA", "BBB", "CCC"]
    bbb = wide["BBB"]
    assert bbb.iloc[1:4].tolist() == [50.0, 50.0, 50.0]  # limit=3 fills days 1-3
    assert math.isnan(bbb.iloc[4])                        # the fourth gap day stays NaN
    assert bbb.iloc[5] == 52.5
    assert wide["CCC"].iloc[:3].isna().all()              # never backfilled

    assert list(data.load_panel("close", tickers=["CCC", "AAA", "ZZZ"]).columns) == ["CCC", "AAA", "ZZZ"]
    sliced = data.load_panel("close", start=str(DAYS[2].date()), end=str(DAYS[4].date()))
    assert len(sliced) == 3


# ----------------------------------------------------------------- latest prices
def test_latest_prices_live_with_per_ticker_store_fallback(stored: FakeYahoo) -> None:
    # 4 of 5 tickers (80%) have a fresh live bar; EEE is filled from the store.
    stored.live = _live({"AAA": (101.0, 102.0), "BBB": (54.0, None), "CCC": (71.0, 72.0),
                         "DDD": (45.0, 46.0)}, NOW)
    prices, source = data.latest_prices(REQUESTED)
    assert source == "live_1m"
    assert prices == {"AAA": 102.0, "BBB": 54.0, "CCC": 72.0, "DDD": 46.0, "EEE": STORED_LAST["EEE"]}


def test_live_only_needs_no_store(fake_yahoo: FakeYahoo) -> None:
    fake_yahoo.live = _live({"AAA": (101.0, 102.0)}, NOW)
    assert data.latest_prices(["AAA"]) == ({"AAA": 102.0}, "live_1m")


@pytest.mark.parametrize("scenario", ["raises", "stale", "thin"])
def test_latest_prices_falls_back_to_last_close(stored: FakeYahoo, scenario: str) -> None:
    if scenario == "raises":
        stored.live = ConnectionError("network unreachable")
    elif scenario == "stale":
        stored.live = _live({t: (1.0, 2.0) for t in REQUESTED}, NOW - pd.Timedelta(hours=2))
    else:  # only 1 of 5 tickers has a live bar, below the 80% coverage bar
        stored.live = _live({"AAA": (101.0, 102.0)}, NOW)
    prices, source = data.latest_prices(REQUESTED)
    assert source == "last_close"
    assert prices == STORED_LAST


def test_latest_prices_raises_only_when_store_needed(fake_yahoo: FakeYahoo) -> None:
    fake_yahoo.live = ConnectionError("network unreachable")
    with pytest.raises(FileNotFoundError):
        data.latest_prices(["AAA"])


# ---------------------------------------------------------------- network smoke
@pytest.mark.network
def test_network_smoke_aapl_and_spy() -> None:
    start = (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    df = data.fetch_history(["AAPL", "SPY"], start=start)
    counts = df["ticker"].value_counts()
    assert counts.get("AAPL", 0) >= 5
    assert counts.get("SPY", 0) >= 5
