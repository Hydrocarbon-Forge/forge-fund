import math

import pytest

from forge.config import LIMITS, Limits
from forge.risk import check_limits, clamp_weights, loss_cap_breached, max_investable, order_sanity


def test_max_investable_is_gross_cap():
    assert max_investable() == pytest.approx(0.80)


def test_clamp_caps_gross_and_single():
    w = clamp_weights({"A": 0.5, "B": 0.4, "C": 0.3, "D": 0.2, "E": 0.1})
    assert sum(w.values()) == pytest.approx(0.80, abs=1e-5)
    assert max(w.values()) <= LIMITS.max_single + 1e-9
    assert check_limits(w) == []


def test_clamp_top_n_and_drops_invalid():
    raw = {f"T{i}": 0.05 + i * 0.001 for i in range(15)}
    raw.update({"NEG": -0.2, "NAN": float("nan"), "INF": float("inf"), "ZERO": 0.0})
    w = clamp_weights(raw)
    assert len(w) == LIMITS.max_positions
    assert not {"NEG", "NAN", "INF", "ZERO"} & set(w)
    assert check_limits(w) == []


def test_clamp_never_scales_up():
    w = clamp_weights({"A": 0.10, "B": 0.10})
    assert w == {"A": 0.10, "B": 0.10}


def test_clamp_redistribution_respects_cap():
    # one dominant name: capped at 25%, remainder redistributed to the rest
    w = clamp_weights({"BIG": 0.9, "S1": 0.05, "S2": 0.05})
    assert w["BIG"] == pytest.approx(0.25)
    # only 3 names, so cannot reach 0.80 gross without breaching the single cap: stays <= 0.75
    assert sum(w.values()) <= 0.75 + 1e-6
    assert check_limits(w) == []


def test_clamp_min_notional():
    w = clamp_weights({"A": 0.20, "B": 0.003}, equity=100_000)   # B = $300 < $500
    assert set(w) == {"A"}


def test_clamp_empty():
    assert clamp_weights({}) == {}
    assert clamp_weights({"A": -1}) == {}


def test_check_limits_flags_each_violation():
    v = check_limits({"A": 0.3, "B": 0.6})
    assert any("weight" in x for x in v) and any("gross" in x for x in v)
    assert any("cash" in x for x in check_limits({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.20}))


def test_custom_limits():
    lim = Limits(max_positions=2, max_single=0.4, max_gross=0.6)
    w = clamp_weights({"A": 0.5, "B": 0.5, "C": 0.5}, lim)
    assert len(w) == 2 and sum(w.values()) == pytest.approx(0.6)


@pytest.mark.parametrize("kw,ok", [
    (dict(qty=10, price=100.0), True),
    (dict(qty=0, price=100.0), False),
    (dict(qty=1.5, price=100.0), False),
    (dict(qty=10, price=float("nan")), False),
    (dict(qty=10, price=-3.0), False),
    (dict(qty=400, price=100.0), False),              # $40k > 30% of $100k
    (dict(qty=2, price=100.0), False),                # $200 BUY < min notional
    (dict(qty=10, price=140.0, ref_close=100.0), False),   # 40% off last close
])
def test_order_sanity(kw, ok):
    r = order_sanity("AAPL", "BUY", kw["qty"], kw["price"], 100_000, kw.get("ref_close"))
    assert r.ok is ok, r.reason


def test_order_sanity_small_sell_allowed_and_bad_side():
    assert order_sanity("AAPL", "SELL", 2, 100.0, 100_000).ok          # closing a dust position is fine
    assert not order_sanity("AAPL", "SHORT", 2, 100.0, 100_000).ok


def test_loss_cap():
    assert loss_cap_breached(100_000, 97_900) == (True, pytest.approx(-0.021))
    assert loss_cap_breached(100_000, 98_100)[0] is False
    assert loss_cap_breached(None, 50_000) == (False, 0.0)
    assert math.isclose(loss_cap_breached(100_000, 101_000)[1], 0.01)
