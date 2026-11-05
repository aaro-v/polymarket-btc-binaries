import numpy as np
import pandas as pd

from btcbin.polymarket import maker, print_before_after, quotes_and_fills


def test_prints_before_and_after_decision_time():
    t = np.array([100, 160, 230])
    p = np.array([0.4, 0.5, 0.6])
    asof, nxt, delay = print_before_after(t, p, 160)
    assert asof == 0.5 and nxt == 0.6 and delay == 70          # a print at exactly t counts as already known
    asof, nxt, delay = print_before_after(t, p, 400)
    assert np.isnan(asof) and np.isnan(nxt)                     # stale beyond 90 seconds


def test_quotes_strictly_before_and_fills_strictly_after():
    # columns: time, outcome (0 up / 1 down), side (+1 buy / -1 sell), price, size
    T = np.array([[90, 0, 1, 0.51, 5], [100, 0, 1, 0.52, 5], [101, 0, 1, 0.55, 5], [105, 1, 1, 0.47, 5], [140, 0, 1, 0.60, 5]])
    q = quotes_and_fills(T, 100)
    assert q["ask_up"] == 0.52 and q["fill_up"] == 0.55
    assert np.isnan(q["ask_dn"]) and q["fill_dn"] == 0.47


def test_maker_fill_requires_trading_through_the_bid():
    start = pd.Timestamp("2026-01-01")
    ts = int(start.timestamp())
    M = pd.DataFrame({"ts": [ts], "start": [start], "up_official": [1.0]})
    models = pd.DataFrame({"start": [start] * 5, "m": range(-4, 1), "p_rev": [0.55] * 5})
    t0 = ts - 240
    at_bid = np.array([[t0 + 5, 0, -1, 0.53, 10]])            # sale exactly at our 0.53 bid: no fill assumed
    assert maker(M, {ts: at_bid}, models, delta=0.02).empty
    through = np.array([[t0 + 5, 0, -1, 0.52, 10]])           # sale below our bid: we would have been hit at 0.53
    d = maker(M, {ts: through}, models, delta=0.02)
    assert len(d) == 1 and d.side.iloc[0] == 1 and np.isclose(d.price.iloc[0], 0.53) and np.isclose(d.pnl.iloc[0], 0.47)
