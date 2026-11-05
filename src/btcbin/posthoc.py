"""Robustness checks run after the holdout. Not part of the protocol.

They test how much each pre-registered conclusion depends on a design choice:
whether the passive-quote comparison (H4) gives both arms the same quoting
opportunities, whether the forecast comparison gives the market a later price
than the model, whether the information test (H1) is driven by noise in a single
price print, and how the taker result changes under a strict one-attempt reading
of "first signal".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcbin import polymarket as pm
from btcbin.holdout import FROZEN


def second_print(M: pd.DataFrame, ms=range(-4, 15), max_gap: int = 150) -> pd.DataFrame:
    """The price print after the first print following t (both within max_gap seconds of t)."""

    rows = []
    for r in M.itertuples():
        for m in ms:
            t = r.ts + 60 * m
            i = np.searchsorted(r.hist_t, t, side="right")
            p2 = r.hist_p[i + 1] if i + 1 < len(r.hist_t) and r.hist_t[i + 1] - t <= max_gap else np.nan
            rows.append((r.start, m, p2))
    return pd.DataFrame(rows, columns=["start", "m", "mkt_next2"])


def market_cells(M: pd.DataFrame, trades: dict) -> set:
    """(window, minute) cells where the market mid used by the passive-quote rule exists."""

    cells = set()
    for r in M.itertuples():
        T = trades.get(r.ts)
        if T is None:
            continue
        for m in pm.PRE:
            q = pm.quotes_and_fills(T, r.ts + 60 * m, window=60)
            if np.isfinite(q["ask_up"]) and np.isfinite(q["ask_dn"]):
                cells.add((r.start, m))
    return cells


def _paired(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, int]:
    daily = pd.concat({"a": a.groupby("day").pnl.mean(), "b": b.groupby("day").pnl.mean()}, axis=1).dropna()
    diff = daily.a - daily.b
    return float(diff.mean()), float(diff.mean() / diff.std() * np.sqrt(len(diff))), len(diff)


def strict_first_signal(X: pd.DataFrame, theta: float) -> pd.DataFrame:
    """One attempt per window: the first minute with a signal; if it does not fill, no trade."""

    d = X[X.p_rev.notna()].copy()
    e_up = d.p_rev - d.ask_up - pm.fee_per_share(d.ask_up, d.fee_rate, d.fee_exp)
    e_dn = (1 - d.p_rev) - d.ask_dn - pm.fee_per_share(d.ask_dn, d.fee_rate, d.fee_exp)
    d["side"] = np.where(e_up.fillna(-9) >= e_dn.fillna(-9), 1, -1)
    d["edge"] = np.where(d.side == 1, e_up, e_dn)
    d["fill"] = np.where(d.side == 1, d.fill_up, d.fill_dn)
    d = d[d.edge > theta].sort_values(["start", "m"]).drop_duplicates("start")
    d = d[d.fill.notna()].copy()
    d["fee"] = pm.fee_per_share(d.fill, d.fee_rate, d.fee_exp)
    d["gross"] = np.where(d.side == 1, d.up_official, 1 - d.up_official) - d.fill
    d["pnl"] = d.gross - d.fee
    return d


def fill_sizes(X_trades: pd.DataFrame, M: pd.DataFrame, trades: dict) -> float:
    ts = dict(zip(M.start, M.ts, strict=True))
    sizes = []
    for r in X_trades.itertuples():
        T, t = trades[ts[r.start]], ts[r.start] + 60 * r.m
        o = 0 if r.side == 1 else 1
        buys = T[(T[:, 1] == o) & (T[:, 2] == 1) & (T[:, 0] > t) & (T[:, 0] <= t + 30)]
        sizes.append(buys[0, 4])
    return float(np.median(sizes))


def same_second_fills(d: pd.DataFrame, M: pd.DataFrame, trades: dict, pick) -> pd.DataFrame:
    """Re-price each fill with ``pick`` (min or max) over the purchases in the same second as the first one."""

    ts = dict(zip(M.start, M.ts, strict=True))
    out = d.copy()
    prices = []
    for r in d.itertuples():
        T, t = trades[ts[r.start]], ts[r.start] + 60 * r.m
        o = 0 if r.side == 1 else 1
        buys = T[(T[:, 1] == o) & (T[:, 2] == 1) & (T[:, 0] > t) & (T[:, 0] <= t + 30)]
        prices.append(pick(buys[buys[:, 0] == buys[0, 0], 3]))
    out["fill"] = prices
    out["fee"] = pm.fee_per_share(out.fill, out.fee_rate, out.fee_exp)
    out["gross"] = np.where(out.side == 1, out.up_official, 1 - out.up_official) - out.fill
    out["pnl"] = out.gross - out.fee
    return out


def tables(samples: dict) -> dict[str, pd.DataFrame]:
    """samples maps 'discovery'/'holdout' to (M, models, trades, H, X)."""

    rows_maker, rows_ll, rows_info, rows_taker = [], [], [], []
    for name, (M, models, trades, H, X) in samples.items():
        delta = FROZEN["maker_delta"]
        mk_market = pm.maker(M, trades, models, delta, "market")
        mk_model = pm.maker(M, trades, models, delta, "model")
        cells = market_cells(M, trades)
        same = models[[c in cells for c in zip(models.start, models.m, strict=True)]]
        mk_model_same = pm.maker(M, trades, same, delta, "model")
        for label, arm in (("as pre-registered", mk_model), ("model restricted to the market arm's cells", mk_model_same)):
            diff, t, days = _paired(arm, mk_market)
            rows_maker.append({"sample": name, "comparison": label, "model_fills": len(arm),
                               "model_pnl_c": arm.pnl.mean() * 100, "market_fills": len(mk_market),
                               "market_pnl_c": mk_market.pnl.mean() * 100, "difference_c": diff * 100, "t": t, "days": days})

        D = H.dropna(subset=["mkt_asof", "mkt_next", "p_rev"])
        for label, sel in (("m -4..0", D.m <= 0), ("m 1..14", D.m >= 1)):
            E = D[sel]
            rows_ll.append({"sample": name, "minutes": label, "n": len(E),
                            **{c: pm.log_loss(E[c], E.up_official).mean() for c in ("mkt_asof", "p_rev", "mkt_next")}})

        P = H.dropna(subset=["mkt_next", "p_rev"]).merge(second_print(M), on=["start", "m"], how="left")
        P = P[P.m <= 0]
        r = pm.info_regression(P, ["mkt_next", "p_rev"])
        rows_info.append({"sample": name, "market price": "first print after t (as pre-registered)", "n": r["n"],
                          "b_market": r["b_mkt_next"], "b_model": r["b_p_rev"], "t_model": r["t_p_rev"]})
        Q = P.dropna(subset=["mkt_next2", "mkt_asof"]).assign(mkt_avg=lambda d: (d.mkt_next + d.mkt_next2) / 2,
                                                             mkt_mid=lambda d: (d.mkt_asof + d.mkt_next) / 2)
        for col, label in (("mkt_next", "first print after t, same rows"),
                           ("mkt_mid", "mean of the last print before t and the first after"),
                           ("mkt_avg", "mean of the two prints after t (adds a later price)")):
            r = pm.info_regression(Q, [col, "p_rev"])
            rows_info.append({"sample": name, "market price": label, "n": r["n"], "b_market": r[f"b_{col}"],
                              "b_model": r["b_p_rev"], "t_model": r["t_p_rev"]})

        theta = FROZEN["taker_theta"]
        frozen, strict = pm.taker_trades(X, "p_rev", theta), strict_first_signal(X, theta)
        for label, d in (("frozen rule (first filled signal)", frozen), ("one attempt per window", strict)):
            rows_taker.append({"sample": name, "rule": label, **pm.summarise(d), "median_fill_size": fill_sizes(d, M, trades)})
        for label, pick in (("frozen rule, cheapest purchase in the fill second", np.min),
                            ("frozen rule, dearest purchase in the fill second", np.max)):
            rows_taker.append({"sample": name, "rule": label, **pm.summarise(same_second_fills(frozen, M, trades, pick))})
        for reg in ("none", "legacy"):
            g = frozen[frozen.regime == reg]
            if len(g):
                rows_taker.append({"sample": name, "rule": f"frozen rule, fee regime {reg}", **pm.summarise(g)})
    return {"maker_same_cells": pd.DataFrame(rows_maker).round(4),
            "forecast_equal_information": pd.DataFrame(rows_ll).round(5),
            "information_price_noise": pd.DataFrame(rows_info).round(4),
            "taker_variants": pd.DataFrame(rows_taker)}
