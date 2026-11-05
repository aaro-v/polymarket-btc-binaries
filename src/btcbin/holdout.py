"""Frozen holdout tests of the protocol (windows 2026-04-01 to 2026-10-31).

Every constant below was fixed on discovery data and committed in
protocols/polymarket_reversal.md before any holdout price, trade or outcome
was read. The holdout is run once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from btcbin import polymarket as pm

FROZEN = {
    "model": "p_rev",                 # reversal model, 2026 coefficients fitted on Binance 2022-2025
    "minutes_primary": (-4, -3, -2, -1, 0),
    "taker_theta": 0.02,              # minimum net edge per share, after the market's own fee schedule
    "quote_and_fill_window_s": 30,
    "maker_delta": 0.02,
    "alpha_one_sided": 0.05,
}


def _p(t: float) -> float:
    return float(1 - norm.cdf(t))


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def holdout_tables(M, H, X, trades, models) -> dict[str, pd.DataFrame]:
    tables, tests = {}, []
    D = H.dropna(subset=["mkt_next", "p_rev"])
    P = D[D.m.isin(FROZEN["minutes_primary"])]

    r = pm.info_regression(P, ["mkt_next", "p_rev"])
    tests.append({"test": "H1 (primary) model adds information to the market price, m -4..0",
                  "statistic": "logit coefficient on logit(p_rev)", "estimate": r["b_p_rev"], "t": r["t_p_rev"],
                  "n": r["n"], "discovery": "0.520 (t 4.41)"})
    r = pm.info_regression(D[D.m >= 1], ["mkt_next", "p_rev"])
    tests.append({"test": "H1b model adds information to the market price, m 1..14",
                  "statistic": "logit coefficient on logit(p_rev)", "estimate": r["b_p_rev"], "t": r["t_p_rev"],
                  "n": r["n"], "discovery": "0.186 (t 4.40)"})
    tk = pm.taker_trades(X, FROZEN["model"], FROZEN["taker_theta"])
    tests.append({"test": "H2 taker rule picks winners at traded prices (gross of fees)",
                  "statistic": "gross P&L per share", "estimate": tk.gross.mean(), "t": pm.day_t(tk.gross, tk.day),
                  "n": len(tk), "discovery": "+0.0211 (t 3.72); +0.0240 (t 2.84) with the current fee rule"})
    tests.append({"test": "H3 taker rule is profitable after the actual fee",
                  "statistic": "net P&L per share", "estimate": tk.pnl.mean(), "t": pm.day_t(tk.pnl, tk.day),
                  "n": len(tk), "discovery": "+0.0181 (t 3.20); +0.0065 (t 0.76) with the current fee rule"})
    delta = FROZEN["maker_delta"]
    mk_model, mk_market = pm.maker(M, trades, models, delta, "model"), pm.maker(M, trades, models, delta, "market")
    daily = pd.concat({"model": mk_model.groupby("day").pnl.mean(), "market": mk_market.groupby("day").pnl.mean()}, axis=1).dropna()
    diff = daily["model"] - daily["market"]
    tests.append({"test": "H4 quoting around the model beats quoting around the market",
                  "statistic": "difference in maker P&L per fill (daily paired)", "estimate": diff.mean(),
                  "t": float(diff.mean() / diff.std() * np.sqrt(len(diff))), "n": len(diff),
                  "discovery": "+0.0091 (t 1.71)"})
    T = pd.DataFrame(tests)
    T["p_one_sided"] = [_p(t) for t in T.t]
    T["p_holm"] = holm(T.p_one_sided.tolist())
    T["reject_at_5pct"] = T.p_holm < FROZEN["alpha_one_sided"]
    tables["frozen_tests"] = T.round(5)

    tables["taker"] = pd.DataFrame([
        pm.summarise(tk, rule="frozen: next taker buy within 30 s"),
        pm.summarise(_fill_at_last_ask(X), rule="robustness: fill at the last taker buy price before t"),
    ])
    tables["taker_by_month"] = tk.assign(month=tk.start.dt.to_period("M").astype(str)).groupby("month").apply(
        lambda g: pd.Series({"trades": len(g), "gross_c": g.gross.mean() * 100, "fee_c": g.fee.mean() * 100,
                             "net_c": g.pnl.mean() * 100})).round(3)
    tables["maker"] = pd.DataFrame([pm.summarise(mk_model, fair_value="model", delta=delta),
                                    pm.summarise(mk_market, fair_value="market", delta=delta)])
    w = H.drop_duplicates("start").dropna(subset=["up_binance"])
    tables["data_summary"] = pd.DataFrame([{
        "windows": len(M), "first": M.start.min(), "last": M.start.max(),
        "windows_with_pre_window_trades": len(trades),
        "binance_outcome_agrees_with_resolution": round((w.up_binance == w.up_official).mean(), 4),
        "fee_schedules": str(sorted(set(zip(M.fee_rate, M.fee_exp, strict=True))))}])
    rows = []
    for name, col in [("market, last print before t", "mkt_asof"), ("market, first print after t", "mkt_next"),
                      ("model, driftless", "p_base"), ("model, reversal", "p_rev")]:
        E = D.dropna(subset=[col])
        rows.append({"forecast": name, "log_loss": pm.log_loss(E[col], E.up_official).mean(),
                     "brier": ((E[col] - E.up_official) ** 2).mean()})
    tables["forecast_scores"] = pd.DataFrame(rows).round(5)
    tables["log_loss_by_minute"] = D.groupby("m").apply(lambda g: pd.Series(
        {c: pm.log_loss(g[c], g.up_official).mean() for c in ("mkt_next", "p_rev", "p_base")})).round(4)
    rows = []
    for name, sel in [("all minutes", D.m >= -4), ("m -4..0", D.m <= 0), ("m 1..5", D.m.between(1, 5)), ("m 6..14", D.m >= 6)]:
        for cols in (["mkt_next"], ["p_rev"], ["mkt_next", "p_rev"]):
            r = pm.info_regression(D[sel], cols)
            r.pop("params")
            rows.append({"minutes": name, "regressors": " + ".join(cols), **r})
    tables["information_regressions"] = pd.DataFrame(rows)
    return tables


def _fill_at_last_ask(X: pd.DataFrame) -> pd.DataFrame:
    d = X[X.p_rev.notna()].copy()
    e_up = d.p_rev - d.ask_up - pm.fee_per_share(d.ask_up, d.fee_rate, d.fee_exp)
    e_dn = (1 - d.p_rev) - d.ask_dn - pm.fee_per_share(d.ask_dn, d.fee_rate, d.fee_exp)
    d["side"] = np.where(e_up.fillna(-9) >= e_dn.fillna(-9), 1, -1)
    d["edge"] = np.where(d.side == 1, e_up, e_dn)
    d["fill"] = np.where(d.side == 1, d.ask_up, d.ask_dn)
    d = d[d.edge > FROZEN["taker_theta"]].sort_values(["start", "m"]).drop_duplicates("start")
    d["fee"] = pm.fee_per_share(d.fill, d.fee_rate, d.fee_exp)
    d["gross"] = np.where(d.side == 1, d.up_official, 1 - d.up_official) - d.fill
    d["pnl"] = d.gross - d.fee
    return d
