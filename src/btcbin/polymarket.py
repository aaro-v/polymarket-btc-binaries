"""Part 3: Polymarket's 15-minute bitcoin markets against the Binance-only models.

Decision minutes m run from -4 (four minutes before the window opens) to 14.
The model at minute m uses Binance bars that closed by t = start + m minutes.
Market prices come from two sources: the CLOB one-minute price history of the
Up token, and individual taker trades in the five minutes before the window
and its first 90 seconds. Outcomes are the official resolutions (Chainlink
BTC/USD), not Binance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

SAMPLES = {"discovery": ("2025-10-10", "2026-03-31 23:59"), "holdout": ("2026-04-01", "2026-10-31 23:59")}
FEE_REGIMES = (("none", None, "2026-01-07"), ("legacy", "2026-01-07", "2026-03-30"), ("current", "2026-03-30", None))
PRE = range(-4, 1)
TICK = 0.01


def fee_per_share(p, rate, exponent):
    """Taker fee in USDC per share at price p.

    Exponent 1 (from 2026-03-30): rate * p * (1 - p), as in Polymarket's fee table
    (crypto rate 0.07: 1.75 USDC per 100 shares at 50 cents). Exponent 2 (the
    January 2026 schedule, rate 0.25): p * rate * (p * (1 - p))**2, 0.78 USDC per
    100 shares at 50 cents, a peak effective rate of 1.56%.
    """

    p = np.asarray(p, dtype=float)
    q = p * (1 - p)
    return np.where(np.asarray(exponent) >= 2, p * np.asarray(rate) * q**2, np.asarray(rate) * q)


def regime(start: pd.Series) -> np.ndarray:
    return np.select([start < "2026-01-07", start < "2026-03-30"], ["none", "legacy"], "current")


def logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def log_loss(p, y, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_markets(path: Path, sample: str) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for r in map(json.loads, f):
            if r.get("missing") or not r.get("closed"):
                continue
            prices, outcomes = json.loads(r["outcomePrices"]), json.loads(r["outcomes"])
            if outcomes[0] != "Up" or set(prices) != {"0", "1"}:
                continue
            fs = (r.get("feeSchedule") or {}) if r.get("feesEnabled") else {}
            hist = r.get("history") or []
            rows.append({"ts": r["ts"], "up_official": float(prices[0] == "1"), "volume": r.get("volume"),
                         "fee_rate": float(fs.get("rate", 0.0)), "fee_exp": float(fs.get("exponent", 1.0)),
                         "hist_t": np.array([h["t"] for h in hist], dtype=np.int64),
                         "hist_p": np.array([h["p"] for h in hist], dtype=float)})
    M = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    M["start"] = pd.to_datetime(M["ts"], unit="s")
    lo, hi = SAMPLES[sample]
    return M[(M.start >= lo) & (M.start <= hi)].reset_index(drop=True)


def load_trades(path: Path, keep: set[int]) -> dict[int, np.ndarray]:
    """Taker trades per window as an array of (time, outcome index, +1 buy / -1 sell, price, size), time-sorted."""

    out = {}
    with path.open() as f:
        for r in map(json.loads, f):
            if r.get("error") or r["ts"] not in keep or not r.get("trades"):
                continue
            T = np.array(r["trades"], dtype=float).reshape(-1, 5)
            out[r["ts"]] = T[np.argsort(T[:, 0], kind="stable")]
    return out


def print_before_after(hist_t: np.ndarray, hist_p: np.ndarray, t: int, max_gap: int = 90) -> tuple[float, float, float]:
    """Last price-history point at or before t, first point strictly after t, and that point's delay (seconds)."""

    i = np.searchsorted(hist_t, t, side="right")
    asof = hist_p[i - 1] if i > 0 and t - hist_t[i - 1] <= max_gap else np.nan
    if i < len(hist_t) and hist_t[i] - t <= max_gap:
        return asof, hist_p[i], float(hist_t[i] - t)
    return asof, np.nan, np.nan


def history_panel(M: pd.DataFrame, models: pd.DataFrame, ms=range(-4, 15)) -> pd.DataFrame:
    rows = []
    for r in M.itertuples():
        for m in ms:
            rows.append((r.start, m, *print_before_after(r.hist_t, r.hist_p, r.ts + 60 * m)))
    X = pd.DataFrame(rows, columns=["start", "m", "mkt_asof", "mkt_next", "next_delay_s"])
    X = X.merge(models, on=["start", "m"], how="left")
    X = X.merge(M[["start", "up_official", "fee_rate", "fee_exp"]], on="start")
    X["day"], X["regime"] = X.start.dt.floor("D"), regime(X.start)
    return X


def quotes_and_fills(T: np.ndarray, t: int, window: int = 30) -> dict:
    """Last taker buy of each outcome in (t - window, t] (a proxy for its ask) and the first taker buy in (t, t + window]."""

    out = {}
    for outcome, name in ((0, "up"), (1, "dn")):
        buys = T[(T[:, 1] == outcome) & (T[:, 2] == 1)]
        before = buys[(buys[:, 0] <= t) & (buys[:, 0] > t - window)]
        after = buys[(buys[:, 0] > t) & (buys[:, 0] <= t + window)]
        out[f"ask_{name}"] = before[-1, 3] if len(before) else np.nan
        out[f"fill_{name}"] = after[0, 3] if len(after) else np.nan
    return out


def trades_panel(M: pd.DataFrame, trades: dict[int, np.ndarray], models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in M.itertuples():
        T = trades.get(r.ts)
        if T is None:
            continue
        for m in PRE:
            rows.append({"start": r.start, "m": m, **quotes_and_fills(T, r.ts + 60 * m)})
    X = pd.DataFrame(rows).merge(models, on=["start", "m"], how="left")
    X = X.merge(M[["start", "up_official", "fee_rate", "fee_exp"]], on="start")
    X["mid"] = (X.ask_up + 1 - X.ask_dn) / 2
    X["day"], X["regime"] = X.start.dt.floor("D"), regime(X.start)
    return X


def info_regression(X: pd.DataFrame, cols: list[str], y: str = "up_official") -> dict:
    """Logit of the outcome on logit-transformed forecasts; standard errors clustered by UTC day."""

    Z = sm.add_constant(pd.DataFrame({c: logit(X[c]) for c in cols}))
    fit = sm.Logit(X[y], Z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": X["day"].factorize()[0]})
    out = {"n": len(X), "pseudo_r2": round(fit.prsquared, 4)}
    for k in fit.params.index:
        out[f"b_{k}"], out[f"t_{k}"] = round(fit.params[k], 4), round(fit.tvalues[k], 2)
    out["params"] = fit.params
    return out


def day_t(values: pd.Series, days: pd.Series, agg: str = "sum") -> float:
    """t-statistic of the mean of daily totals (agg='sum') or daily averages (agg='mean')."""

    daily = values.groupby(days).agg(agg)
    return float(daily.mean() / daily.std() * np.sqrt(len(daily)))


def taker_history(X: pd.DataFrame, prob: str = "p_rev", theta: float = 0.02, half_spread: float = 0.01,
                  ms=range(-4, 15), price: str = "mkt_next") -> pd.DataFrame:
    """Buy the side whose model value exceeds price + half-spread + fee by theta; first signal per window."""

    d = X[X.m.isin(list(ms)) & X[price].between(0.02, 0.98) & X[prob].notna()].copy()
    a_up, a_dn = d[price] + half_spread, 1 - d[price] + half_spread
    e_up = d[prob] - a_up - fee_per_share(a_up, d.fee_rate, d.fee_exp)
    e_dn = (1 - d[prob]) - a_dn - fee_per_share(a_dn, d.fee_rate, d.fee_exp)
    d["side"] = np.where(e_up >= e_dn, 1, -1)
    d["edge"] = np.maximum(e_up, e_dn)
    d["cost"] = np.where(d.side == 1, a_up, a_dn)
    d = d[d.edge > theta].sort_values(["start", "m"]).drop_duplicates("start")
    d["fee"] = fee_per_share(d.cost, d.fee_rate, d.fee_exp)
    d["gross"] = np.where(d.side == 1, d.up_official, 1 - d.up_official) - d.cost
    d["pnl"] = d.gross - d.fee
    return d


def taker_trades(X: pd.DataFrame, prob: str = "p_rev", theta: float = 0.02) -> pd.DataFrame:
    """Causal taker test: the decision uses the last taker buy prices before t, the fill is the next taker buy after t."""

    d = X[X[prob].notna()].copy()
    e_up = d[prob] - d.ask_up - fee_per_share(d.ask_up, d.fee_rate, d.fee_exp)
    e_dn = (1 - d[prob]) - d.ask_dn - fee_per_share(d.ask_dn, d.fee_rate, d.fee_exp)
    d["side"] = np.where(e_up.fillna(-9) >= e_dn.fillna(-9), 1, -1)
    d["edge"] = np.where(d.side == 1, e_up, e_dn)
    d["fill"] = np.where(d.side == 1, d.fill_up, d.fill_dn)
    d = d[(d.edge > theta) & d.fill.notna()].sort_values(["start", "m"]).drop_duplicates("start")
    d["fee"] = fee_per_share(d.fill, d.fee_rate, d.fee_exp)
    d["gross"] = np.where(d.side == 1, d.up_official, 1 - d.up_official) - d.fill
    d["pnl"] = d.gross - d.fee
    return d


def maker(M: pd.DataFrame, trades: dict[int, np.ndarray], models: pd.DataFrame, delta: float = 0.02,
          fair: str = "model", window: int = 60) -> pd.DataFrame:
    """Passive bids for Up at f - delta and for Down at 1 - f - delta, posted at t and live during (t, t + 60].

    A bid fills only if a taker trades strictly through it: a taker sale of that
    outcome below the bid, or a taker purchase of the other outcome above one
    minus the bid. Positions are held to resolution, one fill per window. Makers
    pay no fee; rebates are ignored.
    """

    p = models.set_index(["start", "m"])["p_rev"]
    out = []
    for r in M.itertuples():
        T = trades.get(r.ts)
        if T is None:
            continue
        for m in PRE:
            t = r.ts + 60 * m
            if fair == "model":
                f = p.get((r.start, m), np.nan)
            else:
                q = quotes_and_fills(T, t, window=60)
                f = (q["ask_up"] + 1 - q["ask_dn"]) / 2
            if not np.isfinite(f):
                continue
            b_up = np.floor(np.round((f - delta) / TICK, 6)) * TICK
            b_dn = np.floor(np.round((1 - f - delta) / TICK, 6)) * TICK
            W = T[(T[:, 0] > t) & (T[:, 0] <= t + window)]
            o, s, px = W[:, 1], W[:, 2], W[:, 3]
            hit_up = W[((o == 0) & (s == -1) & (px < b_up - 1e-9)) | ((o == 1) & (s == 1) & (px > 1 - b_up + 1e-9))]
            hit_dn = W[((o == 1) & (s == -1) & (px < b_dn - 1e-9)) | ((o == 0) & (s == 1) & (px > 1 - b_dn + 1e-9))]
            t_up = hit_up[0, 0] if len(hit_up) else np.inf
            t_dn = hit_dn[0, 0] if len(hit_dn) else np.inf
            if np.isinf(t_up) and np.isinf(t_dn):
                continue
            side, price = (1, b_up) if t_up <= t_dn else (-1, b_dn)
            payoff = r.up_official if side == 1 else 1 - r.up_official
            out.append({"start": r.start, "m": m, "side": side, "price": price, "fair": f, "pnl": payoff - price})
            break
    d = pd.DataFrame(out)
    if len(d):
        d["day"], d["regime"] = d.start.dt.floor("D"), regime(d.start)
    return d


def summarise(d: pd.DataFrame, **labels) -> dict:
    if len(d) < 5:
        return {**labels, "trades": len(d)}
    row = {**labels, "trades": len(d), "trades_per_day": round(len(d) / d.day.nunique(), 1)}
    if "edge" in d:
        row["predicted_edge_c"] = round(d.edge.mean() * 100, 2)
    if "gross" in d:
        row["gross_c"] = round(d.gross.mean() * 100, 2)
        row["t_gross"] = round(day_t(d.gross, d.day), 2)
        row["fee_c"] = round(d.fee.mean() * 100, 2)
    row["net_c"] = round(d.pnl.mean() * 100, 2)
    row["t_net"] = round(day_t(d.pnl, d.day), 2)
    return row


def _models(work_dir: Path, M: pd.DataFrame) -> pd.DataFrame:
    R = pd.read_parquet(work_dir / "reversal_panel.parquet", columns=["start", "m", "up", "p_base", "p_rev"])
    return R[R.start.isin(M.start)].rename(columns={"up": "up_binance"})


def discovery_tables(M: pd.DataFrame, H: pd.DataFrame, X: pd.DataFrame, trades: dict, models: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    w = H.drop_duplicates("start").dropna(subset=["up_binance"])
    tables["data_summary"] = pd.DataFrame([{
        "windows": len(M), "first": M.start.min(), "last": M.start.max(),
        "windows_with_pre_window_trades": len(trades),
        "binance_outcome_agrees_with_resolution": round((w.up_binance == w.up_official).mean(), 4),
        "median_delay_of_next_print_s": H.next_delay_s.median()}])
    D = H.dropna(subset=["mkt_asof", "mkt_next", "p_rev"])           # forecast scores: identical rows for every forecast
    rows = []
    for name, col in [("market, last print before t", "mkt_asof"), ("market, first print after t", "mkt_next"),
                      ("model, driftless", "p_base"), ("model, reversal", "p_rev")]:
        rows.append({"forecast": name, "log_loss": log_loss(D[col], D.up_official).mean(),
                     "brier": ((D[col] - D.up_official) ** 2).mean()})
    tables["forecast_scores"] = pd.DataFrame(rows).round(5)
    tables["log_loss_by_minute"] = D.groupby("m").apply(lambda g: pd.Series(
        {c: log_loss(g[c], g.up_official).mean() for c in ("mkt_next", "p_rev", "p_base")})).round(4)
    D = H.dropna(subset=["mkt_next", "p_rev"])                       # information tests: the holdout's row filter
    rows = []
    subsets = [("all minutes", D.m >= -4), ("m -4..0", D.m <= 0), ("m 1..14", D.m >= 1), ("m 1..5", D.m.between(1, 5)),
               ("m 6..14", D.m >= 6)]
    for reg in ("none", "legacy"):
        for name, sel in subsets:
            E = D[sel & (D.regime == reg)]
            r = info_regression(E, ["mkt_next", "p_rev"])
            r.pop("params")
            rows.append({"fee_regime": reg, "minutes": name, **r})
    for name, sel in subsets:
        for cols in (["mkt_next"], ["p_rev"], ["mkt_next", "p_rev"], ["mkt_next", "p_base"]):
            r = info_regression(D[sel], cols)
            r.pop("params")
            rows.append({"fee_regime": "all", "minutes": name, "regressors": " + ".join(cols), **r})
    tables["information_regressions"] = pd.DataFrame(rows)
    bins = [0, .03, .07, .15, .3, .45, .55, .7, .85, .93, .97, 1]
    C = D.assign(bin=pd.cut(D.mkt_next, bins, include_lowest=True), minutes=pd.cut(D.m, [-5, 0, 5, 10, 14]))
    tables["market_calibration"] = C.groupby(["minutes", "bin"], observed=True).agg(
        price=("mkt_next", "mean"), realised=("up_official", "mean"), model=("p_rev", "mean"), n=("up_official", "size")).round(4)
    rows = []
    for reg in ("none", "legacy"):
        S = H[H.regime == reg]
        for ms, label in ((range(-4, 1), "-4..0"), (range(1, 6), "1..5"), (range(6, 11), "6..10"), (range(11, 15), "11..14")):
            for prob in ("p_rev", "p_base"):
                for theta in (0.0, 0.01, 0.02):
                    rows.append(summarise(taker_history(S, prob, theta, ms=ms), fee_regime=reg, minutes=label, model=prob, theta=theta))
        for hs in (0.005, 0.02):
            rows.append(summarise(taker_history(S, "p_rev", 0.02, half_spread=hs, ms=PRE), fee_regime=reg, minutes="-4..0",
                                  model=f"p_rev, half-spread {hs}", theta=0.02))
    tables["taker_price_history"] = pd.DataFrame(rows)
    rows = []
    for reg in ("none", "legacy"):
        S = X[X.regime == reg]
        for prob in ("p_rev", "p_base"):
            for theta in (0.0, 0.01, 0.02, 0.03):
                rows.append(summarise(taker_trades(S, prob, theta), fee_regime=reg, model=prob, theta=theta))
    Xc = X.assign(fee_rate=0.07, fee_exp=1.0)          # the schedule in force from 2026-03-30, applied throughout
    for theta in (0.0, 0.01, 0.02, 0.03, 0.04):
        rows.append(summarise(taker_trades(Xc, "p_rev", theta), fee_regime="all, current fee rule applied", model="p_rev", theta=theta))
    rows.append(summarise(taker_trades(X, "p_rev", 0.02), fee_regime="all, actual fees", model="p_rev", theta=0.02))
    rng = np.random.default_rng(0)
    placebo = X.assign(p_rev=np.where(rng.random(len(X)) < 0.5, 0.99, 0.01))   # random side, always 'traded'
    for reg in ("none", "legacy"):
        rows.append(summarise(taker_trades(placebo[placebo.regime == reg], "p_rev", -1.0), fee_regime=reg,
                              model="placebo: random side, same fill rule", theta=None))
    tables["taker_trades"] = pd.DataFrame(rows)
    d = taker_trades(X, "p_rev", 0.02)
    tables["taker_by_minute"] = d.groupby("m").apply(lambda g: pd.Series(
        {"trades": len(g), "gross_c": g.gross.mean() * 100, "net_c": g.pnl.mean() * 100})).round(3)
    tables["taker_by_month"] = d.assign(month=d.start.dt.to_period("M").astype(str)).groupby("month").apply(
        lambda g: pd.Series({"trades": len(g), "gross_c": g.gross.mean() * 100, "fee_c": g.fee.mean() * 100,
                             "net_c": g.pnl.mean() * 100})).round(3)
    E = X.dropna(subset=["mid", "p_rev"])
    rows = []
    for cols in (["mid"], ["mid", "p_rev"]):
        r = info_regression(E, cols)
        r.pop("params")
        rows.append({"regressors": " + ".join(cols), **r})
    tables["information_trade_quotes"] = pd.DataFrame(rows)
    rows = []
    for fair in ("model", "market"):
        for delta in (0.01, 0.02, 0.03, 0.05):
            d = maker(M, trades, models, delta, fair)
            rows.append(summarise(d, fair_value=fair, delta=delta, fee_regime="all"))
            for reg, g in d.groupby("regime"):
                rows.append(summarise(g, fair_value=fair, delta=delta, fee_regime=reg))
    tables["maker"] = pd.DataFrame(rows)
    return tables


def load_sample(data_dir: Path, work_dir: Path, sample: str):
    M = load_markets(data_dir / "polymarket_markets.jsonl", sample)
    models = _models(work_dir, M)
    trades = load_trades(data_dir / "polymarket_trades.jsonl", set(M.ts))
    return M, models, trades, history_panel(M, models), trades_panel(M, trades, models)


def run(data_dir: Path, work_dir: Path, out_dir: Path, sample: str = "discovery") -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if sample == "posthoc":
        from btcbin import posthoc

        tables = posthoc.tables({s: load_sample(data_dir, work_dir, s) for s in ("discovery", "holdout")})
        for name, table in tables.items():
            table.to_csv(out_dir / f"{name}.csv", index=False)
        return tables
    M, models, trades, H, X = load_sample(data_dir, work_dir, sample)
    if sample == "discovery":
        tables = discovery_tables(M, H, X, trades, models)
    else:
        from btcbin.holdout import holdout_tables
        tables = holdout_tables(M, H, X, trades, models)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=isinstance(table.index, pd.MultiIndex) or table.index.name is not None)
    return tables
