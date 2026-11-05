"""Part 2: probability that a 15-minute window closes up, at each minute of the window.

At minute m of a window that opened at price K, with current price S and a
variance forecast F for the next 15 minutes, the driftless Gaussian price is
Phi(log(S/K) / sqrt(F (15-m)/15)). The empirical version replaces the normal
with the distribution of standardised remaining returns at the same minute in
earlier years. The reversal model adds past returns over 1 to 240 minutes,
scaled by the volatility forecast, in a logistic regression refitted every
year on earlier years only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

ONE = pd.Timedelta("1min")
LAGS = (1, 5, 15, 60, 240)
BEST = "har_seasonal_iv"


def clip(p, eps=1e-4):
    return np.clip(p, eps, 1 - eps)


def logit(p, eps=1e-4):
    p = clip(p, eps)
    return np.log(p / (1 - p))


def log_loss(p, y, eps=1e-4):
    p = clip(p, eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def price_at(px: pd.Series, t: pd.DatetimeIndex) -> np.ndarray:
    """Price at time t: the close of the one-minute bar stamped t-1m."""

    return px.reindex(t - ONE).to_numpy()


def window_panel(px: pd.Series, F: pd.DataFrame, start: str = "2020-01-08", end: str = "2026-10-31 23:30") -> pd.DataFrame:
    """One row per (window, minute m = 1..14): moneyness, outcome and every model's probability."""

    starts = pd.date_range(start, end, freq="15min")
    K, E = price_at(px, starts), price_at(px, starts + 15 * ONE)
    rows = []
    for m in range(1, 15):
        t = starts + m * ONE
        f = F.reindex(t - ONE)
        rows.append(pd.DataFrame({"start": starts, "m": m, "x": np.log(price_at(px, t) / K),
                                  "z_end": np.log(E / price_at(px, t)), "up": (E >= K).astype(float),
                                  **{f"F_{c}": f[c].to_numpy() for c in F.columns}}))
    D = pd.concat(rows, ignore_index=True).dropna()
    D["year"] = D["start"].dt.year
    for c in F.columns:
        D[f"sd_{c}"] = np.sqrt(D[f"F_{c}"] * (15 - D["m"]) / 15)
        D[f"g_{c}"] = norm.cdf(D["x"] / D[f"sd_{c}"])
    D[f"e_{BEST}"] = empirical_probability(D, f"sd_{BEST}")
    return D


def empirical_probability(D: pd.DataFrame, sd_col: str, years=range(2021, 2027), min_obs: int = 1000) -> np.ndarray:
    """P(up) from the empirical distribution of z_end / sd at the same minute in all earlier years."""

    p = np.full(len(D), np.nan)
    for year in years:
        for m in range(1, 15):
            train = D[(D.year < year) & (D.m == m)]
            z = (train["z_end"] / train[sd_col]).to_numpy()
            z = np.sort(z[np.isfinite(z)])
            if len(z) < min_obs:
                continue
            test = ((D.year == year) & (D.m == m)).to_numpy()
            q = (-D.loc[test, "x"] / D.loc[test, sd_col]).to_numpy()
            p[test] = 1 - np.searchsorted(z, q, side="left") / len(z)
    return p


def recalibrate(D: pd.DataFrame, col: str, years=range(2024, 2027), max_obs: int = 1_500_000) -> pd.Series:
    """Logistic recalibration of a probability (slope varying with the minute), fitted on earlier years."""

    out = pd.Series(np.nan, index=D.index)
    design = lambda d: sm.add_constant(pd.DataFrame({"lp": logit(d[col]), "lp_m": logit(d[col]) * d["m"] / 15}), has_constant="add")
    for year in years:
        train = D[(D.year < year) & D[col].notna()]
        train = train.sample(min(len(train), max_obs), random_state=0)
        fit = sm.Logit(train["up"], design(train)).fit(disp=0)
        test = D.year == year
        out[test] = fit.predict(design(D[test]))
    return out


def reversal_panel(px: pd.Series, F: pd.DataFrame, D: pd.DataFrame, start: str = "2022-01-03",
                   end: str = "2026-10-31 23:30", years=range(2024, 2027)) -> tuple[pd.DataFrame, dict]:
    """Minutes m = -4..14 around each window start; driftless and reversal probabilities fitted on earlier years.

    For m <= 0 the opening price is not yet known and the driftless price is 1/2.
    Past returns are scaled by the volatility forecast and by sqrt(tau), the
    square root of the fraction of the window still to run.
    """

    lp = np.log(px)
    f15 = F[BEST]
    starts = pd.date_range(start, end, freq="15min")
    up = (price_at(lp, starts + 15 * ONE) >= price_at(lp, starts)).astype(float)
    rows = []
    for m in range(-4, 15):
        t = starts + m * ONE
        tau = 1.0 if m <= 0 else (15 - m) / 15
        f = f15.reindex(t - ONE).to_numpy()
        d = pd.DataFrame({"start": starts, "m": m, "up": up, "tau": tau})
        for k in LAGS:
            d[f"z{k}"] = (price_at(lp, t) - price_at(lp, t - k * ONE)) / np.sqrt(f * k / 15) * np.sqrt(tau)
        rows.append(d)
    R = pd.concat(rows, ignore_index=True).merge(D[["start", "m", f"e_{BEST}"]], on=["start", "m"], how="left")
    R.loc[R.m <= 0, f"e_{BEST}"] = 0.5
    R = R.dropna()
    R["year"] = R.start.dt.year
    R["lb"] = logit(R[f"e_{BEST}"])
    R["lb_tau"] = R.lb * R.tau
    base, rev = ["lb", "lb_tau"], ["lb", "lb_tau"] + [f"z{k}" for k in LAGS]
    R["p_base"], R["p_rev"], coefs = np.nan, np.nan, {}
    for year in years:
        train, test = R[R.year < year], R.year == year
        for name, cols in (("p_base", base), ("p_rev", rev)):
            fit = sm.Logit(train["up"], sm.add_constant(train[cols])).fit(disp=0)
            R.loc[test, name] = fit.predict(sm.add_constant(R.loc[test, cols], has_constant="add"))
            if name == "p_rev":
                coefs[year] = fit.params.round(4).to_dict()
    return R, coefs


def reversal_bins(R: pd.DataFrame, oos_start: int = 2024, m: int = 0) -> pd.DataFrame:
    """Realised frequency of an up window against the past-hour return (in forecast sigmas) at the window start."""

    oos = R[(R.year >= oos_start) & (R.m == m)]
    bins = pd.qcut(oos["z60"], 10)
    return oos.groupby(bins, observed=True).agg(z60=("z60", "mean"), realised=("up", "mean"), model=("p_rev", "mean"),
                                              n=("up", "size")).reset_index(drop=True)


def run(work_dir: Path, out_dir: Path, oos_start: int = 2024) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet(work_dir / "close_1m.parquet")["close"]
    F = pd.read_parquet(work_dir / "minute_forecasts.parquet")
    D = window_panel(px, F)
    D["recal"] = recalibrate(D, f"e_{BEST}")
    oos = D[D.year >= oos_start]
    candidates = [("coin flip", None)] + [(f"{c} (Gaussian)", f"g_{c}") for c in F.columns] + [
        (f"{BEST} (empirical tails)", f"e_{BEST}"), (f"{BEST} (empirical, recalibrated)", "recal")]
    rows = []
    for name, col in candidates:
        p = 0.5 if col is None else oos[col]
        rows.append({"model": name, "log_loss": log_loss(p, oos["up"]).mean(), "brier": ((p - oos["up"]) ** 2).mean()})
    scores = pd.DataFrame(rows).set_index("model")
    scores["log_loss_gain_vs_ewma_pct"] = (1 - scores["log_loss"] / scores.loc["ewma (Gaussian)", "log_loss"]) * 100
    scores.attrs["n"] = len(oos)
    by_minute = oos.groupby("m").apply(lambda g: pd.Series({"ewma": log_loss(g["g_ewma"], g["up"]).mean(),
                                                            "best_recalibrated": log_loss(g["recal"], g["up"]).mean()}))
    by_minute["gain_pct"] = (1 - by_minute["best_recalibrated"] / by_minute["ewma"]) * 100
    calibration = []
    for col in (f"e_{BEST}", "g_ewma", "recal"):
        bins = pd.cut(oos[col], np.linspace(0, 1, 11), include_lowest=True)
        g = oos.groupby(bins, observed=True).agg(predicted=(col, "mean"), realised=("up", "mean"), n=("up", "size"))
        calibration.append(g.assign(model=col).reset_index(names="bin"))
    R, coefs = reversal_panel(px, F, D)
    late = R[R.year >= oos_start].assign(l_base=lambda d: log_loss(d.p_base, d.up), l_rev=lambda d: log_loss(d.p_rev, d.up))
    rev_m = late.groupby("m")[["l_base", "l_rev"]].mean().assign(gain_pct=lambda d: (1 - d.l_rev / d.l_base) * 100)
    rev_y = late.groupby("year")[["l_base", "l_rev"]].mean().assign(gain_pct=lambda d: (1 - d.l_rev / d.l_base) * 100)
    diff = (late.l_base - late.l_rev).groupby(late.start.dt.floor("D")).mean()
    rev_y.loc["all", ["l_base", "l_rev"]] = late[["l_base", "l_rev"]].mean().to_numpy()
    rev_y.loc["all", "gain_pct"] = (1 - rev_y.loc["all", "l_rev"] / rev_y.loc["all", "l_base"]) * 100
    rev_y["t_day_clustered"] = np.nan
    rev_y.loc["all", "t_day_clustered"] = diff.mean() / diff.std() * np.sqrt(len(diff))
    tables = {"oos_scores": scores, "oos_by_minute": by_minute, "calibration": pd.concat(calibration, ignore_index=True),
              "reversal_by_minute": rev_m, "reversal_by_year": rev_y,
              "reversal_coefficients": pd.DataFrame(coefs).T.rename_axis("test_year"),
              "reversal_by_past_hour_return": reversal_bins(R, oos_start)}
    for name, table in tables.items():
        table.round(5).to_csv(out_dir / f"{name}.csv", index=name not in ("calibration", "reversal_by_past_hour_return"))
    R[["start", "m", "tau", "up", "year", f"e_{BEST}", "p_base", "p_rev"] + [f"z{k}" for k in LAGS]].to_parquet(
        work_dir / "reversal_panel.parquet")
    D[D.year >= 2025][["start", "m", "x", "up", "g_ewma", f"e_{BEST}", "recal"]].to_parquet(work_dir / "binary_panel.parquet")
    return tables
