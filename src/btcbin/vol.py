"""Part 1: forecasting 15-minute realised variance of bitcoin.

Models: a trailing 30-day mean, an EWMA of squared one-minute returns, HAR-RV
on log variance (Corsi 2009), HAR on deseasonalised log variance with an
hour-of-week factor, and the same with Deribit's DVOL implied variance. All
forecasts use information strictly before the window starts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

EPS = 1e-12
TRAIN_END, OOS_START = "2023-12-31 23:59", "2024-01-01"
HAR_COLS = ["rv15", "rv60", "rv1d", "rv1w"]


def log(x):
    return np.log(x + EPS)


def load(data_dir: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    """One-minute closes on a complete grid, one-minute log returns, and DVOL indexed by the time it is known."""

    px = pd.read_parquet(data_dir / "btc_1m.parquet")["close"]
    px = px.reindex(pd.date_range(px.index.min(), px.index.max(), freq="1min")).ffill()
    r = np.log(px).diff().fillna(0.0)
    dvol = pd.read_parquet(data_dir / "dvol_1h.parquet")["dvol"]
    return px, r, dvol


def implied_variance_15m(dvol: pd.Series) -> pd.Series:
    """DVOL is an annualised percentage for a market that trades every day: variance per 15 minutes."""

    return (dvol / 100) ** 2 / (365 * 96)


def slot_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.day_of_week * 96 + index.hour * 4 + index.minute // 15)


def window_table(px: pd.Series, r: pd.Series, dvol: pd.Series) -> pd.DataFrame:
    """One row per window start s: realised variance from s to s+15m, and features known at s.

    A bar stamped t runs from t to t+1m, so the return of bar t is known at t+1m.
    The target sums squared returns of bars s..s+14; features use bars up to s-1.
    """

    r2 = r**2
    rv15 = r2.rolling(15).sum()
    starts = pd.date_range(px.index[0].ceil("15min"), px.index[-1] - pd.Timedelta("15min"), freq="15min")
    past = lambda n: r2.rolling(n).sum().shift(1).reindex(starts)
    w = pd.DataFrame(index=starts)
    w["target"] = rv15.shift(-14).reindex(starts)
    w["rv15"], w["rv60"], w["rv1d"], w["rv1w"] = past(15), past(60) / 4, past(1440) / 96, past(10080) / 672
    for hl in (15, 60, 240):
        w[f"ewma{hl}"] = r2.ewm(halflife=hl).mean().shift(1).reindex(starts) * 15
    w["rv30d"] = r2.rolling(43200, min_periods=20000).mean().shift(1).reindex(starts) * 15
    iv = implied_variance_15m(dvol)
    w["iv15"] = iv.reindex(iv.index.union(starts)).ffill().shift(1).reindex(starts)
    w["slot"] = slot_of_week(starts)
    return w.dropna(subset=["target", "rv15", "rv1w"])


def seasonal_factor(train: pd.DataFrame) -> pd.Series:
    """Mean deviation of log realised variance by 15-minute slot of the week."""

    y = log(train["target"])
    return (y - y.mean()).groupby(train["slot"]).mean()


def _design(df: pd.DataFrame, seas: pd.Series | None, use_iv: bool) -> pd.DataFrame:
    s = df["slot"].map(seas).fillna(0.0) if seas is not None else 0.0
    X = pd.DataFrame({c: log(df[c]) - (s if c == "rv15" else 0.0) for c in HAR_COLS})
    if use_iv:
        X["iv"] = log(df["iv15"])
    return sm.add_constant(X, has_constant="add")


def fit_har(train: pd.DataFrame, seasonal: bool, use_iv: bool):
    """OLS of (deseasonalised) log RV on log HAR terms; returns the fit, the seasonal factor and the residual variance."""

    if use_iv:
        train = train.dropna(subset=["iv15"])
    seas = seasonal_factor(train) if seasonal else None
    y = log(train["target"]) - (train["slot"].map(seas).fillna(0.0) if seas is not None else 0.0)
    fit = sm.OLS(y, _design(train, seas, use_iv), missing="drop").fit()
    return fit, seas, float(np.var(fit.resid))


def predict_har(fit, seas, s2: float, df: pd.DataFrame, use_iv: bool) -> pd.Series:
    s = df["slot"].map(seas).fillna(0.0) if seas is not None else 0.0
    return np.exp(fit.predict(_design(df, seas, use_iv)) + s + s2 / 2)       # log-normal bias correction


def qlike(y, f):
    return y / f - np.log(y / f) - 1


def fixed_split_forecasts(W: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit every model on 2019-2023 and forecast every window."""

    train = W.loc[:TRAIN_END]
    F = pd.DataFrame(index=W.index)
    F["constant_30d"] = W["rv30d"]
    hl = min((15, 60, 240), key=lambda h: qlike(train["target"] + EPS, train[f"ewma{h}"]).mean())
    F["ewma"] = W[f"ewma{hl}"]
    coefs = {"ewma_half_life_minutes": hl}
    for name, seasonal, use_iv in [("har", False, False), ("har_seasonal", True, False), ("har_seasonal_iv", True, True)]:
        fit, seas, s2 = fit_har(train, seasonal, use_iv)
        F[name] = predict_har(fit, seas, s2, W, use_iv)
        coefs[name] = fit.params.round(4).to_dict()
    return F, coefs


def annual_refit_forecasts(W: pd.DataFrame, years=range(2021, 2027), min_train: int = 20000) -> pd.DataFrame:
    """Each calendar year is forecast with models fitted on all earlier years only."""

    F = pd.DataFrame(index=W.index, columns=["har", "har_seasonal", "har_seasonal_iv"], dtype=float)
    for year in years:
        train, sl = W.loc[: f"{year - 1}-12-31 23:59"], slice(f"{year}-01-01", f"{year}-12-31 23:59")
        for name, seasonal, use_iv in [("har", False, False), ("har_seasonal", True, False), ("har_seasonal_iv", True, True)]:
            if (train["iv15"].notna().sum() if use_iv else len(train)) < min_train:
                continue
            fit, seas, s2 = fit_har(train, seasonal, use_iv)
            F.loc[sl, name] = predict_har(fit, seas, s2, W.loc[sl], use_iv)
    F["ewma"], F["constant_30d"] = W["ewma240"], W["rv30d"]
    return F


def evaluate(W: pd.DataFrame, F: pd.DataFrame, start: str = OOS_START, dm_lags: int = 96) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QLIKE, log MSE, Mincer-Zarnowitz regressions and Diebold-Mariano t-statistics against EWMA."""

    f = F.loc[start:].dropna()
    y = W["target"].reindex(f.index)
    keep = y > 0
    y, f = y[keep], f[keep]
    losses, rows = {}, []
    for c in f.columns:
        losses[c] = qlike(y, f[c])
        mz = sm.OLS(np.log(y), sm.add_constant(np.log(f[c]))).fit()
        rows.append({"model": c, "qlike": losses[c].mean(), "log_mse": ((np.log(y) - np.log(f[c])) ** 2).mean(),
                     "mz_r2_log": mz.rsquared, "mz_slope_log": mz.params.iloc[1]})
    table = pd.DataFrame(rows).set_index("model")

    def dm(a: str) -> float:
        d = losses[a] - losses["ewma"]
        return float(sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": dm_lags}).tvalues.iloc[0])

    table["dm_t_vs_ewma"] = [dm(c) if c != "ewma" else np.nan for c in table.index]
    table["qlike_gain_vs_ewma_pct"] = (1 - table["qlike"] / table.loc["ewma", "qlike"]) * 100
    by_year = pd.DataFrame({c: v.groupby(v.index.year).mean() for c, v in losses.items()})
    table.attrs["windows"] = int(len(y))
    return table, by_year


def minute_forecasts(px: pd.Series, r: pd.Series, dvol: pd.Series, W: pd.DataFrame, years=range(2020, 2027)) -> pd.DataFrame:
    """Forecast of variance over the next 15 minutes, made at the close of every one-minute bar.

    The value stamped t uses bars up to and including t, so it is known at t+1m.
    HAR models are refitted every January on all earlier windows.
    """

    r2, idx = r**2, px.index
    feat = pd.DataFrame(index=idx)
    feat["rv15"], feat["rv60"] = r2.rolling(15).sum(), r2.rolling(60).sum() / 4
    feat["rv1d"], feat["rv1w"] = r2.rolling(1440).sum() / 96, r2.rolling(10080).sum() / 672
    iv = implied_variance_15m(dvol)
    feat["iv15"] = iv.reindex(iv.index.union(idx)).ffill().shift(1).reindex(idx)
    feat["slot"] = slot_of_week(idx)
    out = {"constant_30d": r2.rolling(43200, min_periods=20000).mean() * 15, "ewma": r2.ewm(halflife=240).mean() * 15}
    for name, seasonal, use_iv in [("har", False, False), ("har_seasonal_iv", True, True)]:
        pred = pd.Series(np.nan, index=idx)
        for year in years:
            train = W.loc[: f"{year - 1}-12-31 23:59"]
            if use_iv:
                train = train.dropna(subset=["iv15"])
            if len(train) < 20000:
                continue
            fit, seas, s2 = fit_har(train, seasonal, use_iv)
            sl = slice(f"{year}-01-01", f"{year}-12-31 23:59")
            pred.loc[sl] = predict_har(fit, seas, s2, feat.loc[sl], use_iv)
        out[name] = pred
    return pd.DataFrame(out).astype("float32")


def run(data_dir: Path, out_dir: Path, work_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    px, r, dvol = load(data_dir)
    W = window_table(px, r, dvol)
    F, coefs = fixed_split_forecasts(W)
    fixed, fixed_years = evaluate(W, F)
    refit = annual_refit_forecasts(W)
    refit_table, refit_years = evaluate(W, refit)
    pd.Series({k: str(v) for k, v in coefs.items()}).to_csv(out_dir / "coefficients_fit_2019_2023.csv", header=["value"])
    tables = {"oos_fixed_fit": fixed, "oos_fixed_fit_by_year": fixed_years,
              "oos_annual_refit": refit_table, "oos_annual_refit_by_year": refit_years}
    for name, table in tables.items():
        table.round(5).to_csv(out_dir / f"{name}.csv")
    seasonal_factor(W.loc[:TRAIN_END]).to_frame("log_rv_deviation").to_csv(out_dir / "seasonal_factor_2019_2023.csv")
    minute_forecasts(px, r, dvol, W).to_parquet(work_dir / "minute_forecasts.parquet")
    px.to_frame("close").to_parquet(work_dir / "close_1m.parquet")
    return tables
