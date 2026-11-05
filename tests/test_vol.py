import numpy as np
import pandas as pd

from btcbin.vol import qlike, window_table


def _synthetic(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1min")
    r = pd.Series(rng.normal(0, 1e-3, n), index=idx)
    r.iloc[0] = 0.0
    px = 100 * np.exp(r.cumsum())
    dvol = pd.Series(50.0, index=pd.date_range("2019-12-31", periods=n // 60 + 48, freq="1h"))
    return px, r, dvol


def test_target_is_variance_from_window_start_to_end():
    px, r, dvol = _synthetic()
    W = window_table(px, r, dvol)
    s = W.index[200]
    path = np.log(px.loc[s - pd.Timedelta("1min"): s + pd.Timedelta("14min")])  # closes at s, ..., s + 15m
    assert np.isclose(W.loc[s, "target"], (np.diff(path.to_numpy()) ** 2).sum())


def test_features_do_not_use_the_future():
    px, r, dvol = _synthetic()
    W = window_table(px, r, dvol)
    s = W.index[300]
    cut = s - pd.Timedelta("1min")                 # keep bars that closed by s
    W2 = window_table(px.loc[:cut + pd.Timedelta("20min")], r.loc[:cut + pd.Timedelta("20min")], dvol)
    r_trunc = r.copy()
    r_trunc.loc[s:] = 0.0                          # destroy everything from s on
    W3 = window_table(px, r_trunc, dvol)
    cols = ["rv15", "rv60", "rv1d", "rv1w", "ewma60", "iv15"]
    assert np.allclose(W.loc[s, cols].astype(float), W3.loc[s, cols].astype(float), equal_nan=True)
    assert np.allclose(W.loc[s, cols].astype(float), W2.loc[s, cols].astype(float), equal_nan=True)


def test_qlike_is_minimised_by_the_true_variance():
    rng = np.random.default_rng(1)
    y = rng.chisquare(1, 200000) * 2.0             # realised variance with mean 2
    losses = {f: qlike(y, f).mean() for f in (1.0, 1.5, 2.0, 2.5, 3.0)}
    assert min(losses, key=losses.get) == 2.0
