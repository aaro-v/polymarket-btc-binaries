import numpy as np
import pandas as pd

from btcbin.binary import price_at, window_panel


def test_price_at_uses_the_bar_that_closed_at_t():
    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    px = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    assert price_at(px, pd.DatetimeIndex([idx[2]]))[0] == 2.0


def test_driftless_gaussian_price():
    idx = pd.date_range("2024-01-01", periods=60 * 24 * 3, freq="1min")
    rng = np.random.default_rng(0)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 5e-4, len(idx)))), index=idx)
    F = pd.DataFrame({"har_seasonal_iv": np.full(len(idx), 15 * 25e-8)}, index=idx)
    D = window_panel(px, F, "2024-01-02", "2024-01-03")
    row = D.iloc[100]
    sd = np.sqrt(15 * 25e-8 * (15 - row.m) / 15)
    from scipy.stats import norm
    assert np.isclose(row["g_har_seasonal_iv"], norm.cdf(row.x / sd))
    flat = D[np.isclose(D.x, 0)]
    assert np.allclose(flat["g_har_seasonal_iv"], 0.5)
