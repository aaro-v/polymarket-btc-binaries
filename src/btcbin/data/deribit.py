"""Deribit DVOL (30-day implied volatility index for BTC), hourly."""

from __future__ import annotations

import time

import pandas as pd

from btcbin.data.http import get_json

URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def dvol_hourly(start: str = "2021-03-24", end: str = "2026-11-01") -> pd.Series:
    """Hourly DVOL close, indexed by the time the close is known.

    Deribit stamps each candle with its open time, so the close of the candle
    stamped 10:00 is only observed at 11:00; the index is shifted accordingly.
    """

    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    step, rows = 3600 * 1000 * 900, []
    while t0 < t1:
        js = get_json(URL, {"currency": "BTC", "start_timestamp": t0, "end_timestamp": min(t0 + step, t1), "resolution": 3600})
        rows += js["result"]["data"]
        t0 += step
        time.sleep(0.2)
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts").sort_values("ts")
    index = pd.to_datetime(frame["ts"], unit="ms") + pd.Timedelta("1h")
    return pd.Series(frame["close"].to_numpy(), index=index, name="dvol")
