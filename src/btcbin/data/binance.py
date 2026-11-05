"""Binance spot BTCUSDT one-minute klines from the public archive (data.binance.vision)."""

from __future__ import annotations

import concurrent.futures as cf
import io
import zipfile

import pandas as pd

from btcbin.data.http import get

BASE = "https://data.binance.vision/data/spot"


def _parse(body: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        raw = archive.read(archive.namelist()[0]).decode()
    frame = pd.read_csv(io.StringIO(raw), header=None if raw[:1].isdigit() else 0).iloc[:, :6]
    frame.columns = ["open_time", "open", "high", "low", "close", "volume"]
    stamp = pd.to_numeric(frame["open_time"])
    stamp = stamp.where(stamp < 1e14, stamp // 1000)            # archive switched to microseconds in 2025
    frame["time"] = pd.to_datetime(stamp, unit="ms")
    return frame[["time", "open", "high", "low", "close", "volume"]]


def _fetch(url: str) -> bytes | None:
    response = get(url)
    return None if response is None else response.content


def klines_1m(symbol: str = "BTCUSDT", start: str = "2019-01", end: str = "2026-10") -> pd.DataFrame:
    """One-minute bars indexed by bar open time (UTC). Months not yet archived fall back to daily files."""

    months = pd.period_range(start, end, freq="M")
    urls = [f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{m}.zip" for m in months]
    with cf.ThreadPoolExecutor(8) as pool:
        bodies = list(pool.map(_fetch, urls))
    frames = [_parse(b) for b in bodies if b is not None]
    for month, body in zip(months, bodies, strict=True):
        if body is not None:
            continue
        days = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
        daily = [f"{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{d:%Y-%m-%d}.zip" for d in days]
        with cf.ThreadPoolExecutor(8) as pool:
            frames += [_parse(b) for b in pool.map(_fetch, daily) if b is not None]
    bars = pd.concat(frames).drop_duplicates("time").sort_values("time").set_index("time")
    return bars.astype("float64")
