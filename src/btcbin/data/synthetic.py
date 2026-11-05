"""Write a small, deterministic sample of the study's input files."""

from __future__ import annotations

import gzip
import json
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import ndtr

from btcbin.data.polymarket import HOLDOUT_START
from btcbin.polymarket import regime

# Include a December training period and ten market days in each fee regime.
BAR_START = "2025-12-01"
BAR_END = "2026-04-12"
MARKET_STARTS = ("2025-12-22", "2026-03-20", "2026-04-01")
FEE_SCHEDULES = {
    "none": (None, None),
    "legacy": ("crypto_15_min", {"exponent": 2, "rate": 0.25, "rebateRate": 0.2, "takerOnly": True}),
    "current": ("crypto_fees_v2", {"exponent": 1, "rate": 0.07, "rebateRate": 0.2, "takerOnly": True}),
}


def _bars(out: Path, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    minutes = pd.date_range(BAR_START, BAR_END, freq="min", inclusive="left", unit="us", name="time")
    n_hours = len(minutes) // 60
    persistence = 0.98
    slow_vol = lfilter([np.sqrt(1 - persistence**2)], [1, -persistence], rng.normal(size=n_hours))
    hour, weekend = minutes.hour.to_numpy(), minutes.dayofweek.to_numpy() >= 5
    season = 1 + 0.18 * np.cos((hour - 16) * np.pi / 12) - 0.14 * weekend
    sigma = 0.55 / np.sqrt(365 * 1440) * np.exp(0.28 * slow_vol.repeat(60)) * season
    returns = rng.standard_t(4.5, size=len(minutes)) * np.sqrt(2.5 / 4.5) * sigma
    flat = rng.random(len(minutes)) < 0.03
    returns[flat] = 0
    for j in range(60, len(minutes), 15):
        # The next window's drift offsets a small fraction of the last hour's move.
        drift = -0.045 * returns[j - 60:j].sum() / 15
        returns[j:j + 15] += drift * ~flat[j:j + 15]

    log_close = np.log(88_000) + returns.cumsum()
    volume = rng.lognormal(2.3, 1.0, size=len(minutes)) * season * (1 + np.abs(returns) / sigma)
    bars = pd.DataFrame({"close": np.exp(log_close), "volume": volume.astype("float32")}, index=minutes)
    present = rng.random(len(minutes)) >= 0.001
    present[[0, -1]] = True
    bars.loc[present].to_parquet(out / "btc_1m.parquet", compression="zstd")

    hourly_rv = pd.Series((returns.reshape(-1, 60)**2).mean(axis=1))
    annual_vol = np.sqrt(hourly_rv.ewm(halflife=24).mean() * 365 * 1440)
    dvol = (100 * annual_vol.to_numpy() + rng.normal(4, 5, size=n_hours)).clip(15, 140)
    close_times = minutes[::60] + pd.Timedelta(hours=1)
    pd.DataFrame({"dvol": dvol}, index=close_times).to_parquet(out / "dvol_1h.parquet")

    trailing_vol = np.sqrt(pd.Series(returns**2).rolling(60, min_periods=1).mean()).to_numpy()
    return log_close, trailing_vol


def _market(
    start: pd.Timestamp, fee_regime: str, log_close: np.ndarray, trailing_vol: np.ndarray, rng: np.random.Generator,
) -> tuple[dict, np.ndarray, np.ndarray]:
    ts = int(start.timestamp())
    j = (start - pd.Timestamp(BAR_START)) // pd.Timedelta(minutes=1)
    minutes = np.arange(-5, rng.integers(16, 19))
    elapsed = minutes.clip(0, 15)
    move = log_close[j + elapsed - 1] - log_close[j - 1]
    sd = max(trailing_vol[j - 1], 1e-8) * np.sqrt(np.maximum(15 - elapsed, 1))
    prices = ndtr(move / sd) + rng.normal(0, 0.03) + rng.normal(0, 0.018, len(minutes))
    up = bool(log_close[j + 14] >= log_close[j - 1])
    prices[minutes >= 15] = 0.99 if up else 0.01
    prices = prices.clip(0.01, 0.99).round(2)

    fee_type, fees = FEE_SCHEDULES[fee_regime]
    record = {
        "ts": ts,
        "question": f"Bitcoin Up or Down - {start:%Y-%m-%d %H:%M} UTC (synthetic)",
        "conditionId": f"0x{ts:064x}",
        "outcomePrices": json.dumps(["1", "0"] if up else ["0", "1"]),
        "outcomes": json.dumps(["Up", "Down"]),
        "closed": True,
        "volume": round(float(rng.lognormal(np.log(38_900), 1.1)), 2),
        "endDate": (start + pd.Timedelta(minutes=15)).isoformat() + "Z",
        "feesEnabled": fees is not None,
        "feeSchedule": fees,
        "feeType": fee_type,
        "tokens": json.dumps([str(2 * ts), str(2 * ts + 1)]),
        "history": [{"t": ts + 60 * m, "p": p} for m, p in zip(minutes.tolist(), prices.tolist(), strict=True)],
    }
    return record, minutes, prices


def _trades(
    ts: int, minutes: np.ndarray, prices: np.ndarray, rng: np.random.Generator,
) -> pd.DataFrame:
    holdout = ts >= HOLDOUT_START
    median = 163 if holdout else 524
    n = int(np.clip(rng.lognormal(np.log(median), 0.75), 35, 1600))
    offsets = (60 * rng.integers(-5, 2, n) + rng.normal(0, 16, n)).clip(-300, 90).astype("int64")
    offsets.sort()
    outcome = rng.integers(0, 2, n, dtype=np.int8)
    side = np.where(rng.random(n) < 0.93, 1, -1).astype("int8")
    up_price = np.interp(offsets, 60 * minutes, prices)
    mid = np.where(outcome == 0, up_price, 1 - up_price)
    price = (mid + 0.01 * side + rng.normal(0, 0.015, n)).clip(0.01, 0.99).round(2)
    size = rng.lognormal(np.log(8.2 if holdout else 8.8), 1.35, n).round(2).clip(0.01)
    return pd.DataFrame({"window_ts": ts, "time": ts + offsets, "outcome": outcome,
                         "side": side, "price": price, "size": size})


def write_sample(out: Path, seed: int = 42) -> None:
    """Write bars, volatility closes, resolved market records and both trade files."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    log_close, trailing_vol = _bars(out, rng)
    starts = pd.DatetimeIndex(np.concatenate([
        pd.date_range(day, periods=10 * 96, freq="15min").to_numpy() for day in MARKET_STARTS
    ]))
    trades: dict[str, list[pd.DataFrame]] = {"discovery": [], "holdout": []}
    with (
        (out / "polymarket_markets.jsonl.gz").open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
        TextIOWrapper(compressed, encoding="utf-8", newline="\n") as markets,
    ):
        for i, (start, fee_regime) in enumerate(zip(starts, regime(starts), strict=True)):
            if i % 317 == 0:
                record = {"ts": int(start.timestamp()), "missing": True}
            else:
                record, minutes, prices = _market(start, fee_regime, log_close, trailing_vol, rng)
                ts = record["ts"]
                sample = "holdout" if ts >= HOLDOUT_START else "discovery"
                trades[sample].append(_trades(ts, minutes, prices, rng))
            markets.write(json.dumps(record, separators=(",", ":")) + "\n")
    for sample, frames in trades.items():
        pd.concat(frames, ignore_index=True).to_parquet(
            out / f"polymarket_trades_{sample}.parquet", index=False, compression="zstd",
        )
