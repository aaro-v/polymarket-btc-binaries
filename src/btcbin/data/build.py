"""Build the local datasets under data/real (not tracked: vendor terms do not allow redistribution)."""

from __future__ import annotations

from pathlib import Path

from btcbin.data import binance, deribit, polymarket


def build(out: Path, parts: tuple[str, ...] = ("binance", "deribit", "polymarket")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if "binance" in parts:
        bars = binance.klines_1m("BTCUSDT", "2019-01", "2026-10")
        bars.to_parquet(out / "btc_1m.parquet")
        print(f"binance: {len(bars):,} one-minute bars, {bars.index.min()} to {bars.index.max()}")
    if "deribit" in parts:
        dvol = deribit.dvol_hourly("2021-03-24", "2026-11-01")
        dvol.to_frame().to_parquet(out / "dvol_1h.parquet")
        print(f"deribit: {len(dvol):,} hourly DVOL closes")
    if "polymarket" in parts:
        polymarket.download(out, "2025-10-09", "2026-11-01")
        print("polymarket: markets and pre-window taker trades written")
