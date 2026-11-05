"""Polymarket 'Bitcoin Up or Down - 15 minutes' markets.

Each market covers one 15-minute window and is found by its slug
``btc-updown-15m-{window start, unix seconds}``. For every window we keep the
official resolution and fee schedule (Gamma API), the one-minute price history
of the Up token from five minutes before the start to five minutes after the
end (CLOB API), and every taker trade from five minutes before the start to 90
seconds after it (Data API).
"""

from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from btcbin.data.http import get_json

GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com/prices-history"
TRADES = "https://data-api.polymarket.com/trades"


def market(ts: int) -> dict:
    events = get_json(GAMMA, {"slug": f"btc-updown-15m-{ts}"})
    if not events:
        return {"ts": ts, "missing": True}
    m = events[0]["markets"][0]
    rec = {"ts": ts, "question": m.get("question"), "conditionId": m.get("conditionId"),
           "outcomePrices": m.get("outcomePrices"), "outcomes": m.get("outcomes"), "closed": m.get("closed"),
           "volume": m.get("volumeNum"), "endDate": m.get("endDate"), "feesEnabled": m.get("feesEnabled"),
           "feeSchedule": m.get("feeSchedule"), "feeType": m.get("feeType"), "tokens": m.get("clobTokenIds")}
    up_token = json.loads(m["clobTokenIds"])[0]
    history = get_json(CLOB, {"market": up_token, "startTs": ts - 300, "endTs": ts + 1200, "fidelity": 1})
    rec["history"] = (history or {}).get("history", [])
    return rec


def taker_trades(ts: int, condition_id: str) -> dict:
    """Taker trades in [start - 300 s, start + 90 s] as (time, outcome index, +1 buy / -1 sell, price, size)."""

    rows, offset = [], 0
    while True:
        js = get_json(TRADES, {"market": condition_id, "takerOnly": "true", "start": ts - 300, "end": ts + 90,
                               "limit": 500, "offset": offset})
        if js is None:
            return {"ts": ts, "error": True}
        rows += [(x["timestamp"], x["outcomeIndex"], 1 if x["side"] == "BUY" else -1, x["price"], x["size"]) for x in js]
        if len(js) < 500 or offset >= 3000:
            return {"ts": ts, "trades": rows}
        offset += 500


def _done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {r["ts"] for r in map(json.loads, f) if not r.get("error")}


def download(out_dir: Path, start: str = "2025-10-09", end: str = "2026-11-01", threads: int = 8) -> None:
    """Append every window's market record, then its pre-window taker trades; resumable."""

    out_dir.mkdir(parents=True, exist_ok=True)
    markets_path, trades_path = out_dir / "polymarket_markets.jsonl", out_dir / "polymarket_trades.jsonl"
    first = int(pd.Timestamp(start, tz="UTC").timestamp()) // 900 * 900
    last = int(pd.Timestamp(end, tz="UTC").timestamp()) // 900 * 900
    done = _done(markets_path)
    todo = [t for t in range(first, last, 900) if t not in done]
    with markets_path.open("a") as f, cf.ThreadPoolExecutor(threads) as pool:
        for rec in pool.map(market, todo):
            f.write(json.dumps(rec) + "\n")
    with markets_path.open() as f:
        ids = sorted({(r["ts"], r["conditionId"]) for r in map(json.loads, f) if not r.get("missing") and r.get("conditionId")})
    done = _done(trades_path)
    todo = [(t, c) for t, c in ids if t not in done]
    with trades_path.open("a") as f, cf.ThreadPoolExecutor(max(threads - 2, 1)) as pool:
        for rec in pool.map(lambda tc: taker_trades(*tc), todo):
            f.write(json.dumps(rec) + "\n")
    compact(out_dir)


HOLDOUT_START = 1775001600            # 2026-04-01 00:00 UTC


def compact(out_dir: Path) -> None:
    """Gzip the market records and store the trades as parquet, one file per sample (discovery, holdout)."""

    with (out_dir / "polymarket_markets.jsonl").open("rb") as src, gzip.open(out_dir / "polymarket_markets.jsonl.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    frames = []
    with (out_dir / "polymarket_trades.jsonl").open() as f:
        for r in map(json.loads, f):
            if r.get("error") or not r.get("trades"):
                continue
            T = np.array(r["trades"], dtype=float).reshape(-1, 5)
            frames.append(pd.DataFrame({"window_ts": r["ts"], "time": T[:, 0].astype("int64"), "outcome": T[:, 1].astype("int8"),
                                        "side": T[:, 2].astype("int8"), "price": T[:, 3], "size": T[:, 4]}))
    trades = pd.concat(frames, ignore_index=True)
    for name, sel in (("discovery", trades["window_ts"] < HOLDOUT_START), ("holdout", trades["window_ts"] >= HOLDOUT_START)):
        pq.write_table(pa.Table.from_pandas(trades[sel].reset_index(drop=True), preserve_index=False),
                       out_dir / f"polymarket_trades_{name}.parquet", compression="zstd", compression_level=19,
                       use_byte_stream_split=["price", "size"])
