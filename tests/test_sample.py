import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.special import ndtr

from btcbin import binary, polymarket, vol

TRADE_FIELDS = [
    ("window_ts", pa.int64()), ("time", pa.int64()), ("outcome", pa.int8()),
    ("side", pa.int8()), ("price", pa.float64()), ("size", pa.float64()),
]
PARQUET_SCHEMAS = {
    "btc_1m.parquet": (pa.schema([("close", pa.float64()), ("volume", pa.float32()), ("time", pa.timestamp("us"))]), ["time"]),
    "dvol_1h.parquet": (pa.schema([("dvol", pa.float64()), ("time", pa.timestamp("us"))]), ["time"]),
    "polymarket_trades_discovery.parquet": (pa.schema(TRADE_FIELDS), []),
    "polymarket_trades_holdout.parquet": (pa.schema(TRADE_FIELDS), []),
}
MARKETS = "polymarket_markets.jsonl.gz"
LISTED_FIELDS = {
    "ts": int, "question": str, "conditionId": str, "outcomePrices": str, "outcomes": str,
    "closed": bool, "volume": (float, int, type(None)), "endDate": str, "feesEnabled": bool,
    "feeSchedule": (dict, type(None)), "feeType": (str, type(None)), "tokens": str, "history": list,
}
FEE_FIELDS = {"exponent": int, "rate": float, "rebateRate": float, "takerOnly": bool}


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    out = tmp_path_factory.mktemp("sample")
    subprocess.run([sys.executable, "-m", "btcbin.cli", "sample-data", "--out", str(out), "--seed", "42"], check=True)
    return out


def _assert_fields(record, fields):
    assert record.keys() == fields.keys()
    for key, types in fields.items():
        assert type(record[key]) in (types if isinstance(types, tuple) else (types,)), key


def _market_shapes(path: Path) -> set[str]:
    shapes = set()
    with gzip.open(path, "rt", encoding="utf-8") as markets:
        for record in map(json.loads, markets):
            if record.get("missing"):
                _assert_fields(record, {"ts": int, "missing": bool})
                shapes.add("missing")
                continue

            _assert_fields(record, LISTED_FIELDS)
            assert json.loads(record["outcomes"]) == ["Up", "Down"]
            tokens = json.loads(record["tokens"])
            prices = json.loads(record["outcomePrices"])
            assert len(tokens) == len(prices) == 2
            assert all(type(value) is str for value in tokens + prices)
            if record["feesEnabled"]:
                _assert_fields(record["feeSchedule"], FEE_FIELDS)
                assert type(record["feeType"]) is str
                shapes.add("fee")
            else:
                assert record["feeSchedule"] is None and record["feeType"] is None
                shapes.add("no fee")
            for point in record["history"]:
                _assert_fields(point, {"t": int, "p": (float, int)})
                assert 0 <= point["p"] <= 1
    return shapes


def _parquet_schema(path: Path) -> tuple[pa.Schema, list[str]]:
    schema = pq.read_schema(path)
    index = json.loads(schema.metadata[b"pandas"])["index_columns"]
    return schema.remove_metadata(), index


def test_sample_schema(sample):
    assert {file.name for file in sample.iterdir()} == {*PARQUET_SCHEMAS, MARKETS}
    for name, expected in PARQUET_SCHEMAS.items():
        assert _parquet_schema(sample / name) == expected
    assert _market_shapes(sample / MARKETS) == {"missing", "fee", "no fee"}


def test_sample_matches_real_schema(sample):
    real = Path(__file__).resolve().parents[1] / "data" / "real"
    if not real.is_dir():
        pytest.skip("Real data are not installed")
    for name in PARQUET_SCHEMAS:
        assert _parquet_schema(sample / name) == _parquet_schema(real / name)
    assert _market_shapes(sample / MARKETS) == _market_shapes(real / MARKETS)


def test_sample_runs_through_study(sample):
    px, returns, dvol = vol.load(sample)
    windows = vol.window_table(px, returns, dvol)
    assert not windows.empty
    annual = vol.annual_refit_forecasts(windows, years=(2026,), min_train=1000)
    minute_variance = annual[[binary.BEST]].dropna().reindex(px.index, method="ffill")
    start, end = "2026-04-02", "2026-04-03 23:45"
    panel = binary.window_panel(px, minute_variance, start=start, end=end)
    assert len(panel) == 2 * 96 * 14
    assert panel[f"g_{binary.BEST}"].between(0, 1).all()

    markets = polymarket.load_markets(polymarket.markets_path(sample), "holdout")
    markets = markets[markets.start.between(start, end)]
    assert set(markets.up_official) == {0, 1}
    trades = polymarket.load_trades(sample, set(markets.ts))
    assert set(trades) == set(markets.ts)

    # Use the Gaussian window probabilities and a pre-window reversal signal to
    # exercise execution over the same synthetic windows.
    pre = markets[["start"]].merge(pd.DataFrame({"m": list(polymarket.PRE)}), how="cross")
    known = pd.DatetimeIndex(pre.start + pd.to_timedelta(pre.m, unit="min")) - binary.ONE
    prior_move = returns.rolling(60).sum().reindex(known).to_numpy()
    sd = np.sqrt(minute_variance[binary.BEST].reindex(known).to_numpy())
    pre["p_rev"] = ndtr(-0.045 * prior_move / sd)
    models = pd.concat([pre, panel[["start", "m", f"g_{binary.BEST}"]].rename(columns={f"g_{binary.BEST}": "p_rev"})])

    history = polymarket.history_panel(markets, models)
    assert len(history) == len(markets) * 19
    assert history.mkt_asof.between(0, 1).all()
    assert history.p_rev.between(0, 1).all()
    quotes = polymarket.trades_panel(markets, trades, models)
    assert len(quotes) == len(markets) * len(polymarket.PRE)
    assert quotes[["ask_up", "ask_dn", "fill_up", "fill_dn"]].notna().any().all()

    taker = polymarket.taker_trades(quotes)
    maker = polymarket.maker(markets, trades, models)
    for fills in (taker, maker):
        assert not fills.empty
        assert fills.start.is_unique
        assert fills.pnl.between(-1, 1).all()
