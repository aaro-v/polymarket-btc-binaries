import numpy as np
import pandas as pd

from btcbin.polymarket import load_trades


def test_parquet_trades_are_grouped_by_window_and_sorted_by_time(tmp_path):
    frame = pd.DataFrame({"window_ts": [900, 900, 900, 1800], "time": [905, 901, 905, 1801],
                          "outcome": [0, 1, 1, 0], "side": [1, 1, -1, 1], "price": [0.5, 0.49, 0.51, 0.6], "size": [1.0, 2.0, 3.0, 4.0]})
    frame.astype({"outcome": "int8", "side": "int8"}).to_parquet(tmp_path / "polymarket_trades_discovery.parquet", index=False)
    trades = load_trades(tmp_path, {900})
    assert set(trades) == {900}
    T = trades[900]
    assert np.array_equal(T[:, 0], [901, 905, 905])
    assert np.array_equal(T[:, 3], [0.49, 0.5, 0.51])      # rows with equal times keep their stored order
