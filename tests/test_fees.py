import numpy as np

from btcbin.polymarket import fee_per_share


def test_current_crypto_fee_matches_published_table():
    # Polymarket fee table for crypto (rate 0.07), USDC per 100 shares
    table = {0.01: 0.07, 0.10: 0.63, 0.30: 1.47, 0.50: 1.75, 0.70: 1.47, 0.90: 0.63, 0.99: 0.07}
    for price, fee in table.items():
        assert abs(100 * fee_per_share(price, 0.07, 1) - fee) < 0.006


def test_january_2026_schedule_peaks_at_156_basis_points():
    fee = fee_per_share(0.5, 0.25, 2)
    assert np.isclose(fee, 0.0078125)
    assert np.isclose(fee / 0.5, 0.015625)          # effective rate at 50 cents


def test_fee_is_zero_when_disabled_and_symmetric_in_dollars():
    assert fee_per_share(0.42, 0.0, 1) == 0.0
    assert np.isclose(fee_per_share(0.3, 0.07, 1), fee_per_share(0.7, 0.07, 1))
