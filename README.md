# Pricing 15-minute bitcoin binaries

Polymarket lists a binary market on every 15-minute window of the bitcoin
price: "Up" pays one dollar if bitcoin ends the window at or above its opening
price. This project asks how well these contracts can be priced from public
exchange data, and whether the market prices them efficiently. It has three
parts: forecasting 15-minute volatility, turning the forecast into a
probability at every minute of the window, and comparing those probabilities
with the prices and trades of 37,099 resolved Polymarket markets from
October 2025 to October 2026, with a holdout pre-registered before it was
examined.

The main findings:

- A HAR model of realised variance forecasts 15-minute volatility far better
  than an exponentially weighted average: 22% lower QLIKE loss out of sample
  over 2024-2026 (Diebold-Mariano t = -5.6).
- Bitcoin's 15-minute returns reverse its moves of the previous 5 to 240
  minutes. After the weakest tenth of past-hour returns, 55.2% of windows
  closed up; after the strongest tenth, 45.2%. The effect is stable from
  2022 to 2026 and significant out of sample (t = 7.0).
- Polymarket's price is a better forecaster than any model built on Binance
  data; it reflects the oracle that settles the contract and reacts within
  seconds. But it underweights reversal. In the pre-registered holdout (April
  to October 2026, 20,543 markets) the model added information to the market
  price in the minutes before the window opens (logit coefficient 0.29,
  t = 3.1, p < 0.001), and a taker rule based on it bought the winning side
  at traded prices often enough to earn 1.71 cents per share before fees
  (t = 2.3).
- That edge is exactly what Polymarket's current taker fee takes away: 1.73
  cents per share at those prices, for a net of -0.03 cents. Under the
  cheaper fee schedule of January to March 2026 the same rule earned 1.81 cents
  per share net (t = 3.2). It is the fee, not the market's pricing, that
  closes the trade.

| Pre-registered test | Discovery, Oct 2025-Mar 2026 | Holdout, Apr-Oct 2026 | Holm-adjusted p |
|---|---|---|---|
| H1: model adds information before the window opens (primary) | 0.520 (t 4.41) | 0.292 (t 3.14) | 0.003 |
| H1b: same, during the window | 0.186 (t 4.40) | 0.178 (t 5.44) | <0.001 |
| H2: taker rule, gross P&L per share | +2.11 c (t 3.72) | +1.71 c (t 2.32) | 0.020 |
| H3: taker rule, P&L net of the fee | +1.81 c (t 3.20) | -0.03 c (t -0.04) | 0.51 |
| H4: quoting around the model rather than the market, P&L per fill | +0.91 c (t 1.71) | +1.28 c (t 3.11) | 0.003 |

## Data

Binance spot BTCUSDT one-minute bars from the public archive, January 2019 to
October 2026 (4.1 million bars). Deribit's DVOL index, the 30-day implied
volatility of bitcoin options, hourly from March 2021; Deribit stamps each
hourly candle at its open, so its close is used only from the end of the hour.
For Polymarket, every market slug `btc-updown-15m-{start}` from 2025-10-09 to
2026-11-01 (37,248 windows, 120 never listed): the official resolution and fee
schedule, the one-minute price history of the Up token from five minutes
before the window to five minutes after it, and every taker trade from five
minutes before the start to 90 seconds after it.

A bar stamped t covers t to t+1 minute, so its close is the price at t+1
minute and is used only from then on. All forecasts at time t use bars that
closed by t. Polymarket resolves on Chainlink's BTC/USD stream, not Binance, so
every Polymarket result is scored against the official resolution. The two
agree on 98.5% of discovery windows and 96.7% of holdout windows.

## 1. Volatility forecasts

The target is the sum of squared one-minute log returns over a 15-minute
window. The models are a trailing 30-day mean; an EWMA of squared returns
(half-life 240 minutes, chosen on 2019-2023); HAR-RV on logs (Corsi 2009),
with the realised variance of the last 15 minutes, hour, day and week; HAR on
log variance net of an hour-of-week factor; and that model with DVOL implied
variance added. The HAR models are refitted every January on all earlier
windows. Forecasts are scored on the 99,359 windows from January 2024 to
October 2026 ([table](reports/vol/oos_annual_refit.csv)).

| Model | QLIKE | Gain over EWMA | DM t vs EWMA | Mincer-Zarnowitz R2 (logs) |
|---|---:|---:|---:|---:|
| 30-day mean | 0.814 | -64% | 12.7 | 0.12 |
| EWMA | 0.496 | | | 0.50 |
| HAR | 0.387 | 22.0% | -5.6 | 0.65 |
| HAR, hour-of-week | 0.415 | 16.4% | -2.2 | 0.66 |
| HAR, hour-of-week, DVOL | 0.396 | 20.3% | -4.0 | 0.66 |

The hour-of-week factor and implied volatility raise the explained variance
but do not lower QLIKE. The reason is a weakness worth knowing about: on quiet
weekends Binance's price can stay unchanged for minutes, the log variance of
the last quarter-hour collapses, and a seasonal factor that already expects
a calm weekend pushes the forecast lower still. In 2026, twenty windows, the
largest of them on weekends, account for all of the seasonal model's excess
loss over plain HAR; on 2026 weekdays it is better (QLIKE 0.343 against 0.354)
([by year](reports/vol/oos_annual_refit_by_year.csv),
[figure](reports/figures/vol_qlike_by_year.png)).

## 2. The probability of an up window

At minute m of a window that opened at price K, with current price S and a
forecast F of the variance over the next 15 minutes, the driftless Gaussian
probability is Phi(log(S/K) / sqrt(F (15-m)/15)). The empirical version
replaces the normal distribution with the distribution of standardised
remaining returns at the same minute in earlier years, which has fat tails. A
logistic recalibration fitted on earlier years corrects what remains.

Scored on 1.39 million window-minutes from 2024 to 2026
([table](reports/binary/oos_scores.csv)):

| Probability | Log-loss | Brier |
|---|---:|---:|
| 1/2 | 0.6931 | 0.2500 |
| 30-day mean, Gaussian | 0.4904 | 0.1625 |
| EWMA, Gaussian | 0.4733 | 0.1574 |
| HAR, Gaussian | 0.4646 | 0.1552 |
| HAR, hour-of-week, DVOL, Gaussian | 0.4635 | 0.1549 |
| same, empirical tails | 0.4612 | 0.1543 |
| same, recalibrated | 0.4611 | 0.1543 |

The best model's log-loss is 2.6% below the EWMA model's, and the gain grows
through the window, from 0.3% at the first minute to 8.6% at the last, where
the volatility forecast matters most
([by minute](reports/binary/oos_by_minute.csv)). The Gaussian EWMA prices are
too timid: when they say 75% the window closes up 81% of the time
([calibration](reports/figures/binary_calibration.png)).

## 3. Short-horizon reversal

The driftless model assumes the next quarter-hour is a coin flip apart from
where the price already stands. It is not. Adding past returns over 1, 5, 15,
60 and 240 minutes, each scaled by the volatility forecast, to a logistic
regression refitted each year on earlier years only, gives negative
coefficients on every horizon from 5 minutes up, with almost the same values
in every refit ([coefficients](reports/binary/reversal_coefficients.csv)). Out
of sample, from 2024 to 2026, the reversal model lowers log-loss at every
minute, by 0.30% at the window start and 0.19% over all minutes (t = 7.0 with
standard errors clustered by day). Across deciles of the past hour's return, the share of windows that
close up falls from 55.2% to 45.2%
([figure](reports/figures/reversal_past_hour.png)).

## 4. Polymarket

**Discovery.** On the 16,556 markets from 2025-10-10 to 2026-03-31, the market
price ([log-loss 0.518](reports/polymarket/discovery/forecast_scores.csv)),
taken from the first price print after each minute (median 11 seconds later),
beats the reversal model (0.522): the two are level before the window opens
and the market is better at every minute after. The gap widens toward the end
of the window, where the market sees the settlement oracle and the model
only sees Binance ([figure](reports/figures/polymarket_model_minus_market_discovery.png)).
Two checks support the timing: price prints respond to Binance returns before
them but not to returns more than a minute after them, and a placebo that
picks a random side with the same fill rule earns nothing.

Even so, the model adds information to the market price. A logistic
regression of the resolution on the logits of both gives the model a
coefficient of 0.52 before the window opens (t = 4.4) and 0.19 during it
(t = 4.4). Before the window opens the driftless model is 1/2, so this is
the reversal drift. Polymarket prices move with past returns in the right
direction but by too little.

A taker rule turns this into trades. At each of the five minutes up to the
window start, the ask of each side is the last taker purchase in the previous
30 seconds. Buy the side whose model value beats its ask plus the fee by more
than 2 cents, once per window, and fill at the next actual taker purchase of
that side within 30 seconds. On discovery data it earned 2.11 cents per share
gross and 1.81 net, positive in each of the six months
([tables](reports/polymarket/discovery/taker_trades.csv)). The fee schedule
changed twice: none until 6 January 2026; 0.78 cents per share at a 50-cent
price from 7 January; 1.75 cents from 30 March. With the current fee applied
to the whole discovery sample, the net edge was only 0.65 cents (t = 0.8), so
the protocol named "information real, absorbed by the fee" as the most
likely holdout outcome.

**Holdout.** The protocol
([`protocols/polymarket_reversal.md`](protocols/polymarket_reversal.md))
was committed with the frozen code in `src/btcbin/holdout.py` before any
holdout price, trade or resolution was analysed, and the holdout was then run
once ([results](reports/polymarket/holdout/frozen_tests.csv)). Four of five
tests reject at 5% after a Holm adjustment. The model added information
before the window (0.29, t = 3.1) and during it (0.18, t = 5.4). The taker rule
made 4,344 trades, 20 a day, with a gross edge of 1.71 cents per share
(t = 2.3) and a fee of 1.73 cents, for a net of -0.03 cents
([figure](reports/figures/taker_gross_fee_net.png)). Filling at the last
traded price before the decision instead gives +0.18 cents net. Monthly net
P&L was positive from April to August and negative in September and October
([by month](reports/polymarket/holdout/taker_by_month.csv)). Resting bids two
cents inside the model's fair value lost 1.01 cents per fill to better-informed
takers, against 2.25 cents for bids around the market's own mid (difference
t = 3.1). The model reduces adverse selection, but not by enough to make
quoting profitable without rebates.

## Conclusions

The forecasting results hold up: HAR-type models beat EWMA for 15-minute
bitcoin volatility by about a fifth, and fat-tailed, recalibrated probabilities
beat the Gaussian ones a live pricer would typically use. The reversal effect
is the substantive finding. It is a small drift, worth a few percentage points
of probability, but it is stable across five years. A liquid prediction
market that otherwise prices these contracts better than any exchange-only
model still underweights it, out of sample and under pre-registration.

Whether that is money depends on the fee. Polymarket introduced taker fees on
these markets in January 2026 and raised them in March; at the current rate
the fee equals the edge almost exactly. The result is an inefficiency that is
statistically clear and economically closed to takers, and the holdout's last
two months suggest it may also be shrinking.

## Reproduction

```bash
pip install -e ".[dev]"
btcbin build-data --out data/real
btcbin run vol
btcbin run binary
btcbin run polymarket --sample discovery
btcbin run polymarket --sample holdout
pytest
```

`build-data` downloads the Binance archive (about 100 monthly files), DVOL from
Deribit's public API, and the Polymarket markets, price histories and
pre-window trades (about 150,000 requests; resumable). Vendor data stay under
`data/real/`, which is not tracked; the reports hold aggregate tables and
figures only. `btcbin run vol` writes the minute-level forecasts the later
parts read. The tests check Polymarket's fee formula against its published
table, the timing of price prints, quotes and fills, that passive bids fill
only when traded through, the alignment of the variance target, that features
do not change when future data are altered, and the Gaussian pricing formula.

## Limitations

The ask is inferred from taker trades, not from the order book, and the
median taker purchase used as a fill was 11 shares, so the tests say nothing
about trading more than a few tens of shares a window. Passive fills assume a bid is hit only when
a trade goes through it, with no queue and no rebates. The model sees Binance
one-minute closes, while the market sees the settlement oracle and
second-by-second prices; a faster feed would help the model during the window
but probably not before it, where the result lies. The reversal effect was
known to persist in Binance data through 2026 when the protocol was written;
the holdout tested Polymarket's pricing of it, which had not been examined.
