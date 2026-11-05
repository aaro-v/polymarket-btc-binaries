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

- A HAR model of realised variance beats the best exponentially weighted
  average (a 60-minute half-life, chosen on 2019-2023) by 7.6% in QLIKE loss
  out of sample over 2024-2026 (Diebold-Mariano t = -3.5), and by a similar
  margin in each year except 2026, where the two tie.
- Probabilities with fat-tailed, recalibrated remaining-return distributions
  have 1.1% lower log-loss than Gaussian probabilities on the EWMA forecast
  (t = 18), and 4.8% lower in the last minute of the window.
- Bitcoin's 15-minute returns partly reverse its moves of the previous 5 to
  240 minutes. After the weakest tenth of past-hour returns, 55.2% of windows
  closed up; after the strongest tenth, 45.2%. The effect is significant out of
  sample over 2024-2026 (t = 7.0), strong in 2024 and 2026 and weak in 2025.
- Before each window opens, the reversal model adds information to Polymarket's
  price. In the pre-registered holdout (April to October 2026, 20,543 markets)
  its logit coefficient is 0.29 (t = 3.1, Holm-adjusted p = 0.003), and a
  taker rule based on it bought the winning side at traded prices often enough
  to earn 1.71 cents per share before fees (t = 2.3).
- Polymarket's current taker fee at those prices is 1.73 cents per share, so
  the rule nets -0.03 cents. Under the cheaper schedule of January to March
  2026 the same rule earned 2.14 cents per share net (t = 2.7).

| Pre-registered test | Discovery, Oct 2025-Mar 2026 | Holdout, Apr-Oct 2026 | Holm-adjusted p |
|---|---|---|---|
| H1: model adds information before the window opens (primary) | 0.520 (t 4.41) | 0.292 (t 3.14) | 0.003 |
| H1b: same, during the window | 0.186 (t 4.40) | 0.178 (t 5.44) | <0.001 |
| H2: taker rule, gross P&L per share | +2.11 c (t 3.72) | +1.71 c (t 2.32) | 0.020 |
| H3: taker rule, P&L net of the fee | +1.81 c (t 3.20) | -0.03 c (t -0.04) | 0.51 |
| H4: quoting around the model rather than the market, P&L per fill | +0.91 c (t 1.71) | +1.28 c (t 3.11) | 0.003, not valid (see section 4) |

H4 compares the two quoting rules in different minutes, so it does not
support its conclusion (section 4). Section 5 reports robustness checks.

## Data

Binance spot BTCUSDT one-minute bars from the public archive, January 2019 to
October 2026 (4.1 million bars, about 4,000 missing). Deribit's DVOL index, the
30-day implied volatility of bitcoin options, hourly from March 2021; Deribit
stamps each hourly candle at its open, so its close is used only from the end
of the hour. For Polymarket, every market slug `btc-updown-15m-{start}` from
2025-10-09 to 2026-11-01 (37,248 windows, 120 never listed): the official
resolution and fee schedule, the one-minute price history of the Up token from
five minutes before the window to five minutes after it, and every taker trade
from five minutes before the start to 90 seconds after it.

A bar stamped t covers t to t+1 minute, so its close is the price at t+1
minute and is used only from then on. All forecasts at time t use bars that
closed by t. Polymarket resolves on Chainlink's BTC/USD stream, not Binance, so
every Polymarket result is scored against the official resolution. The two
agree on 98.5% of discovery windows and 96.7% of holdout windows.

## 1. Volatility forecasts

The target is the sum of squared one-minute log returns over a 15-minute
window. The models are a trailing 30-day mean; EWMAs of squared returns with
half-lives of 15, 60 and 240 minutes; HAR-RV on logs (Corsi 2009), with the
realised variance of the last 15 minutes, hour, day and week; HAR on log
variance net of an hour-of-week factor; and that model with DVOL implied
variance added. The HAR models are refitted every January on all earlier
windows. Returns of missing bars are left out, and window features are scaled
from the bars available (at least 90% of each window). Forecasts are scored on
the windows from January 2024 to October 2026
([table](reports/vol/oos_annual_refit.csv)).

| Model | QLIKE | Gain over 60-minute EWMA | DM t | Mincer-Zarnowitz R2 (logs) |
|---|---:|---:|---:|---:|
| 30-day mean | 0.814 | -104% | 15.6 | 0.12 |
| EWMA, 240-minute half-life | 0.496 | -24.5% | 26.4 | 0.50 |
| EWMA, 15-minute half-life | 0.400 | -0.3% | 0.4 | 0.65 |
| EWMA, 60-minute half-life | 0.399 | | | 0.62 |
| HAR | 0.368 | 7.6% | -3.5 | 0.65 |
| HAR, hour-of-week | 0.374 | 6.3% | -1.4 | 0.66 |
| HAR, hour-of-week, DVOL | 0.383 | 3.9% | -0.9 | 0.66 |

HAR beats the 60-minute EWMA in 2024 (QLIKE 0.328 against 0.376) and 2025
(0.341 against 0.378) and ties it in 2026 (0.449 against 0.450). The
hour-of-week factor helps in 2024 and 2025 (0.319, 0.332) and hurts in 2026
(0.489); implied volatility raises the explained variance but does not lower
QLIKE ([by year](reports/vol/oos_annual_refit_by_year.csv),
[figure](reports/figures/vol_qlike_by_year.png)).

Missing bars matter more than they look. Carrying the price forward over them
gives windows with zero realised variance, extreme points for a regression on
log variance and for choosing an EWMA half-life. Treated that way, the
2019-2023 data select a 240-minute half-life, against which HAR looks 22%
better; against the 60-minute EWMA the gain is then only 2.9% and not
significant ([table](reports/vol/oos_annual_refit_missing_as_zero.csv)). The
minute-level forecasts used in parts 2 and 3 were frozen with the protocol
before this was found, and still treat missing bars as zero returns.

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
| EWMA (240-minute), Gaussian | 0.4733 | 0.1574 |
| EWMA (60-minute), Gaussian | 0.4661 | 0.1555 |
| HAR, Gaussian | 0.4646 | 0.1552 |
| HAR, hour-of-week, DVOL, Gaussian | 0.4635 | 0.1549 |
| same, empirical tails | 0.4612 | 0.1543 |
| same, recalibrated | 0.4611 | 0.1543 |

The best model's log-loss is 1.1% below the 60-minute EWMA's (t = 18 on daily
mean differences), and the gain grows through the window, from 0.1% at the
first minute to 4.8% at the last, where the volatility forecast matters most
([by minute](reports/binary/oos_by_minute.csv)). Gaussian EWMA prices are too
timid: when they say 75% the window closes up 79% of the time
([calibration](reports/figures/binary_calibration.png)).

## 3. Short-horizon reversal

The driftless model assumes the next quarter-hour is a coin flip apart from
where the price already stands. It is not. Adding past returns over 1, 5, 15,
60 and 240 minutes, each scaled by the volatility forecast, to a logistic
regression refitted each year on earlier years gives negative coefficients on
every horizon from 5 minutes up
([coefficients](reports/binary/reversal_coefficients.csv)). The coefficients
are similar across refits, partly because each refit's sample contains the
previous one. Out of sample, from 2024 to 2026, the reversal model lowers
log-loss at every minute, by 0.30% at the window start and 0.19% over all
minutes (t = 7.0 on daily mean differences); by year the t-statistics are 7.2,
1.6 and 3.7 ([by year](reports/binary/reversal_by_year.csv)). Across deciles
of the past hour's return, the share of windows that close up falls from 55.2%
to 45.2% ([figure](reports/figures/reversal_past_hour.png)).

## 4. Polymarket

**Discovery.** The 16,556 markets from 2025-10-10 to 2026-03-31 were used to
develop the tests. Compared at the same moment, the model and the market are
close. Before the window opens both are near a coin flip. During it, the
model's log-loss (0.462) is lower than that of the last market price before
each minute (0.495) and higher than that of the first price after it (0.457,
median 11 seconds later)
([table](reports/polymarket/posthoc/forecast_equal_information.csv)). Price
prints respond to Binance returns before them but not to returns more than a
minute after them, so their timestamps are not late.

A logistic regression of the resolution on the logits of the market price and
the model gives the model a coefficient of 0.52 before the window opens
(t = 4.4) and 0.19 during it (t = 4.4). Before the window opens the driftless
model is 1/2, so the information is the reversal drift.

A taker rule turns this into trades. At each of the five minutes up to the
window start, the ask of each side is the last taker purchase in the previous
30 seconds. Buy the side whose model value beats its ask plus the fee by more
than 2 cents, once per window, and fill at the next actual taker purchase of
that side within 30 seconds. On discovery data it earned 2.11 cents per share
gross and 1.81 net: 1.68 cents in the fee-free months to 6 January 2026
(t = 2.2) and 2.14 cents net of a 0.76-cent fee from 7 January to 29 March
(t = 2.7) ([tables](reports/polymarket/discovery/taker_trades.csv)). A placebo
that picks a random side with the same fills earns nothing. The fee rose to
1.75 cents at a 50-cent price on 30 March; with that fee applied to the whole
discovery sample, the net edge was only 0.65 cents (t = 0.8), so the protocol
named "information real, absorbed by the fee" as the most likely holdout
outcome.

**Holdout.** The protocol
([`protocols/polymarket_reversal.md`](protocols/polymarket_reversal.md))
was committed with the frozen code in `src/btcbin/holdout.py` before any
holdout price, trade or resolution was analysed, and the holdout was then run
once ([results](reports/polymarket/holdout/frozen_tests.csv)).

The model added information to the market price before the window (0.29,
t = 3.1) and during it (0.18, t = 5.4). Measured against a single price print,
this test cannot tell a slow market from a noisy price. Averaging the last
print before each minute with the first one after it leaves the coefficient
at 0.26 (t = 2.6). Averaging the first two prints after it, the second about a
minute later, removes it ([checks](reports/polymarket/posthoc/information_price_noise.csv)).
The market therefore catches up with the reversal drift within a minute or two.

The taker rule made 4,344 trades, 20 a day, with a gross edge of 1.71 cents
per share (t = 2.3) and a fee of 1.73 cents, for a net of -0.03 cents
([figure](reports/figures/taker_gross_fee_net.png)). The variants:
- Read strictly as one attempt per window, with no trade if the first signal
  does not fill, the rule makes 3,931 trades at 1.83 cents gross and 0.10
  cents net.
- Filling at the last traded price before the decision gives +0.18 cents net.
- Several purchases often share the first second after the decision. Filling
  at the cheapest or the dearest of them gives +0.14 or -0.14 cents net.

Net P&L was positive each month from April to August and negative in
September and October; in October the gross edge itself was -1.4 cents
([by month](reports/polymarket/holdout/taker_by_month.csv)).

The passive-quote test (H4) rejected as registered, but the rejection is an
artefact. Bids around the market's mid can only be placed in minutes where both
sides traded recently, mostly at the window start, while bids around the model
are placed from four minutes before. Restricted to the same minutes, the two
lose the same amount (-2.30 and -2.25 cents per fill; difference -0.05 cents,
t = -0.1) ([checks](reports/polymarket/posthoc/maker_same_cells.csv)).

## 5. Robustness

These checks are not part of the protocol and do not change the frozen model
or the pre-registered code.

- Missing Binance bars are masked. Carrying prices forward over them instead
  selects a 240-minute EWMA half-life and makes HAR look 22% better than EWMA;
  with masking the selected half-life is 60 minutes and the gain is 7.6%.
- Part 2's EWMA comparison uses the 60-minute EWMA as well: the gain is 1.1%,
  against 2.6% over a 240-minute EWMA.
- H4 does not support its conclusion (section 4).
- Compared at the same moment, the model and the market forecast about
  equally well; the market looks better only against a later market price.
- The 1.81-cent discovery net covers the whole discovery sample, not only the
  January fee schedule. The holdout has a single fee schedule, and the gross
  edge fell between the two samples, so the data show only that the edge and
  the fee are about equal, not that the fee alone stops the trade from paying.
- The discovery information regressions use the same rows as the holdout's.

The checks are in `src/btcbin/posthoc.py` and `reports/polymarket/posthoc/`.

## Conclusions

The forecasting gains are modest. HAR
beats a well-chosen EWMA for 15-minute bitcoin volatility by a few percent, and
fat-tailed, recalibrated probabilities beat Gaussian ones, most of all late in
the window. The reversal drift is small but persistent over five years. In a
liquid prediction market it is priced with a lag: before each window opens the
reversal model adds information to the latest price, and a taker rule built on
it picked winners at traded prices in a pre-registered holdout. Polymarket's
current fee is about the size of that edge, so the trade does not pay today,
and in the last two months of the holdout the edge itself disappeared.

## Reproduction

```bash
pip install -e ".[dev]"
btcbin build-data --out data/real
btcbin run vol
btcbin run binary
btcbin run polymarket --sample discovery
btcbin run polymarket --sample holdout
btcbin run polymarket --sample posthoc
pytest
```

`build-data` downloads the Binance archive (about 100 monthly files), DVOL from
Deribit's public API, and the Polymarket markets, price histories and
pre-window trades (about 150,000 requests; resumable), then stores them
compactly: the one-minute closes and volumes and the trades as parquet, the
market records as gzipped JSON lines. Vendor data stay under `data/real/`,
which is not tracked; the reports hold aggregate tables and figures only. `btcbin run vol` writes the minute-level forecasts the later
parts read. The tests check Polymarket's fee formula against its published
table, the timing of price prints, quotes and fills, that passive bids fill
only when traded through, the alignment of the variance target, that features
do not change when future data are altered, and the Gaussian pricing formula.

## Limitations

The ask is inferred from taker trades, not from the order book, and the
median taker purchase used as a fill was 15 shares in discovery and 10 in the
holdout, so the tests say nothing about trading more than a few tens of
shares a window. Trade timestamps may be block times a few seconds after
matching. Passive fills assume a bid is hit only when a trade goes through it,
with no queue and no rebates. The model sees Binance one-minute closes, while
the market sees the settlement oracle and second-by-second prices. Before the
window opens, the variance forecast that scales past returns uses the previous
quarter-hour's seasonal factor; this is consistent in training and testing and
uses no future data. The reversal effect was known to persist in Binance data
through 2026 when the protocol was written; the holdout tested Polymarket's
pricing of it, which had not been examined.
