# Protocol: short-horizon reversal in Polymarket's 15-minute bitcoin markets

Written 2026-11-02, before the holdout.

## Question

Polymarket lists a binary market on every 15-minute window: "Up" pays one
dollar if the Chainlink BTC/USD price at the end of the window is at or above
its price at the start. The Binance-only pricing model of this project
(volatility forecast, empirical tails, and a drift from short-horizon reversal
in past returns) is worse than the market as a forecaster, but on discovery
data it adds information the market price lacks before and at the start of
each window. The holdout asks whether that information is still there, and
whether it pays after Polymarket's taker fee.

## Samples

- Discovery: windows starting 2025-10-10 00:00 to 2026-03-31 23:45 UTC
  (16,556 resolved markets). Fees: none until 2026-01-06; from 2026-01-07,
  rate 0.25 with exponent 2 (0.78 USDC per 100 shares at 50 cents); from
  2026-03-30, rate 0.07 with exponent 1 (1.75 USDC per 100 shares at 50 cents).
- Holdout: windows starting 2026-04-01 00:00 to 2026-10-31 23:45 UTC, all
  under the rate 0.07, exponent 1 schedule.

## What has and has not been looked at

All 37,248 window slugs from 2025-10-09 to 2026-11-01 have been downloaded
(metadata, resolution, one-minute price history of the Up token), as well as
the taker trades from five minutes before to 90 seconds after the start of
every window. From the holdout, only the fee-schedule metadata, the count of
missing windows (one in June 2026) and the number of trades per market (an
integrity check) have been read. No holdout price, trade price or resolution
has been analysed.

The Binance side is not blind. Part 2 of the project evaluates the reversal
model on Binance data from 2024 to October 2026, so it is known that reversal
in bitcoin's own returns persisted through 2026. The holdout tests concern
Polymarket prices and resolutions, which have not been examined.

## Frozen model

`p_rev` is the reversal model of `btcbin.binary.reversal_panel`, with the
coefficients for 2026 fitted on Binance windows of 2022-2025 and not refitted:

| const | logit(base) | logit(base) x tau | z1 | z5 | z15 | z60 | z240 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0112 | 1.1065 | -0.2626 | 0.0129 | -0.0428 | -0.0658 | -0.0862 | -0.0511 |

`base` is the driftless empirical-tail probability (1/2 before the window
opens), `tau` the fraction of the window left (1 before it opens), and `zk` the
Binance log return over the past k minutes divided by the square root of k/15
times the HAR-seasonal-DVOL variance forecast, times the square root of tau. At
decision time t = start + m minutes the model uses Binance bars that closed by t.

## Discovery results that motivate the tests

- The market is the better forecaster: log-loss 0.518 for the first price print
  after t (median 11 s later) against 0.522 for the reversal model.
- Logistic regression of the resolution on logit(market) and logit(p_rev),
  standard errors clustered by UTC day: coefficient on the model 0.520 (t 4.41)
  for m = -4..0 and 0.186 (t 4.40) for m = 1..14. Before the window opens the
  driftless model is a constant 1/2, so there the information is the reversal drift alone.
- Taker rule (below), with each market's actual fee: 7,983 trades, gross
  +2.11 cents per share (t 3.72), net +1.81 cents (t 3.20), positive in each of
  the six months. A placebo that picks a random side with the same fills earns
  nothing (0.0 and +0.55 cents gross in the two fee regimes). With the
  current, higher fee applied to the whole discovery sample (and used by the
  rule), the gross edge is +2.40 cents (t 2.84) but the net is +0.65 cents
  (t 0.76), so profitability under the holdout's fee is uncertain.
- Passive quotes (below) around the model earn +0.21 cents per fill against
  -0.64 cents around the market, a difference of 0.91 cents (t 1.71 on
  173 days).

Variants compared on discovery data: thresholds 0 to 0.05, decision minutes
-4..0, 1..5, 6..10 and 11..14, fills at the next price print plus a half-spread
of 0.5, 1 or 2 cents, fills at the next taker trade or at the last taker trade,
the driftless model in place of the reversal model, and passive quotes 1, 2, 3
and 5 cents inside fair value. In-window minutes lost money for a taker
under fees with every variant; the price-print version of the pre-window
taker rule was weaker than the trade version (it nets out to about zero under
the January fee). The frozen choices below are the round middle values, not the
best discovery cells.

## Frozen tests

All on holdout windows. One-sided tests at 5%, with standard errors or
t-statistics computed over UTC days; the p-values of H1 to H4 are also
reported with a Holm adjustment across the five tests.

**H1 (primary).** For m = -4..0, with the market price taken as the first
price-history print within 90 s after t, the logit coefficient on logit(p_rev)
in a regression of the resolution on a constant, logit(market) and
logit(p_rev) is positive.

**H1b.** The same for m = 1..14.

**Taker rule.** At m = -4, -3, -2, -1, 0, the ask of each outcome is the last
taker purchase of that outcome in the 30 s up to t. Buy the outcome whose
model value exceeds its ask plus the market's taker fee per share by more than
0.02, at most one trade per window (the first signal), one share, held to
resolution. The fill is the first taker purchase of that outcome in the 30 s
after t; no fill, no trade.

**H2.** The taker rule's mean gross P&L per share (resolution minus fill
price) is positive.

**H3.** Its mean P&L net of the actual fee is positive.

**Passive quotes.** At each m = -4..0 and for 60 s, bid for Up at fair value
minus 0.02 and for Down at one minus fair value minus 0.02, rounded down to the
cent. A bid fills only if a taker trades through it (a taker sale of that
outcome below the bid, or a taker purchase of the other outcome above one minus
the bid). One fill per window, held to resolution, no fee, no rebate. Fair value
is either p_rev or the market mid (average of the last taker purchase price of
Up and one minus that of Down, over the previous 60 s).

**H4.** The daily mean P&L per fill is higher with the model as fair value
than with the market mid (paired over days).

## Interpretation fixed in advance

- H1 rejected: the market under-weights short-horizon reversal out of sample.
- H1 not rejected: the discovery result does not replicate, and the model
  claims nothing beyond being a cheaper but worse forecaster than the market.
- H2 rejected but not H3: the information is real at traded prices but the
  current fee absorbs it, which is what the counterfactual on discovery data
  predicts as the most likely outcome.
- H3 rejected: the rule is profitable after the fee in force. Capacity is
  limited: the median first taker purchase used as a fill is 11 shares.

The holdout is run once with `btcbin run polymarket --sample holdout`, whose
code (`src/btcbin/holdout.py`) is committed together with this protocol.
