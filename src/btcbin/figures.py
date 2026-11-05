"""Report figures, drawn from the CSV tables only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _figure(width=7.0, height=3.8):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def vol_by_year(reports: Path, out: Path) -> None:
    t = pd.read_csv(reports / "vol" / "oos_annual_refit_by_year.csv", index_col=0)
    models = [("ewma", "EWMA"), ("har", "HAR"), ("har_seasonal_iv", "HAR + seasonality + DVOL")]
    fig, ax = _figure()
    x, w = np.arange(len(t)), 0.26
    for i, (col, label) in enumerate(models):
        ax.bar(x + (i - 1) * w, t[col], w - 0.02, color=SERIES[i], label=label)
    ax.set_xticks(x, [str(i) for i in t.index])
    _style(ax, "Out-of-sample QLIKE of 15-minute variance forecasts (lower is better)", "", "QLIKE")
    ax.set_ylim(0, t[[c for c, _ in models]].to_numpy().max() * 1.3)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper left")
    _save(fig, out / "vol_qlike_by_year.png")


def calibration(reports: Path, out: Path) -> None:
    t = pd.read_csv(reports / "binary" / "calibration.csv")
    fig, ax = _figure(5.2, 4.6)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    for i, (col, label) in enumerate([("g_ewma", "EWMA, Gaussian"), ("recal", "HAR + seasonality + DVOL, empirical tails, recalibrated")]):
        g = t[t.model == col]
        ax.plot(g.predicted, g.realised, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=label)
    _style(ax, "Calibration, 2024-2026 window-minutes", "predicted probability of an up window", "realised frequency")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    _save(fig, out / "binary_calibration.png")


def reversal(reports: Path, out: Path) -> None:
    t = pd.read_csv(reports / "binary" / "reversal_by_past_hour_return.csv")
    fig, ax = _figure()
    ax.axhline(0.5, color=GRID, linewidth=1)
    ax.plot(t.z60, t.realised, color=SERIES[0], linewidth=2, marker="o", markersize=5, label="realised")
    ax.plot(t.z60, t.model, color=SERIES[1], linewidth=2, linestyle="--", label="reversal model (fitted on earlier years)")
    _style(ax, "Share of windows that close up, by the return over the previous hour (2024-2026)",
           "previous-hour return at the window start, in forecast standard deviations", "share up")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    _save(fig, out / "reversal_past_hour.png")


def market_vs_model(reports: Path, out: Path, sample: str = "discovery") -> None:
    path = reports / "polymarket" / sample / "log_loss_by_minute.csv"
    if not path.exists():
        return
    t = pd.read_csv(path, index_col=0)
    gap = (t["p_rev"] - t["mkt_next"]) * 1000
    fig, ax = _figure()
    ax.bar(t.index, gap, 0.7, color=SERIES[0])
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(-4, 15, 2))
    _style(ax, f"Log-loss of the reversal model minus Polymarket's price ({sample})",
           "minute relative to the window start", "difference x 1000 (positive: market better)")
    _save(fig, out / f"polymarket_model_minus_market_{sample}.png")


def taker_decomposition(reports: Path, out: Path) -> None:
    hold = reports / "polymarket" / "holdout" / "taker.csv"
    if not hold.exists():
        return
    disc = pd.read_csv(reports / "polymarket" / "discovery" / "taker_trades.csv")
    pick = lambda regime: disc[(disc.fee_regime == regime) & (disc.model == "p_rev") & (disc.theta == 0.02)].iloc[0]
    h = pd.read_csv(hold).iloc[0]
    groups = [("Discovery,\nfees as charged", pick("all, actual fees")),
              ("Discovery,\ncurrent fee rule", pick("all, current fee rule applied")),
              ("Holdout,\ncurrent fee", h)]
    fig, ax = _figure()
    x, w = np.arange(len(groups)), 0.26
    for i, (col, label) in enumerate([("gross_c", "gross"), ("fee_c", "fee"), ("net_c", "net")]):
        vals = [g[col] for _, g in groups]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=SERIES[i], label=label)
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(f"{v:+.2f}" if col != "fee_c" else f"{v:.2f}", (b.get_x() + b.get_width() / 2, max(v, 0)),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [g for g, _ in groups])
    _style(ax, "Pre-window taker rule: cents per share", "", "cents per share")
    ax.set_ylim(min(-0.5, ax.get_ylim()[0]), 3.2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper right")
    _save(fig, out / "taker_gross_fee_net.png")


def render(reports: Path) -> None:
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    vol_by_year(reports, out)
    calibration(reports, out)
    reversal(reports, out)
    for sample in ("discovery", "holdout"):
        market_vs_model(reports, out, sample)
    taker_decomposition(reports, out)
