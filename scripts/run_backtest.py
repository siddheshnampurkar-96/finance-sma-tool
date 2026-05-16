"""Run the Vibe v2.1 backtest against the bundled real-data CSVs.

Loads daily OHLCV for the S&P 500 index (1990-2018) and AAPL
(2015-2020), pulls daily VIX over the same period, and produces:

- reports/signals.csv        every (date, ticker, label, score, fwd_*)
- reports/summary.csv        per-bucket count + mean/median/hit-rate
- reports/strategy_stats.csv CAGR + max drawdown for strategy vs B&H
- reports/equity_*.png       equity curves
- reports/fwd_return_*.png   30-day forward-return distribution by bucket

Caveats:
- S&P 500 dataset ends 2018-12-21, so this run does NOT see COVID/2022.
- For SP500-as-asset, the "sector proxy" is the index itself
  (self-referential). For AAPL, SP500 stands in for the tech sector.
- Long-only naive strategy: 100% LONG when latest signal is Strong/Mild
  Buy; 100% CASH otherwise. No transaction costs / slippage / sizing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vibe import backtest
from vibe.sample_data import load_aapl, load_sp500, load_vix

REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)


def _strategy_stats_row(name: str, equity: pd.Series) -> dict:
    return {
        "series": name,
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "final_equity": float(equity.iloc[-1]),
        "cagr": backtest.cagr(equity),
        "max_drawdown": backtest.max_drawdown(equity),
    }


def _plot_equity(curve: pd.DataFrame, title: str, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    curve["strategy"].plot(ax=ax, label="Vibe strategy (long/cash)", linewidth=1.4)
    curve["benchmark"].plot(ax=ax, label="Buy & hold", linewidth=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (log scale, starting at 1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def _plot_forward_returns(signals: pd.DataFrame, horizon: int, dest: Path, title: str) -> None:
    bucket_order = ["Strong Buy", "Mild Buy", "Hold", "Mild Sell", "Strong Sell"]
    data = [
        signals.loc[signals["label"] == b, f"fwd_{horizon}"].dropna().to_numpy()
        for b in bucket_order
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=bucket_order, showfliers=False, patch_artist=True)
    colors = ["#2ecc71", "#a8e6cf", "#bdc3c7", "#f7b267", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_title(f"{title} — {horizon}-day forward return by bucket")
    ax.set_ylabel(f"{horizon}-day forward return")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def main() -> None:
    print("Loading datasets...")
    sp500 = load_sp500()
    vix = load_vix()
    aapl = load_aapl()
    print(f"  SP500: {sp500.index.min().date()} → {sp500.index.max().date()} ({len(sp500)} rows)")
    print(f"  VIX  : {vix.index.min().date()} → {vix.index.max().date()} ({len(vix)} rows)")
    print(f"  AAPL : {aapl.index.min().date()} → {aapl.index.max().date()} ({len(aapl)} rows)")

    # SP500 backtest from 1990 (when VIX starts) to dataset end.
    sp_start = pd.Timestamp("1990-12-31")
    sp_end = sp500.index.max()
    print(f"\nRunning SP500 backtest {sp_start.date()} → {sp_end.date()}...")
    sp_result = backtest.run(
        asset_histories={"SP500": sp500},
        sector_proxies={"SP500": sp500},  # self-referential (single-asset case)
        vix_close=vix,
        start=sp_start,
        end=sp_end,
    )
    print(f"  signals: {len(sp_result.signals)}")

    # AAPL backtest — use SP500 as the proxy "sector" benchmark.
    aapl_start = aapl.index.min() + pd.Timedelta(days=320)  # need ≥220 trading days
    aapl_end = aapl.index.max()
    print(f"\nRunning AAPL backtest {aapl_start.date()} → {aapl_end.date()}...")
    aapl_result = backtest.run(
        asset_histories={"AAPL": aapl},
        sector_proxies={"AAPL": sp500},
        vix_close=vix,
        start=aapl_start,
        end=aapl_end,
    )
    print(f"  signals: {len(aapl_result.signals)}")

    # Save signal rows + per-bucket summaries.
    sp_result.signals.to_csv(REPORTS / "signals_sp500.csv", index=False)
    aapl_result.signals.to_csv(REPORTS / "signals_aapl.csv", index=False)
    sp_result.summary.to_csv(REPORTS / "summary_sp500.csv", index=False)
    aapl_result.summary.to_csv(REPORTS / "summary_aapl.csv", index=False)

    # Strategy stats.
    rows = []
    for label, curve in [("SP500 strategy", sp_result.equity_curves["SP500"]["strategy"]),
                        ("SP500 buy&hold", sp_result.equity_curves["SP500"]["benchmark"]),
                        ("AAPL strategy", aapl_result.equity_curves["AAPL"]["strategy"]),
                        ("AAPL buy&hold", aapl_result.equity_curves["AAPL"]["benchmark"])]:
        rows.append(_strategy_stats_row(label, curve))
    stats = pd.DataFrame(rows)
    stats.to_csv(REPORTS / "strategy_stats.csv", index=False)

    # Plots.
    _plot_equity(sp_result.equity_curves["SP500"], "S&P 500 (1991-2018)", REPORTS / "equity_sp500.png")
    _plot_equity(aapl_result.equity_curves["AAPL"], "AAPL (2016-2020)", REPORTS / "equity_aapl.png")
    _plot_forward_returns(sp_result.signals, 30, REPORTS / "fwd_returns_sp500.png", "S&P 500")
    _plot_forward_returns(aapl_result.signals, 30, REPORTS / "fwd_returns_aapl.png", "AAPL")

    # Print summary to stdout.
    print("\n=== SP500 per-bucket summary ===")
    print(sp_result.summary.to_string(index=False))
    print("\n=== AAPL per-bucket summary ===")
    print(aapl_result.summary.to_string(index=False))
    print("\n=== Strategy stats ===")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
