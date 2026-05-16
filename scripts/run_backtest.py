"""Round-5 DCA backtest: vanilla daily DCA vs Vibe-enhanced DCA.

Loads daily OHLCV for the S&P 500 index (1990-2018) and AAPL
(2015-2020), pulls daily VIX, runs the Vibe engine on each (which
emits ``dca_action`` and ``dca_multiplier`` alongside the 5-bucket
label), and produces:

- reports/dca_metrics.csv          headline table (vanilla + grid)
- reports/dca_signals_sp500.csv    every (date, dca_action, dca_multiplier)
- reports/dca_signals_aapl.csv
- reports/dca_costbasis_sp500.png  cost basis over time, vanilla vs enhanced
- reports/dca_costbasis_aapl.png
- reports/dca_value_sp500.png      portfolio value over time
- reports/dca_value_aapl.png

The signal is recomputed per-asset; sensitivity grid varies the bulk
multiplier formula (base, slope, cap) by post-processing the existing
signals frame, so it costs no extra engine evaluations.

Caveats: SP500 dataset ends 2018-12-21 (no COVID/2022). AAPL uses SP500
as the sector proxy. No transaction costs / slippage / fractional-share
limits modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vibe import backtest
from vibe.engine import AssetEngine
from vibe.sample_data import load_aapl, load_sp500, load_vix

REPORTS = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

DAILY_AMOUNT = 10.0  # $10/day baseline; ratios are what matter


@dataclass(frozen=True)
class BulkConfig:
    name: str
    base: float
    slope: float
    cap: float


GRID = [
    BulkConfig("conservative", base=1.5, slope=1.0, cap=3.0),
    BulkConfig("default", base=1.5, slope=2.0, cap=5.0),
    BulkConfig("aggressive", base=1.5, slope=4.0, cap=8.0),
]


def _override_multiplier(signals: pd.DataFrame, cfg: BulkConfig) -> pd.Series:
    """Re-derive the bulk multiplier with overridden parameters.

    Only days with dca_action=='bulk' are affected; pause/regular days
    keep their original multiplier (0.0 / 1.0).
    """
    dip_atr = ((signals["sma_200"] - signals["close"]) / signals["atr_14"]).clip(lower=0)
    new_mult = (cfg.base + cfg.slope * dip_atr).clip(lower=cfg.base, upper=cfg.cap)
    return signals["dca_multiplier"].where(
        signals["dca_action"] != "bulk", new_mult
    )


def _simulate_with_override(history, signals, cfg: BulkConfig, *, enable_pause=True):
    sig = signals.copy()
    sig["dca_multiplier"] = _override_multiplier(sig, cfg)
    flow = backtest.DCAFlow(
        daily_amount=DAILY_AMOUNT, enable_bulk=True, enable_pause=enable_pause
    )
    return backtest.simulate_dca(history, sig, flow)


def _simulate_vanilla(history):
    flow = backtest.DCAFlow(
        daily_amount=DAILY_AMOUNT, enable_bulk=False, enable_pause=False
    )
    empty = pd.DataFrame({"date": [], "dca_action": [], "dca_multiplier": []})
    return backtest.simulate_dca(history, empty, flow)


def _plot_two_series(
    sim_vanilla: pd.DataFrame,
    sim_enhanced: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    dest: Path,
    log_y: bool = False,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    sim_vanilla[column].plot(ax=ax, label="Vanilla DCA", linewidth=1.0, alpha=0.85)
    sim_enhanced[column].plot(ax=ax, label="Vibe-enhanced DCA", linewidth=1.4)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def _row(asset: str, label: str, sim: pd.DataFrame) -> dict:
    m = backtest.dca_metrics(sim)
    return {
        "asset": asset,
        "config": label,
        "total_invested": round(m["total_invested"], 2),
        "final_value": round(m["final_value"], 2),
        "profit": round(m["profit"], 2),
        "profit_per_$": round(m["profit_per_dollar"], 4),
        "avg_cost_basis": round(m["avg_cost_basis"], 4),
        "max_dd_pv": round(m["max_drawdown_portfolio_value"], 4),
    }


def _summarize_action_counts(signals: pd.DataFrame) -> dict:
    if signals.empty:
        return {}
    counts = signals["dca_action"].value_counts().to_dict()
    return {
        "bulk_days": int(counts.get("bulk", 0)),
        "regular_days": int(counts.get("regular", 0)),
        "pause_days": int(counts.get("pause", 0)),
        "total_days": int(len(signals)),
    }


def run_one(
    asset_name: str,
    history: pd.DataFrame,
    sector_proxy: pd.DataFrame | None,
    vix: pd.Series,
    start: pd.Timestamp,
):
    """Backtest one asset.

    sector_proxy=None means "this is a broad-market ETF/index; skip the
    sector gate" (sector_above defaults to True). Required for SP500
    which would otherwise be self-referentially contradictory.
    """
    end = history.index.max()
    print(f"\n--- {asset_name}: {start.date()} → {end.date()} ---")
    sector_proxies = {} if sector_proxy is None else {asset_name: sector_proxy}
    result = backtest.run(
        asset_histories={asset_name: history},
        sector_proxies=sector_proxies,
        vix_close=vix,
        start=start,
        end=end,
    )
    signals = result.signals
    signals.to_csv(REPORTS / f"dca_signals_{asset_name.lower()}.csv", index=False)
    print(f"  signal-days: {_summarize_action_counts(signals)}")

    eval_window = history.loc[start:end]
    sim_vanilla = _simulate_vanilla(eval_window)

    rows = [_row(asset_name, "vanilla", sim_vanilla)]
    sims = {"vanilla": sim_vanilla}
    for cfg in GRID:
        sim = _simulate_with_override(eval_window, signals, cfg, enable_pause=True)
        rows.append(_row(asset_name, f"enhanced/{cfg.name}", sim))
        sims[cfg.name] = sim
    # also: default without pause
    sim_no_pause = _simulate_with_override(eval_window, signals, GRID[1], enable_pause=False)
    rows.append(_row(asset_name, "enhanced/default-nopause", sim_no_pause))
    sims["default-nopause"] = sim_no_pause

    # Plots use the "default" enhanced config vs vanilla.
    _plot_two_series(
        sim_vanilla,
        sims["default"],
        "cost_basis",
        "Average cost basis ($/share)",
        f"{asset_name} — cost basis: vanilla vs enhanced",
        REPORTS / f"dca_costbasis_{asset_name.lower()}.png",
    )
    _plot_two_series(
        sim_vanilla,
        sims["default"],
        "portfolio_value",
        "Portfolio value ($)",
        f"{asset_name} — portfolio value: vanilla vs enhanced",
        REPORTS / f"dca_value_{asset_name.lower()}.png",
        log_y=True,
    )
    _plot_two_series(
        sim_vanilla,
        sims["default"],
        "total_invested",
        "Cumulative invested ($)",
        f"{asset_name} — cumulative $ deployed",
        REPORTS / f"dca_invested_{asset_name.lower()}.png",
    )

    return rows, signals


def main() -> None:
    print("Loading datasets...")
    sp500 = load_sp500()
    vix = load_vix()
    aapl = load_aapl()
    print(f"  SP500: {sp500.index.min().date()} → {sp500.index.max().date()}")
    print(f"  VIX  : {vix.index.min().date()} → {vix.index.max().date()}")
    print(f"  AAPL : {aapl.index.min().date()} → {aapl.index.max().date()}")

    all_rows = []

    sp_start = pd.Timestamp("1990-12-31")
    # SP500 is a broad-market index; no separate "sector" exists, so we
    # pass None and the engine treats sector_above as True throughout.
    rows, _ = run_one("SP500", sp500, None, vix, sp_start)
    all_rows.extend(rows)

    aapl_start = aapl.index.min() + pd.Timedelta(days=350)  # need 240 trading days warmup
    rows, _ = run_one("AAPL", aapl, sp500, vix, aapl_start)
    all_rows.extend(rows)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(REPORTS / "dca_metrics.csv", index=False)

    print("\n=== DCA METRICS ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
