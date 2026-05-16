"""Walk-forward backtester for the Vibe v2.1 signal engine.

Design notes
------------
The backtester is **data-agnostic**: it takes already-loaded OHLCV
DataFrames plus a VIX series and replays them date by date, calling
``AssetEngine.compute`` against the slice of history available *at*
that date (no look-ahead).

For each (ticker, date) it records:
- the bucket label and continuous score
- N-day forward returns at the horizons configured

Aggregates produced:
- per-bucket count, mean / median forward return, hit rate (% positive)
- equity curve of a naive long-only strategy that holds 100% of the
  asset on the day AFTER a Buy signal and 100% cash otherwise
- benchmark buy-and-hold equity curve

The strategy is intentionally crude — no position sizing, transaction
costs, slippage, or risk management. Its purpose is to demonstrate
whether the signal *discriminates*; not to claim a tradeable edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from vibe.engine import AssetEngine, AssetInputs, Signal
from vibe.indicators import sma
from vibe.macro import MacroContext, compute_vix_high_stress

BUY_LABELS = {"Strong Buy", "Mild Buy"}
SELL_LABELS = {"Strong Sell", "Mild Sell"}


@dataclass(frozen=True)
class BacktestConfig:
    forward_horizons: tuple[int, ...] = (10, 30, 60)
    min_history_required: int = 220


@dataclass
class BacktestResult:
    signals: pd.DataFrame
    summary: pd.DataFrame
    equity_curves: dict[str, pd.DataFrame] = field(default_factory=dict)


def _date_index_intersection(asset: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    idx = asset.index
    mask = (idx >= start) & (idx <= end)
    return idx[mask]


def _sector_above_sma200_series(sector: pd.DataFrame) -> pd.Series:
    sma200 = sma(sector["Close"], 200)
    return (sector["Close"] > sma200).rename("sector_above_sma200")


def run(
    asset_histories: dict[str, pd.DataFrame],
    sector_proxies: dict[str, pd.DataFrame],
    vix_close: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Walk forward day by day and collect signals + forward returns.

    Args:
        asset_histories: {ticker: OHLCV DataFrame}.
        sector_proxies: {ticker: sector ETF (or proxy) OHLCV DataFrame}.
            For each ticker, only ``Close`` is required.
        vix_close: daily VIX close indexed by date.
        start: first evaluation date.
        end: last evaluation date (inclusive).
        config: forward horizons + minimum history.

    Returns:
        BacktestResult with a long-format ``signals`` frame, a per-bucket
        ``summary`` frame, and one equity curve per ticker.
    """
    cfg = config or BacktestConfig()
    records: list[dict] = []
    equity_curves: dict[str, pd.DataFrame] = {}

    # Precompute sector "above SMA-200" series once per proxy.
    sector_above_series = {
        ticker: _sector_above_sma200_series(proxy)
        for ticker, proxy in sector_proxies.items()
    }

    # Precompute VIX high-stress flag forward-looking O(N) once.
    vix_sma200 = sma(vix_close, 200)
    vix_std200 = vix_close.rolling(200).std()
    vix_high = (vix_close > vix_sma200 + 1.5 * vix_std200).fillna(False)

    for ticker, history in asset_histories.items():
        dates = _date_index_intersection(history, start, end)
        signals_for_ticker: list[dict] = []

        for date in dates:
            hist_slice = history.loc[:date]
            if len(hist_slice) < cfg.min_history_required:
                continue

            # Sector flag at this date (use the most recent known value).
            sec_series = sector_above_series.get(ticker)
            if sec_series is None:
                sector_above = True
            else:
                sec_view = sec_series.loc[:date]
                if sec_view.empty or pd.isna(sec_view.iloc[-1]):
                    sector_above = True
                else:
                    sector_above = bool(sec_view.iloc[-1])

            vix_view = vix_high.loc[:date]
            stress_today = bool(vix_view.iloc[-1]) if not vix_view.empty else False
            macro = MacroContext(vix_high_stress=stress_today)

            inputs = AssetInputs(
                history=hist_slice,
                sector_above_sma200=sector_above,
            )
            signal = AssetEngine.compute(inputs, macro)
            signals_for_ticker.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "label": signal.label,
                    "score": signal.score,
                    "close": float(history.loc[date, "Close"]),
                    "vix_high_stress": stress_today,
                    "sector_above_sma200": sector_above,
                    "dca_action": signal.dca_action,
                    "dca_multiplier": signal.dca_multiplier,
                    "sma_200": signal.inputs.get("sma_200", float("nan")),
                    "atr_14": signal.inputs.get("atr_14", float("nan")),
                    "sma200_slope_20": signal.inputs.get("sma200_slope_20", float("nan")),
                    "consec_below_lower_band": signal.inputs.get(
                        "consec_below_lower_band", float("nan")
                    ),
                }
            )

        if not signals_for_ticker:
            continue

        sig_df = pd.DataFrame(signals_for_ticker).set_index("date").sort_index()

        # Forward returns
        close_full = history["Close"]
        for h in cfg.forward_horizons:
            fwd = close_full.shift(-h)
            ret = (fwd - close_full) / close_full
            sig_df[f"fwd_{h}"] = sig_df.index.map(ret)

        records.extend(sig_df.reset_index().to_dict("records"))
        equity_curves[ticker] = _build_equity_curve(sig_df, history, start, end)

    signals = pd.DataFrame.from_records(records)
    if not signals.empty:
        signals = signals.sort_values(["ticker", "date"]).reset_index(drop=True)
    summary = _build_summary(signals, cfg.forward_horizons)

    return BacktestResult(signals=signals, summary=summary, equity_curves=equity_curves)


def _build_equity_curve(
    signals: pd.DataFrame,
    history: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Naive long/cash strategy vs buy-and-hold benchmark.

    Position on day t+1 is determined by the signal generated at close
    of day t (no look-ahead). Position is LONG iff the most recent signal
    is in BUY_LABELS; otherwise CASH. Daily close-to-close returns are
    applied to a 1.0 starting equity for both strategy and benchmark.
    """
    eval_window = history.loc[start:end].copy()
    eval_window["return"] = eval_window["Close"].pct_change().fillna(0.0)

    # Carry the latest signal label forward across days without a fresh
    # signal (shouldn't happen with daily evaluation, but defensive).
    sig_aligned = signals["label"].reindex(eval_window.index).ffill()

    # Position for return on day t comes from yesterday's signal.
    position_long = sig_aligned.shift(1).isin(BUY_LABELS).fillna(False)

    strategy_return = np.where(position_long, eval_window["return"], 0.0)
    strategy_equity = (1.0 + pd.Series(strategy_return, index=eval_window.index)).cumprod()
    benchmark_equity = (1.0 + eval_window["return"]).cumprod()

    return pd.DataFrame(
        {
            "strategy": strategy_equity,
            "benchmark": benchmark_equity,
            "position_long": position_long,
            "label": sig_aligned,
        }
    )


def _build_summary(signals: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    bucket_order = ["Strong Buy", "Mild Buy", "Hold", "Mild Sell", "Strong Sell"]
    rows = []
    for label in bucket_order:
        bucket = signals[signals["label"] == label]
        row = {"label": label, "count": int(len(bucket))}
        for h in horizons:
            col = f"fwd_{h}"
            valid = bucket[col].dropna()
            row[f"mean_fwd_{h}"] = float(valid.mean()) if not valid.empty else float("nan")
            row[f"median_fwd_{h}"] = float(valid.median()) if not valid.empty else float("nan")
            row[f"hit_rate_{h}"] = (
                float((valid > 0).mean()) if not valid.empty else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (1 / years) - 1)


# ---------- DCA simulation (round-5 product output) ------------------------


@dataclass(frozen=True)
class DCAFlow:
    """One DCA scenario: per-day contribution amount + outcomes.

    daily_amount: baseline $ contributed every trading day before
        signal-based adjustments.
    enable_bulk: if True, days flagged ``bulk`` get ``daily_amount *
        dca_multiplier`` instead (the extra is "additive" — from a
        separate reserve, not borrowed from other days).
    enable_pause: if True, days flagged ``pause`` get $0.
    """

    daily_amount: float
    enable_bulk: bool = True
    enable_pause: bool = True


def simulate_dca(
    history: pd.DataFrame,
    signals: pd.DataFrame,
    flow: DCAFlow,
) -> pd.DataFrame:
    """Run a daily DCA simulation through ``history`` using ``signals``.

    Each day we deposit a contribution (regular / bulk / paused), buy
    fractional shares at that day's Close, and mark the running totals.

    Returns a DataFrame indexed by date with columns:
        contribution, action, shares_bought, total_shares,
        total_invested, portfolio_value, cost_basis
    """
    sig_aligned = signals.set_index("date").reindex(history.index)
    action = sig_aligned["dca_action"].fillna("regular")
    multiplier = sig_aligned["dca_multiplier"].fillna(1.0)

    rows = []
    total_shares = 0.0
    total_invested = 0.0
    for date, row in history.iterrows():
        act = action.loc[date]
        mult = multiplier.loc[date]
        if act == "pause" and flow.enable_pause:
            contribution = 0.0
        elif act == "bulk" and flow.enable_bulk:
            contribution = flow.daily_amount * float(mult)
        else:
            contribution = flow.daily_amount
        close = float(row["Close"])
        shares_bought = contribution / close if close > 0 else 0.0
        total_shares += shares_bought
        total_invested += contribution
        portfolio_value = total_shares * close
        cost_basis = total_invested / total_shares if total_shares > 0 else float("nan")
        rows.append(
            {
                "date": date,
                "action": act,
                "contribution": contribution,
                "shares_bought": shares_bought,
                "total_shares": total_shares,
                "total_invested": total_invested,
                "portfolio_value": portfolio_value,
                "cost_basis": cost_basis,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def dca_metrics(simulation: pd.DataFrame) -> dict:
    """Roll up a single DCA run into headline numbers."""
    if simulation.empty:
        return {}
    final = simulation.iloc[-1]
    pv_curve = simulation["portfolio_value"]
    return {
        "total_invested": float(final["total_invested"]),
        "total_shares": float(final["total_shares"]),
        "final_value": float(final["portfolio_value"]),
        "profit": float(final["portfolio_value"] - final["total_invested"]),
        "profit_per_dollar": float(
            (final["portfolio_value"] - final["total_invested"]) / final["total_invested"]
        )
        if final["total_invested"] > 0
        else float("nan"),
        "avg_cost_basis": float(final["cost_basis"]),
        "max_drawdown_portfolio_value": max_drawdown(pv_curve.where(pv_curve > 0).dropna()),
    }
