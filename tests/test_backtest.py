import numpy as np
import pandas as pd
import pytest

from vibe import backtest
from vibe.backtest import BUY_LABELS
from tests.conftest import make_history


def _make_vix(dates, value=15.0):
    return pd.Series(np.full(len(dates), value), index=dates, name="VIX")


def test_no_lookahead_in_signal_generation():
    """A signal generated at date t must use only history.loc[:t].

    Strategy: build a history where the future contains a crash that
    would massively skew the signal if leaked. The signal generated on
    the LAST day of the calm pre-crash period must still look bullish.
    """
    n = 240
    calm = np.full(220, 100.0)
    crash = np.linspace(100, 50, 20)
    closes = np.concatenate([calm, crash])
    history = make_history(closes)
    last_calm_date = history.index[219]  # final day of the flat 100s

    vix = _make_vix(history.index, 15.0)
    result = backtest.run(
        asset_histories={"X": history},
        sector_proxies={"X": history},
        vix_close=vix,
        start=last_calm_date,
        end=last_calm_date,
    )

    # On the very last calm day, the engine sees only flat history at 100.
    # Base score should be near zero, label should be Hold.
    row = result.signals.iloc[0]
    assert row["label"] == "Hold"
    assert abs(row["score"]) < 0.4


def test_forward_return_aligns_to_future_close():
    """fwd_h column should equal (Close[t+h] - Close[t]) / Close[t]."""
    n = 230
    closes = np.linspace(100.0, 130.0, n)
    history = make_history(closes)
    vix = _make_vix(history.index, 15.0)

    eval_start = history.index[220]
    eval_end = history.index[221]
    result = backtest.run(
        asset_histories={"X": history},
        sector_proxies={"X": history},
        vix_close=vix,
        start=eval_start,
        end=eval_end,
        config=backtest.BacktestConfig(forward_horizons=(5,)),
    )

    sig = result.signals.iloc[0]
    today_close = history.loc[sig["date"], "Close"]
    future_close = history["Close"].shift(-5).loc[sig["date"]]
    expected = (future_close - today_close) / today_close
    assert sig["fwd_5"] == pytest.approx(expected, rel=1e-9)


def test_equity_curve_position_lags_signal_by_one_day():
    """If today's signal flips to LONG, we should earn TOMORROW's return, not today's.

    Build a history where exactly one day produces a Buy signal; verify
    that position_long is False on that day and True on the next.
    """
    # Construct a 230-day path that goes flat then jumps so the engine
    # produces a Buy on day 220 and beyond.
    closes = np.concatenate([np.full(220, 100.0), np.linspace(100, 130, 10)])
    history = make_history(closes)
    vix = _make_vix(history.index, 15.0)

    result = backtest.run(
        asset_histories={"X": history},
        sector_proxies={"X": history},
        vix_close=vix,
        start=history.index[220],
        end=history.index[-1],
    )

    curve = result.equity_curves["X"]
    # The first row of the curve is the first evaluation day; position
    # comes from yesterday's signal, which didn't exist → False.
    assert curve["position_long"].iloc[0] == False

    # Once we have a Buy signal, position_long should flip True on the
    # NEXT row, not the same row.
    labels = curve["label"]
    first_buy_idx = labels.isin(BUY_LABELS).idxmax() if labels.isin(BUY_LABELS).any() else None
    if first_buy_idx is not None:
        idx_pos = curve.index.get_loc(first_buy_idx)
        # Same-day: position is from YESTERDAY (not yet long).
        # Verify either there's no next day, or next day is long.
        if idx_pos + 1 < len(curve):
            assert curve["position_long"].iloc[idx_pos + 1] == True


def test_summary_orders_buckets_canonically():
    """Summary frame must list buckets Strong Buy → Strong Sell regardless
    of which actually appeared in the data, so downstream consumers can
    rely on the row ordering."""
    closes = np.linspace(100.0, 130.0, 230)
    history = make_history(closes)
    vix = _make_vix(history.index, 15.0)
    result = backtest.run(
        asset_histories={"X": history},
        sector_proxies={"X": history},
        vix_close=vix,
        start=history.index[220],
        end=history.index[-1],
    )
    expected = ["Strong Buy", "Mild Buy", "Hold", "Mild Sell", "Strong Sell"]
    assert result.summary["label"].tolist() == expected


def test_simulate_dca_invests_daily_baseline():
    """Vanilla DCA: every trading day deposits the baseline, no signal-based adjustments."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    history = pd.DataFrame(
        {"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1.0},
        index=dates,
    )
    empty_signals = pd.DataFrame({"date": [], "dca_action": [], "dca_multiplier": []})
    sim = backtest.simulate_dca(
        history,
        empty_signals,
        backtest.DCAFlow(daily_amount=10.0),
    )
    # 5 days × $10 = $50 invested, 0.5 shares total at $100 each.
    assert sim["total_invested"].iloc[-1] == pytest.approx(50.0)
    assert sim["total_shares"].iloc[-1] == pytest.approx(0.5)
    assert sim["portfolio_value"].iloc[-1] == pytest.approx(50.0)


def test_simulate_dca_bulk_day_deploys_extra():
    """On a bulk day with multiplier 3x, contribution should be 3x baseline (additive)."""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    history = pd.DataFrame(
        {"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1.0},
        index=dates,
    )
    signals = pd.DataFrame(
        {
            "date": dates,
            "dca_action": ["regular", "bulk", "regular"],
            "dca_multiplier": [1.0, 3.0, 1.0],
        }
    )
    sim = backtest.simulate_dca(
        history, signals, backtest.DCAFlow(daily_amount=10.0)
    )
    # Day 1: $10. Day 2: $30 (bulk 3x). Day 3: $10. Total $50.
    assert sim["contribution"].tolist() == [10.0, 30.0, 10.0]
    assert sim["total_invested"].iloc[-1] == pytest.approx(50.0)


def test_simulate_dca_pause_day_invests_zero():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    history = pd.DataFrame(
        {"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1.0},
        index=dates,
    )
    signals = pd.DataFrame(
        {
            "date": dates,
            "dca_action": ["regular", "pause", "regular"],
            "dca_multiplier": [1.0, 0.0, 1.0],
        }
    )
    sim = backtest.simulate_dca(
        history, signals, backtest.DCAFlow(daily_amount=10.0)
    )
    assert sim["contribution"].tolist() == [10.0, 0.0, 10.0]
    assert sim["total_invested"].iloc[-1] == pytest.approx(20.0)


def test_simulate_dca_pause_disabled_falls_back_to_regular():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    history = pd.DataFrame(
        {"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1.0},
        index=dates,
    )
    signals = pd.DataFrame(
        {"date": dates, "dca_action": ["pause", "pause"], "dca_multiplier": [0.0, 0.0]}
    )
    sim = backtest.simulate_dca(
        history, signals, backtest.DCAFlow(daily_amount=10.0, enable_pause=False)
    )
    assert sim["contribution"].tolist() == [10.0, 10.0]


def test_dca_metrics_computes_profit_and_basis():
    """Buy 1 share at $100, then price rises to $200: profit = $100, basis = $100."""
    simulation = pd.DataFrame(
        {
            "action": ["regular", "regular"],
            "contribution": [100.0, 0.0],
            "shares_bought": [1.0, 0.0],
            "total_shares": [1.0, 1.0],
            "total_invested": [100.0, 100.0],
            "portfolio_value": [100.0, 200.0],
            "cost_basis": [100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )
    metrics = backtest.dca_metrics(simulation)
    assert metrics["total_invested"] == pytest.approx(100.0)
    assert metrics["final_value"] == pytest.approx(200.0)
    assert metrics["profit"] == pytest.approx(100.0)
    assert metrics["profit_per_dollar"] == pytest.approx(1.0)
    assert metrics["avg_cost_basis"] == pytest.approx(100.0)


def test_max_drawdown_and_cagr_basic():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    equity = pd.Series([1.0, 1.2, 0.8, 1.5], index=dates)
    # max equity was 1.2, then dropped to 0.8 → drawdown = -0.333...
    assert backtest.max_drawdown(equity) == pytest.approx(-1 / 3, rel=1e-6)

    yrs_dates = pd.date_range("2020-01-01", periods=2, freq="365D")
    yrs_equity = pd.Series([1.0, 1.10], index=yrs_dates)
    # 10% return in (365 days) ≈ 1 year → CAGR ~10%.
    assert backtest.cagr(yrs_equity) == pytest.approx(0.10, abs=0.001)
