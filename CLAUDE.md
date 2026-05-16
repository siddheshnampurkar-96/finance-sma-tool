# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Phases 1–2 complete (engine + walk-forward backtester + DCA simulation). No data layer, brokerage adapter, or UI yet.

## Project layout

```
src/vibe/
  indicators.py   # SMA, ATR, OBV, slope, consecutive_below — pure functions
  macro.py        # MacroContext + compute_vix_high_stress(vix_series)
  engine.py       # AssetEngine.compute(AssetInputs, MacroContext) -> Signal
                  # Signal also carries dca_action + dca_multiplier (round 5)
  backtest.py     # run() walk-forward + simulate_dca() + per-bucket summary
  sample_data.py  # Loaders for the 3 github-raw datasets we backtest against
tests/
  conftest.py     # synthetic OHLCV helpers
  test_indicators.py
  test_engine.py  # one test per rule (5-bucket) + DCA branches
  test_macro.py
  test_backtest.py
scripts/
  run_backtest.py # End-to-end DCA backtest entrypoint
reports/          # Generated CSVs + plots from the last backtest run
```

## Commands

- Install dev deps: `pip install -e ".[dev]"`
- Run tests: `pytest`
- Run a single test: `pytest tests/test_engine.py::test_strong_buy_on_steady_uptrend -v`

## Signal-engine architecture

The engine evaluates one asset on one day and returns a `Signal { label, score, reasons[], inputs{}, dca_action, dca_multiplier }`. Two outputs share the same computation:

**5-bucket label** (transparency / future features):
1. Pre-earnings hard override (`0 < days_to_earnings <= 3` → Hold, score=0).
2. VIX-adjusted parameters: `k = 1.0` (high stress) else `0.65`; bear window `N_bear = 7` else `5`.
3. Base score: `(Close - SMA_200) / (k * ATR_14)`, clamped to `[-1, +1]`.
4. Modifiers (added then clamped): sector amplifies/dampens by sign of base; OBV-vs-price divergence ±; volume confirmation bullish/bearish; SMA-200 slope ±0.10; bear regime soft contribution -0.7.
5. Bucket: `>=+0.7` Strong Buy · `+0.4..+0.7` Mild Buy · `-0.4..+0.4` Hold · `-0.7..-0.4` Mild Sell · `<=-0.7` Strong Sell.

**DCA action** (round-5 primary product output):
- `dca_action ∈ {"pause", "regular", "bulk"}` + `dca_multiplier ∈ [0.0, 5.0]`.
- **Bulk** fires when ALL of: Close ≤ SMA-200, SMA-200 slope > 0, sector_above_sma200, no earnings blackout. Multiplier = `clamp(1.5 + 2.0 * dip_atr, 1.5, 5.0)` where `dip_atr = (SMA_200 - Close) / ATR_14`.
- **Pause** fires only when 3-of-3 structural failures align: confirmed bear regime + falling SMA-200 + sector below its SMA-200.
- Earnings window keeps DCA at regular pace (no bulk, no pause).
- For broad-market index/ETF holdings (no separate sector), pass `sector_above_sma200=True` (the engine's caller decides).

Every rule that fires is recorded in `reasons[]` so the UI can show *why* a signal moved.

## Conventions

- Indicators are pure functions over a pandas DataFrame; NaNs in warm-up periods are preserved.
- The engine assumes at least 220 rows of history (SMA-200 + 20-day slope).
- Tests pin each rule's contribution to the locked spec — any drift in the formula breaks the corresponding test, which is intentional. Update tests and the locked plan together if a contribution value changes.

## Plan reference

Round-1 through round-5 decisions live in `~/.claude/plans/federated-wiggling-frost.md`. Round 5 reframes the product around DCA enhancement (the 5-bucket sell labels stay in the engine but are no longer the primary product). Read it before extending.

## Backtest

`scripts/run_backtest.py` pulls SPX daily 1950-2018 + VIX daily + AAPL 2015-2020 (all from github raw, the only finance hosts the environment allows). Runs walk-forward signal generation, then simulates daily DCA paths: vanilla baseline + 3 enhanced configs (conservative / default / aggressive bulk multipliers) + pause-disabled variant. Outputs go to `reports/`.
