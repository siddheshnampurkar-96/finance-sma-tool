# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Phase 1 complete: the standalone signal engine (`vibe` Python package) is implemented and unit-tested. No data layer, brokerage adapter, backtester, or UI yet — those are deferred to later phases per the locked plan.

## Project layout

```
src/vibe/
  indicators.py   # SMA, ATR, OBV, slope, consecutive_below — pure functions
  macro.py        # MacroContext + compute_vix_high_stress(vix_series)
  engine.py       # AssetEngine.compute(AssetInputs, MacroContext) -> Signal
tests/
  conftest.py     # synthetic OHLCV helpers
  test_indicators.py
  test_engine.py  # one test per signal branch + integration tests for bucket landings
  test_macro.py
```

## Commands

- Install dev deps: `pip install -e ".[dev]"`
- Run tests: `pytest`
- Run a single test: `pytest tests/test_engine.py::test_strong_buy_on_steady_uptrend -v`

## Signal-engine architecture

The engine evaluates one asset on one day and returns a `Signal { label, score, reasons[], inputs{} }`. The locked formula:

1. **Pre-earnings hard override**: if `0 < days_to_earnings <= 3`, return Hold immediately with score=0.
2. **VIX-adjusted parameters**: `k = 1.0` (high stress) else `0.65`; bear-regime window `N_bear = 7` else `5`.
3. **Base score**: `(Close - SMA_200) / (k * ATR_14)`, clamped to `[-1, +1]`.
4. **Modifiers** (added then clamped): sector amplifies/dampens by sign of base; OBV-vs-price divergence ±; volume confirmation bullish (key reversal off lower band) / bearish (high-volume break of SMA-200); SMA-200 slope ±0.10; bear regime soft contribution -0.7.
5. **Bucket**: `>=+0.7` Strong Buy, `+0.4..+0.7` Mild Buy, `-0.4..+0.4` Hold, `-0.7..-0.4` Mild Sell, `<=-0.7` Strong Sell.

Every rule that fires is recorded in `reasons[]` so the UI can show *why* a signal moved.

## Conventions

- Indicators are pure functions over a pandas DataFrame; NaNs in warm-up periods are preserved.
- The engine assumes at least 220 rows of history (SMA-200 + 20-day slope).
- Tests pin each rule's contribution to the locked spec — any drift in the formula breaks the corresponding test, which is intentional. Update tests and the locked plan together if a contribution value changes.

## Plan reference

Round-1 through round-4 decisions and the locked formula live in `~/.claude/plans/federated-wiggling-frost.md`. Read it before extending the engine.
