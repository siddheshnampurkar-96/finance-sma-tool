import numpy as np
import pandas as pd
import pytest

from vibe import AssetEngine, AssetInputs, MacroContext
from tests.conftest import make_history


# ---------- helpers --------------------------------------------------------


def reasons_by_rule(signal):
    return {r.rule: r for r in signal.reasons}


def calm_macro():
    return MacroContext(vix_high_stress=False)


def stressed_macro():
    return MacroContext(vix_high_stress=True)


# ---------- pre-earnings hard override -------------------------------------


def test_pre_earnings_forces_hold_regardless_of_setup():
    """Per round-4 lock: pre-earnings is a hard override to Hold, score=0.

    This test would fail if anyone reintroduces a soft-suppress variant
    or removes the override entirely.
    """
    closes = np.linspace(50, 150, 220)  # screaming uptrend
    history = make_history(closes)
    inputs = AssetInputs(
        history=history,
        sector_above_sma200=True,
        days_to_earnings=2,
    )
    signal = AssetEngine.compute(inputs, calm_macro())
    assert signal.label == "Hold"
    assert signal.score == 0.0
    assert "earnings_blackout" in reasons_by_rule(signal)


def test_post_earnings_does_not_override():
    """Per round-4 lock: NO post-earnings override; ATR inflation handles it.

    A symmetric hard-override would set score to 0 for days_since_earnings <= 3
    and break this test.
    """
    closes = np.linspace(50, 150, 220)
    history = make_history(closes)
    inputs = AssetInputs(
        history=history,
        sector_above_sma200=True,
        days_since_earnings=1,
    )
    signal = AssetEngine.compute(inputs, calm_macro())
    assert signal.label == "Strong Buy"


# ---------- bucket landings (integration) ----------------------------------


def test_strong_buy_on_steady_uptrend():
    closes = np.linspace(50, 150, 220)
    history = make_history(closes)
    inputs = AssetInputs(history=history, sector_above_sma200=True)
    signal = AssetEngine.compute(inputs, calm_macro())
    assert signal.label == "Strong Buy"
    assert signal.score >= AssetEngine.STRONG_THRESHOLD


def test_strong_sell_on_breakdown_after_long_flat():
    """200 days flat then a 20-day crash should fire bear regime.

    Bear regime contributes -0.7 (round-4 soft, not a floor); combined
    with base at -1.0 and negative slope it should land Strong Sell.
    """
    closes = np.concatenate(
        [np.full(200, 100.0), np.linspace(100.0, 50.0, 20, endpoint=False)]
    )
    history = make_history(closes)
    inputs = AssetInputs(history=history, sector_above_sma200=True)
    signal = AssetEngine.compute(inputs, calm_macro())
    assert signal.label == "Strong Sell"
    assert "bear_regime" in reasons_by_rule(signal)


def test_hold_on_flat_history(flat_history):
    inputs = AssetInputs(history=flat_history, sector_above_sma200=True)
    signal = AssetEngine.compute(inputs, calm_macro())
    assert signal.label == "Hold"
    assert abs(signal.score) < AssetEngine.MILD_THRESHOLD


# ---------- individual rules --------------------------------------------------


def test_base_score_clamps_to_one_for_far_above_sma():
    closes = np.concatenate([np.full(200, 100.0), np.full(20, 200.0)])
    history = make_history(closes)
    inputs = AssetInputs(history=history, sector_above_sma200=True)
    signal = AssetEngine.compute(inputs, calm_macro())
    assert reasons_by_rule(signal)["base"].contribution == pytest.approx(1.0)


def test_sector_amplifies_when_aligned_with_base_direction():
    """Healthy sector + bullish base → modifier is +0.15.

    Healthy sector + bearish base → modifier is -0.15 (idiosyncratic
    failure amplification per A3 in the critique).
    """
    closes = np.concatenate([np.full(200, 100.0), np.full(20, 102.0)])
    history = make_history(closes)
    bullish = AssetEngine.compute(
        AssetInputs(history=history, sector_above_sma200=True), calm_macro()
    )
    assert reasons_by_rule(bullish)["sector_health"].contribution == pytest.approx(0.15)

    closes_down = np.concatenate([np.full(200, 100.0), np.full(20, 98.0)])
    bearish_with_healthy_sector = AssetEngine.compute(
        AssetInputs(history=make_history(closes_down), sector_above_sma200=True),
        calm_macro(),
    )
    assert reasons_by_rule(bearish_with_healthy_sector)["sector_health"].contribution == pytest.approx(-0.15)


def test_sector_dampens_when_sector_broken():
    closes_down = np.concatenate([np.full(200, 100.0), np.full(20, 98.0)])
    broken = AssetEngine.compute(
        AssetInputs(history=make_history(closes_down), sector_above_sma200=False),
        calm_macro(),
    )
    # broken sector + bearish base → modifier is +0.15 (dampens bearish)
    assert reasons_by_rule(broken)["sector_health"].contribution == pytest.approx(0.15)


def test_obv_bearish_divergence_subtracts_quarter_point():
    """Price climbs while OBV falls = bearish divergence per the spec.

    Construct: prices drift up by 0.1/day, but volume tags ALL down days
    in the first half. Net OBV slope over 20 days is negative while
    price slope is positive.
    """
    n = 220
    closes = np.full(n, 100.0)
    closes[-21:] = np.linspace(100.0, 102.0, 21)  # mild uptrend over last 20 days
    # Build OBV negative by faking heavy down-volume days within the window
    volumes = np.full(n, 1_000_000.0)
    # Force every other recent day to be a small down-day with huge volume
    for i in range(n - 20, n - 1, 2):
        closes[i] = closes[i - 1] - 0.05  # tiny down dip
        volumes[i] = 50_000_000  # huge volume on the down day
    history = make_history(closes, volumes=volumes)
    inputs = AssetInputs(history=history, sector_above_sma200=True)
    signal = AssetEngine.compute(inputs, calm_macro())
    assert "obv_bearish_divergence" in reasons_by_rule(signal)
    assert reasons_by_rule(signal)["obv_bearish_divergence"].contribution == pytest.approx(-0.25)


def test_bear_regime_fires_at_n5_in_calm_market():
    """5 consecutive closes below the lower band triggers bear regime in calm VIX."""
    # 200 flat days at 100, then 6 days well below the lower band (~99.35).
    closes = np.concatenate([np.full(200, 100.0), np.full(20, 95.0)])
    history = make_history(closes)
    inputs = AssetInputs(history=history, sector_above_sma200=True)
    signal = AssetEngine.compute(inputs, calm_macro())
    bear = reasons_by_rule(signal).get("bear_regime")
    assert bear is not None
    assert bear.contribution == pytest.approx(AssetEngine.BEAR_REGIME)


def test_vix_high_stress_widens_bear_regime_window_to_seven():
    """6 consecutive closes below band: fires in calm (N=5), NOT in stress (N=7).

    This is the key behavioral difference of the high-VIX regime per
    the round-4 lock (along with the wider buffer multiplier k).
    """
    # 200 flat + 6 days at 95 (well below band)
    closes = np.concatenate([np.full(200, 100.0), np.full(6, 95.0)])
    # Pad to MIN_HISTORY=220
    closes = np.concatenate([closes, np.full(220 - len(closes), 95.0)])
    # We want EXACTLY ~6 consecutive trailing days below band. Reset earlier closes to 100.
    closes = np.concatenate([np.full(214, 100.0), np.full(6, 95.0)])
    history = make_history(closes)
    inputs = AssetInputs(history=history, sector_above_sma200=True)

    calm = AssetEngine.compute(inputs, calm_macro())
    stressed = AssetEngine.compute(inputs, stressed_macro())

    # In calm: 6 >= N_bear=5, bear regime fires.
    assert "bear_regime" in reasons_by_rule(calm)
    # In stress: N_bear=7, AND k widens to 1.0 making the band lower
    # (sma200 - 1.0*ATR vs sma200 - 0.65*ATR), so fewer days are below.
    # Either via the count threshold OR the wider band, bear regime must NOT fire.
    assert "bear_regime" not in reasons_by_rule(stressed)


def test_vix_high_stress_widens_buffer_multiplier():
    """k widens from 0.65 to 1.0 in high stress, so the same close
    produces a smaller base magnitude."""
    closes = np.concatenate([np.full(200, 100.0), np.full(20, 100.5)])
    history = make_history(closes)
    inputs = AssetInputs(history=history, sector_above_sma200=True)
    calm_base = reasons_by_rule(AssetEngine.compute(inputs, calm_macro()))["base"].contribution
    stress_base = reasons_by_rule(AssetEngine.compute(inputs, stressed_macro()))["base"].contribution
    # Same scenario, smaller base in stress (or equal if both clamped).
    assert stress_base <= calm_base


def test_sma_slope_positive_contributes_plus_ten():
    closes = np.linspace(80, 120, 220)
    history = make_history(closes)
    inputs = AssetInputs(history=history, sector_above_sma200=True)
    signal = AssetEngine.compute(inputs, calm_macro())
    slope_pos = reasons_by_rule(signal).get("sma200_slope_positive")
    assert slope_pos is not None
    assert slope_pos.contribution == pytest.approx(0.10)


def test_sma_slope_negative_contributes_minus_ten():
    closes = np.linspace(120, 80, 220)
    history = make_history(closes)
    inputs = AssetInputs(history=history, sector_above_sma200=False)
    signal = AssetEngine.compute(inputs, calm_macro())
    slope_neg = reasons_by_rule(signal).get("sma200_slope_negative")
    assert slope_neg is not None
    assert slope_neg.contribution == pytest.approx(-0.10)


# ---------- validation -----------------------------------------------------


def test_insufficient_history_raises():
    history = make_history(np.full(100, 100.0))
    with pytest.raises(ValueError, match="at least 220"):
        AssetEngine.compute(
            AssetInputs(history=history, sector_above_sma200=True),
            calm_macro(),
        )


def test_score_always_clamped_to_unit_range():
    """Even with every bearish rule firing the score should not exceed -1.0."""
    closes = np.concatenate([np.full(200, 100.0), np.linspace(100.0, 30.0, 20, endpoint=False)])
    history = make_history(closes)
    inputs = AssetInputs(history=history, sector_above_sma200=True)
    signal = AssetEngine.compute(inputs, calm_macro())
    assert -1.0 <= signal.score <= 1.0
