"""Per-asset signal engine implementing the Vibe v2.1 locked formula.

The formula and its rationale are documented in
~/.claude/plans/federated-wiggling-frost.md (round 4 lock).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from vibe.indicators import atr, consecutive_below, obv, slope_n, sma
from vibe.macro import MacroContext


@dataclass(frozen=True)
class AssetInputs:
    """Inputs the engine needs for one asset on one evaluation day.

    Attributes:
        history: OHLCV DataFrame indexed by trading date (ascending).
            Must contain at least 220 rows so SMA-200 plus a 20-day
            slope can be computed.
        sector_above_sma200: Comparator regime. For equities this is the
            stock's sector SPDR closing above its own 200-day SMA. For
            ETF holdings the caller substitutes SPY-above-SPY-SMA-200.
        days_to_earnings: Trading days until the next scheduled report,
            or None if no earnings are scheduled (e.g., for ETFs).
        days_since_earnings: Trading days since the last earnings
            release, or None if too far in the past to matter. Kept for
            inspection only — the locked spec applies no post-earnings
            override (ATR inflation provides natural dampening).
    """

    history: pd.DataFrame
    sector_above_sma200: bool
    days_to_earnings: Optional[int] = None
    days_since_earnings: Optional[int] = None


@dataclass(frozen=True)
class Reason:
    rule: str
    contribution: float
    detail: str


@dataclass(frozen=True)
class Signal:
    label: str
    score: float
    reasons: list[Reason]
    inputs: dict[str, float] = field(default_factory=dict)


class AssetEngine:
    # Magnitude contributions
    SECTOR_MOD = 0.15
    OBV_BEARISH_DIV = -0.25
    OBV_BULLISH_CONV = 0.10
    BULL_VOL_CONFIRM = 0.15
    BEAR_VOL_CONFIRM = -0.15
    SLOPE_POSITIVE = 0.10
    SLOPE_NEGATIVE = -0.10
    BEAR_REGIME = -0.70

    # VIX-adjusted parameters
    BUFFER_LOW_STRESS = 0.65
    BUFFER_HIGH_STRESS = 1.00
    BEAR_WINDOW_LOW_STRESS = 5
    BEAR_WINDOW_HIGH_STRESS = 7

    # Windows
    SLOPE_WINDOW = 20
    VOL_SMA_WINDOW = 20
    SMA_WINDOW = 200
    ATR_WINDOW = 14
    LOWER_BAND_TOUCH_LOOKBACK = 5

    # Earnings
    EARNINGS_BLACKOUT_DAYS = 3

    # Bucket thresholds
    STRONG_THRESHOLD = 0.70
    MILD_THRESHOLD = 0.40

    MIN_HISTORY = SMA_WINDOW + SLOPE_WINDOW  # 220 rows

    @classmethod
    def compute(cls, inputs: AssetInputs, macro: MacroContext) -> Signal:
        history = inputs.history
        if len(history) < cls.MIN_HISTORY:
            raise ValueError(
                f"Need at least {cls.MIN_HISTORY} rows of history "
                f"(SMA-{cls.SMA_WINDOW} + {cls.SLOPE_WINDOW}-day slope); got {len(history)}"
            )

        # Pre-earnings hard override — return immediately
        d2e = inputs.days_to_earnings
        if d2e is not None and 0 < d2e <= cls.EARNINGS_BLACKOUT_DAYS:
            return Signal(
                label="Hold",
                score=0.0,
                reasons=[
                    Reason(
                        rule="earnings_blackout",
                        contribution=0.0,
                        detail=f"{d2e} trading day(s) to earnings; technicals suppressed.",
                    )
                ],
                inputs={"days_to_earnings": float(d2e)},
            )

        # Indicators
        sma200 = sma(history["Close"], cls.SMA_WINDOW)
        atr14 = atr(history, cls.ATR_WINDOW)
        obv_series = obv(history)
        vol_sma = sma(history["Volume"], cls.VOL_SMA_WINDOW)
        sma200_slope = slope_n(sma200, cls.SLOPE_WINDOW)
        price_slope = slope_n(history["Close"], cls.SLOPE_WINDOW)
        obv_slope_norm = slope_n(obv_series, cls.SLOPE_WINDOW) / vol_sma

        # VIX-adjusted parameters
        k = cls.BUFFER_HIGH_STRESS if macro.vix_high_stress else cls.BUFFER_LOW_STRESS
        n_bear = cls.BEAR_WINDOW_HIGH_STRESS if macro.vix_high_stress else cls.BEAR_WINDOW_LOW_STRESS

        # Today's snapshot
        close = float(history["Close"].iloc[-1])
        prev_close = float(history["Close"].iloc[-2])
        prev_high = float(history["High"].iloc[-2])
        volume_today = float(history["Volume"].iloc[-1])
        sma200_today = float(sma200.iloc[-1])
        atr_today = float(atr14.iloc[-1])
        vol_sma_today = float(vol_sma.iloc[-1])
        sma200_slope_today = float(sma200_slope.iloc[-1])
        price_slope_today = float(price_slope.iloc[-1])
        obv_slope_norm_today = float(obv_slope_norm.iloc[-1])

        # Bands
        lower_band = sma200 - k * atr14
        lower_band_today = float(lower_band.iloc[-1])

        # Base score: distance from SMA in (k * ATR) units
        if atr_today <= 0:
            base = 0.0
        else:
            base = (close - sma200_today) / (k * atr_today)
            base = float(np.clip(base, -1.0, 1.0))

        score = base
        reasons: list[Reason] = [
            Reason(
                rule="base",
                contribution=base,
                detail=(
                    f"Close {close:.2f} vs SMA-200 {sma200_today:.2f} "
                    f"(buffer k={k}, ATR={atr_today:.2f})."
                ),
            )
        ]

        # Sector / comparator
        base_sign = np.sign(base)
        if base_sign != 0:
            sector_mod = (cls.SECTOR_MOD if inputs.sector_above_sma200 else -cls.SECTOR_MOD) * base_sign
            score += sector_mod
            reasons.append(
                Reason(
                    rule="sector_health",
                    contribution=float(sector_mod),
                    detail=(
                        "Sector above SMA-200 (healthy)"
                        if inputs.sector_above_sma200
                        else "Sector below SMA-200 (broken)"
                    ),
                )
            )

        # OBV-vs-price divergence
        if price_slope_today > 0 and obv_slope_norm_today < 0:
            score += cls.OBV_BEARISH_DIV
            reasons.append(
                Reason(
                    rule="obv_bearish_divergence",
                    contribution=cls.OBV_BEARISH_DIV,
                    detail="Price up, OBV down — possible stealth distribution.",
                )
            )
        elif price_slope_today < 0 and obv_slope_norm_today > 0:
            score += cls.OBV_BULLISH_CONV
            reasons.append(
                Reason(
                    rule="obv_bullish_convergence",
                    contribution=cls.OBV_BULLISH_CONV,
                    detail="Price down, OBV up — possible accumulation.",
                )
            )

        # Volume confirmation — bullish (reversal off support on volume)
        recent_low = float(history["Low"].iloc[-cls.LOWER_BAND_TOUCH_LOOKBACK:].min())
        touched_lower_band = recent_low <= lower_band_today
        if (
            volume_today > vol_sma_today
            and close > prev_high
            and touched_lower_band
        ):
            score += cls.BULL_VOL_CONFIRM
            reasons.append(
                Reason(
                    rule="bullish_volume_confirm",
                    contribution=cls.BULL_VOL_CONFIRM,
                    detail="High-volume key reversal at lower-band support.",
                )
            )

        # Volume confirmation — bearish (major-trend break on volume)
        if (
            volume_today > vol_sma_today
            and close < sma200_today
            and prev_close >= float(sma200.iloc[-2])
        ):
            score += cls.BEAR_VOL_CONFIRM
            reasons.append(
                Reason(
                    rule="bearish_volume_confirm",
                    contribution=cls.BEAR_VOL_CONFIRM,
                    detail="High-volume close below SMA-200 (major trend break).",
                )
            )

        # SMA-200 slope
        if sma200_slope_today > 0:
            score += cls.SLOPE_POSITIVE
            reasons.append(
                Reason(
                    rule="sma200_slope_positive",
                    contribution=cls.SLOPE_POSITIVE,
                    detail="SMA-200 rising over last 20 days.",
                )
            )
        elif sma200_slope_today < 0:
            score += cls.SLOPE_NEGATIVE
            reasons.append(
                Reason(
                    rule="sma200_slope_negative",
                    contribution=cls.SLOPE_NEGATIVE,
                    detail="SMA-200 falling over last 20 days.",
                )
            )

        # Bear regime (soft contribution per round-4 lock)
        consec_below = consecutive_below(history["Close"], lower_band)
        if consec_below >= n_bear:
            score += cls.BEAR_REGIME
            reasons.append(
                Reason(
                    rule="bear_regime",
                    contribution=cls.BEAR_REGIME,
                    detail=(
                        f"{consec_below} consecutive closes below lower band "
                        f"(threshold {n_bear}, VIX stress={macro.vix_high_stress})."
                    ),
                )
            )

        # Clamp and bucket
        score = float(np.clip(score, -1.0, 1.0))
        label = cls._bucket(score)

        return Signal(
            label=label,
            score=score,
            reasons=reasons,
            inputs={
                "close": close,
                "sma_200": sma200_today,
                "atr_14": atr_today,
                "k_buffer": k,
                "n_bear": float(n_bear),
                "lower_band": lower_band_today,
                "vol_today": volume_today,
                "vol_sma_20": vol_sma_today,
                "sma200_slope_20": sma200_slope_today,
                "price_slope_20": price_slope_today,
                "obv_slope_20_norm": obv_slope_norm_today,
                "consec_below_lower_band": float(consec_below),
                "vix_high_stress": float(macro.vix_high_stress),
                "sector_above_sma200": float(inputs.sector_above_sma200),
            },
        )

    @classmethod
    def _bucket(cls, score: float) -> str:
        if score >= cls.STRONG_THRESHOLD:
            return "Strong Buy"
        if score >= cls.MILD_THRESHOLD:
            return "Mild Buy"
        if score <= -cls.STRONG_THRESHOLD:
            return "Strong Sell"
        if score <= -cls.MILD_THRESHOLD:
            return "Mild Sell"
        return "Hold"
