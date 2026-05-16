"""Macro context shared across per-asset signal computations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vibe.indicators import sma


@dataclass(frozen=True)
class MacroContext:
    """Macro state evaluated once per run, used by every asset engine.

    Attributes:
        vix_high_stress: True if VIX is in an elevated-stress regime.
            Triggers wider bear-regime window and (per the v2.0 spec)
            a wider buffer multiplier on the base score.
    """

    vix_high_stress: bool


def compute_vix_high_stress(vix_close: pd.Series, sigma_multiplier: float = 1.5) -> bool:
    """Decide whether today's VIX close is in the high-stress regime.

    Definition: VIX > SMA_200(VIX) + sigma_multiplier * std_200(VIX).

    Returns False if there is insufficient history (< 200 closes).
    """
    if len(vix_close) < 200:
        return False
    rolling_mean = sma(vix_close, 200)
    rolling_std = vix_close.rolling(200).std()
    today = vix_close.iloc[-1]
    upper = rolling_mean.iloc[-1] + sigma_multiplier * rolling_std.iloc[-1]
    return bool(today > upper)
