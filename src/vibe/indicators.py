"""Pure-function indicators used by the signal engine.

Each function takes an OHLCV DataFrame indexed by date with columns
Open/High/Low/Close/Volume and returns a Series aligned to the input
index. NaNs in warm-up periods are preserved so callers can decide how
to handle insufficient history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def atr(history: pd.DataFrame, window: int = 14) -> pd.Series:
    high = history["High"]
    low = history["Low"]
    prev_close = history["Close"].shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def obv(history: pd.DataFrame) -> pd.Series:
    close = history["Close"]
    volume = history["Volume"]
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def slope_n(series: pd.Series, n: int) -> pd.Series:
    """Discrete slope: series[t] - series[t-n]."""
    return series - series.shift(n)


def consecutive_below(series: pd.Series, threshold: pd.Series) -> int:
    """Count consecutive trailing rows where series < threshold.

    Returns 0 if the most recent row is NOT below; otherwise walks
    backwards until the condition breaks or history is exhausted.
    """
    below = (series < threshold).to_numpy()
    if not below[-1]:
        return 0
    count = 0
    for value in below[::-1]:
        if value:
            count += 1
        else:
            break
    return count
