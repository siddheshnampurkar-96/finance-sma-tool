"""Synthetic price-series helpers shared across engine tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

DEFAULT_VOLUME = 1_000_000


def make_history(closes, volumes=None, high_offset=0.5, low_offset=0.5):
    """Build an OHLCV DataFrame from a list/array of close prices.

    - Open = Close - 0.1 (arbitrary, not used by the engine).
    - High = Close + high_offset.
    - Low = Close - low_offset.
    - Volume defaults to DEFAULT_VOLUME on every bar.
    - Date index is business-day backwards from today.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if volumes is None:
        volumes = np.full(n, DEFAULT_VOLUME, dtype=float)
    else:
        volumes = np.asarray(volumes, dtype=float)
    dates = pd.date_range(end=pd.Timestamp("2025-01-02"), periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": closes - 0.1,
            "High": closes + high_offset,
            "Low": closes - low_offset,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


@pytest.fixture
def flat_history():
    """260 days of price=100, volume=1M. SMA-200 ≈ 100, ATR ≈ 1, slopes ≈ 0.

    Sized for the round-5 long-trend window (60d), which dominates the
    earlier 20d slope warmup."""
    return make_history(np.full(260, 100.0))
