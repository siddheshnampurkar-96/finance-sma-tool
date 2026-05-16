import numpy as np
import pandas as pd

from vibe.indicators import atr, consecutive_below, obv, slope_n, sma
from tests.conftest import make_history


def test_sma_matches_rolling_mean():
    closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(closes, 3)
    assert result.iloc[-1] == 4.0  # mean of 3, 4, 5
    assert np.isnan(result.iloc[0])  # warm-up


def test_atr_window_handles_initial_warmup():
    history = make_history(np.linspace(100, 110, 30))
    result = atr(history, window=14)
    # First 14 values are warm-up; index 13 onward are defined.
    assert np.isnan(result.iloc[12])
    assert not np.isnan(result.iloc[14])


def test_obv_signs_with_price_direction():
    # Closes go up, down, up, up — OBV should be: +V, +V-V, +V-V+V, +V-V+V+V
    closes = [10.0, 11.0, 10.5, 11.5, 12.0]
    volumes = [100, 100, 100, 100, 100]
    history = make_history(closes, volumes=volumes)
    result = obv(history)
    # First diff is NaN→0 (sign=0); subsequent diffs: +, -, +, +
    expected_signs = [0, 1, -1, 1, 1]
    cumulative = np.cumsum(np.array(expected_signs) * np.array(volumes))
    np.testing.assert_array_equal(result.to_numpy(), cumulative)


def test_slope_n_is_difference():
    s = pd.Series([1, 2, 4, 7, 11], dtype=float)
    result = slope_n(s, 2)
    # result[-1] = 11 - 4 = 7
    assert result.iloc[-1] == 7.0


def test_consecutive_below_counts_trailing_run():
    series = pd.Series([10.0, 11.0, 9.0, 8.0, 7.0])
    threshold = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    assert consecutive_below(series, threshold) == 3


def test_consecutive_below_zero_when_today_above():
    series = pd.Series([10.0, 9.0, 9.0, 11.0])
    threshold = pd.Series([10.0, 10.0, 10.0, 10.0])
    assert consecutive_below(series, threshold) == 0
