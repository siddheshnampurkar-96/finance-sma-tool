import numpy as np
import pandas as pd

from vibe.macro import compute_vix_high_stress


def test_vix_low_stress_when_below_mean_plus_sigma():
    vix = pd.Series(np.full(250, 15.0))
    assert compute_vix_high_stress(vix) is False


def test_vix_high_stress_when_today_spikes_above_band():
    base = np.full(250, 15.0)
    base[-1] = 40.0  # today spikes
    vix = pd.Series(base)
    assert compute_vix_high_stress(vix) is True


def test_vix_returns_false_on_insufficient_history():
    vix = pd.Series(np.full(100, 15.0))
    assert compute_vix_high_stress(vix) is False
