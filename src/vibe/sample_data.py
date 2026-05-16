"""Fetch the bundled real-data CSVs we use for backtesting.

The network policy in this environment only permits github-raw, so we
pin to three public datasets:

- vijinho/sp500              S&P 500 daily OHLCV (1950-01-03 to 2018-12-21)
- datasets/finance-vix       VIX daily OHLC (1990-01-02 to present)
- krishnaik06/Stock-MArket-Forecasting   AAPL daily OHLCV (2015-05-27 to 2020-05-22)

CSVs are cached under ``cache_dir`` (default: ``./.data_cache``) so
re-runs don't re-download.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

SP500_URL = "https://raw.githubusercontent.com/vijinho/sp500/master/csv/sp500.csv"
VIX_URL = "https://raw.githubusercontent.com/datasets/finance-vix/master/data/vix-daily.csv"
AAPL_URL = "https://raw.githubusercontent.com/krishnaik06/Stock-MArket-Forecasting/master/AAPL.csv"

DEFAULT_CACHE_DIR = Path(".data_cache")


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response, dest.open("wb") as out:
        out.write(response.read())
    return dest


def load_sp500(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    path = _download(SP500_URL, cache_dir / "sp500.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def load_vix(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.Series:
    path = _download(VIX_URL, cache_dir / "vix.csv")
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    df = df.set_index("DATE").sort_index()
    return df["CLOSE"].astype(float).rename("VIX")


def load_aapl(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    path = _download(AAPL_URL, cache_dir / "aapl.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    # Use adjusted columns to honour splits/dividends.
    out = pd.DataFrame(
        {
            "Open": df["adjOpen"],
            "High": df["adjHigh"],
            "Low": df["adjLow"],
            "Close": df["adjClose"],
            "Volume": df["adjVolume"],
        }
    ).astype(float)
    return out
