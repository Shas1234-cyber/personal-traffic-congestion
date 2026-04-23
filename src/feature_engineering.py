"""
feature_engineering.py
-----------------------
Transforms the clean dataset into a rich feature matrix for ML.

Features created:
  - Calendar        : hour, dow, month, day_of_year
  - Cyclical         : sin/cos encodings for hour, dow, month
  - Binary flags     : is_weekend, is_rush_hour, is_night
  - Weather OHE      : one-hot columns for each weather category
  - Lag features     : 1h, 2h, 24h, 168h (1-week) lagged traffic
  - Rolling stats    : 3h / 6h / 24h mean; 3h std
  - Trend            : 7-day centred rolling mean (traffic_trend)
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLEAN_CSV, FEAT_CSV, DATA_PROC, TARGET,
    WEATHER_CATEGORIES, FEATURE_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["Timestamp"]
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek        # 0=Monday
    df["month"]       = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode periodic features with sin/cos so 23h and 0h are neighbours."""
    df = df.copy()
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
    return df


def add_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"]     = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    return df


def add_weather_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Weather into weather_<category> columns."""
    df = df.copy()
    for cat in WEATHER_CATEGORIES:
        df[f"weather_{cat}"] = (df["Weather"] == cat).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shift the target column to create look-back features."""
    df = df.copy()
    df["lag_1h"]   = df[TARGET].shift(1)
    df["lag_2h"]   = df[TARGET].shift(2)
    df["lag_24h"]  = df[TARGET].shift(24)
    df["lag_168h"] = df[TARGET].shift(168)    # 1 week
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and std on past windows (no data leakage – min_periods)."""
    df = df.copy()
    tv = df[TARGET]
    df["rolling_mean_3h"]  = tv.shift(1).rolling(3,  min_periods=1).mean()
    df["rolling_mean_6h"]  = tv.shift(1).rolling(6,  min_periods=1).mean()
    df["rolling_mean_24h"] = tv.shift(1).rolling(24, min_periods=1).mean()
    df["rolling_std_3h"]   = tv.shift(1).rolling(3,  min_periods=1).std().fillna(0)
    return df


def add_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """7-day (168-hour) centred rolling mean as a long-horizon trend signal."""
    df = df.copy()
    df["traffic_trend"] = (
        df[TARGET]
        .rolling(window=168, center=True, min_periods=24)
        .mean()
        .fillna(df[TARGET].mean())
    )
    return df


def encode_events(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Events is an integer (0/1)."""
    df = df.copy()
    df["Events"] = df["Events"].astype(int)
    return df


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any feature or target is NaN (mostly from lag warm-up)."""
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)
    log.info(f"  Dropped {before - len(df)} NaN rows (lag warm-up).  Remaining: {len(df):,}")
    return df


def save_features(df: pd.DataFrame, path: Path = FEAT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  Saved features → {path}  ({len(df):,} rows, {df.shape[1]} cols)")


# ─────────────────────────────────────────────────────────────────────────────
def run_feature_engineering(
    clean_path: Path = CLEAN_CSV,
    out_path:   Path = FEAT_CSV,
) -> pd.DataFrame:
    log.info("═" * 50)
    log.info("STEP 2 – FEATURE ENGINEERING")
    log.info("═" * 50)

    df = pd.read_csv(clean_path, parse_dates=["Timestamp"])
    log.info(f"  Loaded clean data: {df.shape}")

    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_binary_flags(df)
    df = add_weather_ohe(df)
    df = encode_events(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_feature(df)
    df = drop_na_rows(df)

    save_features(df, out_path)
    log.info(f"Feature engineering complete.  Final feature count: {len(FEATURE_COLS)}")
    return df


if __name__ == "__main__":
    run_feature_engineering()
