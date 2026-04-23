"""
data_preprocessing.py
---------------------
Loads the raw CSV, cleans it, and saves a tidy version to
data/processed/traffic_clean.csv.

Steps:
  1. Parse timestamps
  2. Validate & coerce column types
  3. Handle missing values (forward-fill then median fallback)
  4. Remove extreme outliers via IQR clipping
  5. Save the clean frame
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path so config is importable whether run directly or via main
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_CSV, CLEAN_CSV, DATA_PROC, TARGET, WEATHER_CATEGORIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV and do minimal type coercion."""
    log.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    log.info(f"  Rows: {len(df):,}  |  Cols: {df.shape[1]}")
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the Timestamp column to datetime."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    n_bad = df["Timestamp"].isna().sum()
    if n_bad:
        log.warning(f"  {n_bad} unparseable timestamps → dropped")
        df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    log.info(f"  Date range: {df['Timestamp'].min()} → {df['Timestamp'].max()}")
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Events is boolean and Weather is a clean string."""
    df = df.copy()
    # Events: accept bool, int, or string
    if df["Events"].dtype == object:
        df["Events"] = df["Events"].map({"True": True, "False": False, "1": True, "0": False})
    df["Events"] = df["Events"].astype(bool)
    # Weather: strip whitespace, title-case
    df["Weather"] = df["Weather"].str.strip().str.title()
    unknown = ~df["Weather"].isin(WEATHER_CATEGORIES)
    if unknown.any():
        log.warning(f"  {unknown.sum()} unknown weather values → replaced with 'Clear'")
        df.loc[unknown, "Weather"] = "Clear"
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill time series gaps; fall back to column median."""
    df = df.copy()
    missing_before = df[TARGET].isna().sum()
    if missing_before:
        log.info(f"  Missing {TARGET}: {missing_before} → forward-fill then median")
        df[TARGET] = df[TARGET].fillna(method="ffill").fillna(df[TARGET].median())
    return df


def remove_outliers_iqr(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Clip Traffic Volume to [Q1 - factor*IQR, Q3 + factor*IQR]."""
    df = df.copy()
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    before = len(df)
    df[TARGET] = df[TARGET].clip(lo, hi)
    log.info(f"  Outlier clip: [{lo:.0f}, {hi:.0f}]  (factor={factor})")
    return df


def save_clean(df: pd.DataFrame, path: Path = CLEAN_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  Saved clean data → {path}  ({len(df):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
def run_preprocessing(raw_path: Path = RAW_CSV, out_path: Path = CLEAN_CSV) -> pd.DataFrame:
    """Full pipeline: load → clean → save → return."""
    log.info("═" * 50)
    log.info("STEP 1 – DATA PREPROCESSING")
    log.info("═" * 50)

    df = load_raw(raw_path)
    df = parse_timestamps(df)
    df = coerce_types(df)
    df = handle_missing(df)
    df = remove_outliers_iqr(df)
    save_clean(df, out_path)

    log.info(f"Preprocessing complete.  Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    run_preprocessing()
