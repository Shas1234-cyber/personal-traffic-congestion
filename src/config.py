"""
config.py
---------
Central configuration for the Traffic Congestion Prediction project.
All paths, hyperparameters, and constants live here so nothing is
hard-coded anywhere else.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # project root
DATA_RAW   = BASE_DIR / "data" / "raw"
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS    = BASE_DIR / "reports" / "figures"

RAW_CSV    = DATA_RAW  / "traffic_dataset_with_trend.csv"
CLEAN_CSV  = DATA_PROC / "traffic_clean.csv"
FEAT_CSV   = DATA_PROC / "traffic_features.csv"

MODEL_RF   = MODELS_DIR / "random_forest.pkl"
MODEL_XGB  = MODELS_DIR / "xgboost.pkl"
MODEL_LR   = MODELS_DIR / "linear_regression.pkl"
BEST_MODEL = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"

# ── Target & feature columns ─────────────────────────────────────────────────
TARGET = "Traffic Volume"

WEATHER_CATEGORIES = ["Clear", "Cloudy", "Rain", "Snow"]

# Numeric feature columns (after engineering)
FEATURE_COLS = [
    "hour", "day_of_week", "month", "day_of_year",
    "is_weekend", "is_rush_hour", "is_night",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "Events",
    "weather_Clear", "weather_Cloudy", "weather_Rain", "weather_Snow",
    "lag_1h", "lag_2h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_6h", "rolling_mean_24h",
    "rolling_std_3h",
    "traffic_trend",
]

# ── Model hyperparameters ────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
}

XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

LR_PARAMS = {
    "fit_intercept": True,
}

# ── Training split ───────────────────────────────────────────────────────────
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# ── Congestion thresholds (vehicles / hour) ───────────────────────────────────
CONGESTION_BINS   = [0, 700, 1100, 1600, 9999]
CONGESTION_LABELS = ["Low", "Moderate", "High", "Severe"]

# ── Misc ─────────────────────────────────────────────────────────────────────
FIGURE_DPI = 150
PLOT_STYLE  = "seaborn-v0_8-whitegrid"
