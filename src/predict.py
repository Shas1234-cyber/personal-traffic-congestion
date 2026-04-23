"""
predict.py
----------
Inference module – loads the saved best model and makes real-time
predictions from a single user input dict.

Used by both the Streamlit app and any downstream API.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_MODEL, SCALER_PATH, MODELS_DIR,
    FEATURE_COLS, CONGESTION_BINS, CONGESTION_LABELS, WEATHER_CATEGORIES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Load artefacts once at import time ────────────────────────────────────────
_model  = None
_scaler = None
_best_name = "Unknown"

def _load():
    global _model, _scaler, _best_name
    if _model is None:
        _model = joblib.load(BEST_MODEL)
        log.info(f"Loaded model from {BEST_MODEL}")
    if _scaler is None and SCALER_PATH.exists():
        _scaler = joblib.load(SCALER_PATH)
        log.info(f"Loaded scaler from {SCALER_PATH}")
    rp = MODELS_DIR / "results.json"
    if rp.exists():
        import json
        _best_name = json.loads(rp.read_text()).get("best", "Best Model")


# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(inputs: Dict[str, Any]) -> np.ndarray:
    """
    Convert a user-facing input dict into the engineered feature vector
    expected by the model.

    Required keys:
        hour       (int  0-23)
        day_of_week (int 0-6, 0=Monday)
        month      (int  1-12)
        weather    (str  "Clear" | "Cloudy" | "Rain" | "Snow")
        events     (bool)
        lag_1h     (float) – traffic volume 1 hour ago   (use 0 if unknown)
        lag_2h     (float) – traffic volume 2 hours ago  (use 0 if unknown)
        lag_24h    (float) – same hour yesterday         (use 0 if unknown)
        lag_168h   (float) – same hour last week         (use 0 if unknown)

    Optional / defaulted:
        rolling_mean_3h, rolling_mean_6h, rolling_mean_24h,
        rolling_std_3h, traffic_trend
    """
    h   = int(inputs["hour"])
    dow = int(inputs["day_of_week"])
    m   = int(inputs["month"])

    row = {
        "hour":             h,
        "day_of_week":      dow,
        "month":            m,
        "day_of_year":      int(inputs.get("day_of_year", _doy(m, h))),
        "is_weekend":       int(dow >= 5),
        "is_rush_hour":     int(h in [7, 8, 9, 17, 18, 19]),
        "is_night":         int(h >= 22 or h <= 5),
        "hour_sin":         np.sin(2 * np.pi * h   / 24),
        "hour_cos":         np.cos(2 * np.pi * h   / 24),
        "dow_sin":          np.sin(2 * np.pi * dow / 7),
        "dow_cos":          np.cos(2 * np.pi * dow / 7),
        "month_sin":        np.sin(2 * np.pi * m   / 12),
        "month_cos":        np.cos(2 * np.pi * m   / 12),
        "Events":           int(bool(inputs.get("events", False))),
    }

    # One-hot weather
    weather = str(inputs.get("weather", "Clear")).title()
    for cat in WEATHER_CATEGORIES:
        row[f"weather_{cat}"] = int(weather == cat)

    # Lag + rolling (default to reasonable average if not supplied)
    avg = float(inputs.get("lag_1h", inputs.get("avg_traffic", 1200)))
    row["lag_1h"]   = float(inputs.get("lag_1h",   avg))
    row["lag_2h"]   = float(inputs.get("lag_2h",   avg))
    row["lag_24h"]  = float(inputs.get("lag_24h",  avg))
    row["lag_168h"] = float(inputs.get("lag_168h", avg))
    row["rolling_mean_3h"]  = float(inputs.get("rolling_mean_3h",  avg))
    row["rolling_mean_6h"]  = float(inputs.get("rolling_mean_6h",  avg))
    row["rolling_mean_24h"] = float(inputs.get("rolling_mean_24h", avg))
    row["rolling_std_3h"]   = float(inputs.get("rolling_std_3h",  150))
    row["traffic_trend"]    = float(inputs.get("traffic_trend",    avg))

    vec = np.array([row[col] for col in FEATURE_COLS], dtype=np.float32).reshape(1, -1)
    return vec


def _doy(month: int, hour: int) -> int:
    """Rough day-of-year from month alone."""
    days = [0,31,59,90,120,151,181,212,243,273,304,334]
    return days[month - 1] + 15


# ─────────────────────────────────────────────────────────────────────────────
def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    End-to-end prediction.  Returns a dict with:
        volume          – predicted vehicles/hour (int)
        congestion_level – "Low" / "Moderate" / "High" / "Severe"
        congestion_pct  – normalised 0-100 severity score
        model_name      – name of the model used
    """
    _load()

    vec = build_feature_vector(inputs)

    # Apply scaler if the best model is linear (needs scaling)
    if _scaler is not None and "Linear" in _best_name:
        vec = _scaler.transform(vec)

    volume = float(_model.predict(vec)[0])
    volume = max(0, volume)   # no negative traffic

    # Map volume → congestion label
    congestion_level = CONGESTION_LABELS[-1]
    for i, (lo, hi) in enumerate(zip(CONGESTION_BINS[:-1], CONGESTION_BINS[1:])):
        if lo <= volume < hi:
            congestion_level = CONGESTION_LABELS[i]
            break

    # Normalised 0-100 congestion score
    max_vol = CONGESTION_BINS[-2]        # upper realistic cap
    congestion_pct = min(100, int(volume / max_vol * 100))

    return {
        "volume":           int(volume),
        "congestion_level": congestion_level,
        "congestion_pct":   congestion_pct,
        "model_name":       _best_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke test
    sample = {
        "hour": 8, "day_of_week": 1, "month": 6,
        "weather": "Rain", "events": True,
        "lag_1h": 1400, "lag_2h": 1300,
        "lag_24h": 1350, "lag_168h": 1380,
    }
    result = predict(sample)
    print("\nSample prediction:")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
