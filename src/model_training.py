"""
model_training.py
-----------------
Trains Random Forest, XGBoost, and Linear Regression on the engineered
feature set.  Evaluates each with RMSE / MAE / R², selects the best
model, and persists everything to disk.

Usage:
    python src/model_training.py
"""

import sys
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model   import LinearRegression
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    log_msg = "XGBoost not installed – using GradientBoostingRegressor as substitute"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FEAT_CSV, MODELS_DIR, BEST_MODEL, SCALER_PATH,
    MODEL_RF, MODEL_XGB, MODEL_LR,
    FEATURE_COLS, TARGET,
    RF_PARAMS, XGB_PARAMS, LR_PARAMS,
    TEST_SIZE, RANDOM_SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def log_metrics(name: str, metrics: dict) -> None:
    log.info(
        f"  {name:<22}  RMSE={metrics['RMSE']:7.1f}  "
        f"MAE={metrics['MAE']:7.1f}  R²={metrics['R2']:.4f}  "
        f"MAPE={metrics['MAPE']:.2f}%"
    )


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_data(path: Path = FEAT_CSV):
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    log.info(f"  Loaded feature data: {df.shape}")

    # Ensure all expected features exist
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    return X, y, df


# ─── Scaling ──────────────────────────────────────────────────────────────────
def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ─── Model training ───────────────────────────────────────────────────────────
def train_linear_regression(X_tr, y_tr):
    log.info("  Training Linear Regression …")
    t0 = time.time()
    m = LinearRegression(**LR_PARAMS)
    m.fit(X_tr, y_tr)
    log.info(f"  Finished in {time.time()-t0:.1f}s")
    return m


def train_random_forest(X_tr, y_tr):
    log.info("  Training Random Forest …")
    t0 = time.time()
    m = RandomForestRegressor(**RF_PARAMS)
    m.fit(X_tr, y_tr)
    log.info(f"  Finished in {time.time()-t0:.1f}s")
    return m


def train_xgboost(X_tr, y_tr):
    if _XGBOOST_AVAILABLE:
        log.info("  Training XGBoost …")
        t0 = time.time()
        m = XGBRegressor(**XGB_PARAMS, verbosity=0)
        m.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)], verbose=False)
        log.info(f"  Finished in {time.time()-t0:.1f}s")
    else:
        log.info("  Training GradientBoosting (XGBoost substitute) …")
        t0 = time.time()
        m = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.85, random_state=42
        )
        m.fit(X_tr, y_tr)
        log.info(f"  Finished in {time.time()-t0:.1f}s")
    return m


# ─── Main pipeline ────────────────────────────────────────────────────────────
def run_training(feat_path: Path = FEAT_CSV) -> dict:
    log.info("═" * 50)
    log.info("STEP 3 – MODEL TRAINING & EVALUATION")
    log.info("═" * 50)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_data(feat_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=False
    )
    log.info(f"  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # Scale (LR needs it; tree models don't but scaling doesn't hurt)
    scaler   = fit_scaler(X_tr)
    X_tr_sc  = scaler.transform(X_tr)
    X_te_sc  = scaler.transform(X_te)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"  Scaler saved → {SCALER_PATH}")

    # ── Train all models ──────────────────────────────────────────────────────
    models = {}

    lr      = train_linear_regression(X_tr_sc, y_tr)
    rf      = train_random_forest(X_tr, y_tr)        # tree – no scaling needed
    xgb     = train_xgboost(X_tr, y_tr)

    models["Linear Regression"] = (lr,  X_te_sc, MODEL_LR)
    models["Random Forest"]     = (rf,  X_te,    MODEL_RF)
    models["XGBoost"]           = (xgb, X_te,    MODEL_XGB)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = {}
    log.info("\n  ── Test-set metrics ──────────────────────────────────────")
    for name, (model, X_eval, save_path) in models.items():
        y_pred  = model.predict(X_eval)
        metrics = compute_metrics(y_te, y_pred)
        log_metrics(name, metrics)
        results[name] = metrics
        joblib.dump(model, save_path)
        log.info(f"  Saved {name} → {save_path}")

    # ── Best model ────────────────────────────────────────────────────────────
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model_obj = {
        "Linear Regression": lr,
        "Random Forest":     rf,
        "XGBoost":           xgb,
    }[best_name]
    best_X = {
        "Linear Regression": X_te_sc,
        "Random Forest":     X_te,
        "XGBoost":           X_te,
    }[best_name]

    joblib.dump(best_model_obj, BEST_MODEL)
    log.info(f"\n  ★ Best model: {best_name}  (RMSE={results[best_name]['RMSE']:.1f})")
    log.info(f"  Saved best model → {BEST_MODEL}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    results_path = MODELS_DIR / "results.json"
    # Convert numpy types to native Python for JSON serialization
    def _native(obj):
        if isinstance(obj, dict):
            return {k: _native(v) for k, v in obj.items()}
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    payload = {"best": best_name, "metrics": _native(results)}
    results_path.write_text(json.dumps(payload, indent=2))
    log.info(f"  Results JSON → {results_path}")

    # ── Feature importance (tree models) ──────────────────────────────────────
    for name, model, path in [
        ("Random Forest", rf, MODELS_DIR / "rf_feature_importance.csv"),
        ("XGBoost",       xgb, MODELS_DIR / "xgb_feature_importance.csv"),
    ]:
        fi = pd.DataFrame({
            "feature":   FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi.to_csv(path, index=False)
        log.info(f"  {name} feature importances → {path}")

    log.info("\nTraining complete.")
    return results


if __name__ == "__main__":
    run_training()
