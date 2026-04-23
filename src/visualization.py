"""
visualization.py
----------------
Generates all EDA and model-evaluation charts and saves them to
reports/figures/.

Charts produced:
  01_traffic_distribution.png
  02_hourly_pattern.png
  03_weekly_pattern.png
  04_monthly_pattern.png
  05_weather_impact.png
  06_events_impact.png
  07_heatmap_hour_dow.png
  08_time_series_sample.png
  09_model_comparison.png
  10_feature_importance_rf.png
  11_feature_importance_xgb.png
  12_actual_vs_predicted.png
  13_residuals.png
  14_congestion_distribution.png

Usage:
    python src/visualization.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless backend – safe in all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FEAT_CSV, CLEAN_CSV, MODELS_DIR, REPORTS,
    MODEL_RF, MODEL_XGB, MODEL_LR, BEST_MODEL, SCALER_PATH,
    FEATURE_COLS, TARGET, CONGESTION_BINS, CONGESTION_LABELS,
    FIGURE_DPI, PLOT_STYLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.style.use(PLOT_STYLE)
PALETTE   = "viridis"
ACCENT    = "#2C7BB6"
HIGHLIGHT = "#D7191C"
BG        = "#F8F9FA"

def _save(fig: plt.Figure, name: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info(f"  Saved → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("Traffic Volume Distribution", fontsize=15, fontweight="bold")

    axes[0].hist(df[TARGET], bins=60, color=ACCENT, edgecolor="white", linewidth=0.4)
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Vehicles / Hour")
    axes[0].set_ylabel("Frequency")

    sns.boxplot(y=df[TARGET], ax=axes[1], color=ACCENT)
    axes[1].set_title("Box-plot")
    axes[1].set_ylabel("Vehicles / Hour")

    _save(fig, "01_traffic_distribution.png")


def plot_hourly_pattern(df: pd.DataFrame) -> None:
    hourly = df.groupby("hour")[TARGET].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.plot(hourly["hour"], hourly["mean"], color=ACCENT, linewidth=2.5, label="Mean")
    ax.fill_between(
        hourly["hour"],
        hourly["mean"] - hourly["std"],
        hourly["mean"] + hourly["std"],
        alpha=0.25, color=ACCENT, label="±1 Std Dev",
    )
    for rh in [7, 8, 9, 17, 18, 19]:
        ax.axvline(rh, color=HIGHLIGHT, linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_title("Average Traffic by Hour of Day", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Traffic Volume")
    ax.set_xticks(range(0, 24))
    ax.legend()
    _save(fig, "02_hourly_pattern.png")


def plot_weekly_pattern(df: pd.DataFrame) -> None:
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = df.groupby("day_of_week")[TARGET].mean().reset_index()
    weekly.columns = ["day_of_week", "mean"]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    colors = [HIGHLIGHT if i >= 5 else ACCENT for i in weekly["day_of_week"]]
    ax.bar(weekly["day_of_week"], weekly["mean"], color=colors, edgecolor="white")
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_labels)
    ax.set_title("Average Traffic by Day of Week", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Traffic Volume")
    _save(fig, "03_weekly_pattern.png")


def plot_monthly_pattern(df: pd.DataFrame) -> None:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df.groupby("month")[TARGET].mean().reset_index()
    monthly.columns = ["month", "mean"]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.plot(monthly["month"], monthly["mean"], marker="o", color=ACCENT, linewidth=2.5)
    ax.fill_between(monthly["month"], monthly["mean"], alpha=0.15, color=ACCENT)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_title("Average Traffic by Month", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Traffic Volume")
    _save(fig, "04_monthly_pattern.png")


def plot_weather_impact(df: pd.DataFrame) -> None:
    weather_stats = df.groupby("Weather")[TARGET].agg(["mean","median","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    x = range(len(weather_stats))
    bars = ax.bar(x, weather_stats["mean"], color=sns.color_palette("Set2", len(weather_stats)),
                  edgecolor="white", zorder=2)
    ax.errorbar(x, weather_stats["mean"], yerr=weather_stats["std"],
                fmt="none", color="black", capsize=5, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(weather_stats["Weather"])
    ax.set_title("Traffic Volume by Weather Condition", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Traffic Volume")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f"{bar.get_height():.0f}", ha="center", fontsize=9)
    _save(fig, "05_weather_impact.png")


def plot_events_impact(df: pd.DataFrame) -> None:
    event_stats = df.groupby("Events")[TARGET].agg(["mean","std","count"]).reset_index()
    labels = ["No Event", "Event"]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    colors = [ACCENT, HIGHLIGHT]
    bars = ax.bar(labels, event_stats["mean"], color=colors, edgecolor="white", width=0.4)
    ax.errorbar(labels, event_stats["mean"], yerr=event_stats["std"],
                fmt="none", color="black", capsize=8)
    ax.set_title("Traffic Volume: Events vs No Events", fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Traffic Volume")
    for bar, row in zip(bars, event_stats.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f"n={row.count:,}", ha="center", fontsize=9)
    _save(fig, "06_events_impact.png")


def plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(values=TARGET, index="hour", columns="day_of_week", aggfunc="mean")
    pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                annot=False, ax=ax, cbar_kws={"label": "Avg Vehicles/Hour"})
    ax.set_title("Traffic Heatmap: Hour × Day of Week", fontsize=13, fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    _save(fig, "07_heatmap_hour_dow.png")


def plot_time_series_sample(df: pd.DataFrame) -> None:
    """Plot two continuous weeks from a representative period."""
    sample = df[(df["month"] == 3)].head(24 * 14)
    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    ax.plot(sample["Timestamp"], sample[TARGET], color=ACCENT, linewidth=1.2)
    ax.set_title("Traffic Volume – 2-Week Sample (March)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Vehicles / Hour")
    fig.autofmt_xdate()
    _save(fig, "08_time_series_sample.png")


def plot_model_comparison(results: dict) -> None:
    """Bar chart of RMSE / MAE / R² for all models."""
    names   = list(results.keys())
    rmses   = [v["RMSE"] for v in results.values()]
    maes    = [v["MAE"]  for v in results.values()]
    r2s     = [v["R2"]   for v in results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

    for ax, vals, title, color in zip(
        axes,
        [rmses, maes, r2s],
        ["RMSE (lower ✓)", "MAE (lower ✓)", "R² (higher ✓)"],
        [ACCENT, "#2CA25F", HIGHLIGHT],
    ):
        bars = ax.bar(names, vals, color=color, edgecolor="white")
        ax.set_title(title)
        ax.set_xticklabels(names, rotation=12, ha="right")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    _save(fig, "09_model_comparison.png")


def plot_feature_importance(model_path: Path, name: str, fname: str) -> None:
    try:
        model = joblib.load(model_path)
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).nlargest(20)
        fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
        fi.sort_values().plot.barh(ax=ax, color=ACCENT, edgecolor="white")
        ax.set_title(f"Top 20 Feature Importances – {name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        _save(fig, fname)
    except Exception as e:
        log.warning(f"  Could not plot feature importance for {name}: {e}")


def plot_actual_vs_predicted(y_true, y_pred, name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=ACCENT)
    lim = (min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()))
    ax.plot(lim, lim, color=HIGHLIGHT, linewidth=1.5, linestyle="--", label="Perfect fit")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual Traffic Volume")
    ax.set_ylabel("Predicted Traffic Volume")
    ax.set_title(f"Actual vs Predicted – {name}", fontsize=13, fontweight="bold")
    ax.legend()
    _save(fig, "12_actual_vs_predicted.png")


def plot_residuals(y_true, y_pred, name: str) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color=ACCENT)
    axes[0].axhline(0, color=HIGHLIGHT, linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual vs Predicted")
    axes[1].hist(residuals, bins=60, color=ACCENT, edgecolor="white")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    fig.suptitle(f"Residual Analysis – {name}", fontsize=13, fontweight="bold")
    _save(fig, "13_residuals.png")


def plot_congestion_distribution(df: pd.DataFrame) -> None:
    df = df.copy()
    df["Congestion Level"] = pd.cut(
        df[TARGET], bins=CONGESTION_BINS, labels=CONGESTION_LABELS
    )
    counts = df["Congestion Level"].value_counts().reindex(CONGESTION_LABELS)
    colors = ["#2CA25F", "#FEC44F", "#F03B20", "#7B0D1E"]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    bars = ax.bar(CONGESTION_LABELS, counts.values, color=colors, edgecolor="white")
    ax.set_title("Congestion Level Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Hours Count")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{bar.get_height():,}", ha="center", fontsize=10)
    _save(fig, "14_congestion_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
def run_visualization(results: dict = None) -> None:
    log.info("═" * 50)
    log.info("STEP 4 – VISUALIZATION")
    log.info("═" * 50)

    df = pd.read_csv(FEAT_CSV, parse_dates=["Timestamp"])
    log.info(f"  Loaded feature data: {df.shape}")

    plot_traffic_distribution(df)
    plot_hourly_pattern(df)
    plot_weekly_pattern(df)
    plot_monthly_pattern(df)
    plot_weather_impact(df)
    plot_events_impact(df)
    plot_heatmap(df)
    plot_time_series_sample(df)
    plot_congestion_distribution(df)

    # Model comparison (needs results dict)
    if results is None:
        results_path = MODELS_DIR / "results.json"
        if results_path.exists():
            import json
            payload = json.loads(results_path.read_text())
            results = payload.get("metrics", {})

    if results:
        plot_model_comparison(results)

    # Feature importances
    plot_feature_importance(MODEL_RF,  "Random Forest", "10_feature_importance_rf.png")
    plot_feature_importance(MODEL_XGB, "XGBoost",       "11_feature_importance_xgb.png")

    # Actual vs predicted + residuals for best model
    try:
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_SEED
        X = df[FEATURE_COLS].values.astype("float32")
        y = df[TARGET].values.astype("float32")
        _, X_te, _, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                            random_state=RANDOM_SEED, shuffle=False)
        best = joblib.load(BEST_MODEL)
        import json
        bname = json.loads((MODELS_DIR/"results.json").read_text()).get("best","Best")
        if "Linear" in bname:
            scaler = joblib.load(SCALER_PATH)
            X_te   = scaler.transform(X_te)
        y_pred = best.predict(X_te)
        plot_actual_vs_predicted(y_te, y_pred, bname)
        plot_residuals(y_te, y_pred, bname)
    except Exception as e:
        log.warning(f"  Skipped actual-vs-predicted: {e}")

    log.info(f"\nAll figures saved to {REPORTS}")


if __name__ == "__main__":
    run_visualization()
