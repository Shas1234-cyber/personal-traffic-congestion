"""
streamlit_app.py
----------------
Production-level Streamlit dashboard for Traffic Congestion Prediction.

Sections:
  🏠 Home / Live Predictor
  📊 EDA Dashboard
  🤖 Model Insights
  📈 Time-Series Forecast Simulation
  ℹ️  About
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import (
    FEAT_CSV, MODELS_DIR, REPORTS,
    BEST_MODEL, SCALER_PATH, MODEL_RF, MODEL_XGB, MODEL_LR,
    FEATURE_COLS, TARGET, CONGESTION_LABELS, CONGESTION_BINS,
    WEATHER_CATEGORIES,
)
from predict import predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* General */
  .main { background: #0E1117; }
  h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #1E2A3A, #16213E);
    border: 1px solid #2C3E50;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
  }
  .metric-label { font-size: 13px; color: #A0AEC0; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 36px; font-weight: 700; margin: 6px 0; }
  .metric-sub   { font-size: 12px; color: #718096; }

  /* Congestion badges */
  .badge-low      { background:#1a472a; color:#2ecc71; padding:6px 16px; border-radius:20px; font-weight:700; }
  .badge-moderate { background:#7d5a00; color:#f39c12; padding:6px 16px; border-radius:20px; font-weight:700; }
  .badge-high     { background:#641e16; color:#e74c3c; padding:6px 16px; border-radius:20px; font-weight:700; }
  .badge-severe   { background:#2c0e0e; color:#ff0000; padding:6px 16px; border-radius:20px; font-weight:700; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #111827; }
  .sidebar-logo { font-size: 28px; font-weight: 800; color: #60A5FA; text-align: center; padding: 10px 0; }

  /* Progress bar wrapper */
  .progress-wrap { background:#1F2937; border-radius:8px; height:14px; overflow:hidden; margin:6px 0; }
  .progress-fill { height:100%; border-radius:8px; transition: width 0.4s ease; }
</style>
""", unsafe_allow_html=True)


# ── Data / model loaders ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if FEAT_CSV.exists():
        return pd.read_csv(FEAT_CSV, parse_dates=["Timestamp"])
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for name, path in [("Random Forest", MODEL_RF), ("XGBoost", MODEL_XGB), ("Linear Regression", MODEL_LR)]:
        if path.exists():
            models[name] = joblib.load(path)
    return models

@st.cache_data(show_spinner=False)
def load_results():
    p = MODELS_DIR / "results.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}

@st.cache_data(show_spinner=False)
def load_feature_importance(name):
    path = MODELS_DIR / f"{'rf' if 'Forest' in name else 'xgb'}_feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🚦 TCP</div>', unsafe_allow_html=True)
        st.markdown("**Traffic Congestion Predictor**")
        st.caption("ML-powered real-time prediction engine")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["🏠 Live Predictor", "📊 EDA Dashboard", "🤖 Model Insights",
             "📈 Forecast Simulation", "ℹ️  About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        df = load_data()
        if not df.empty:
            st.metric("Dataset Rows",   f"{len(df):,}")
            st.metric("Date Range",     "Jan – Dec 2023")
            st.metric("Feature Count",  str(len(FEATURE_COLS)))
        results = load_results()
        if results:
            best = results.get("best", "—")
            st.metric("Best Model", best.split()[0])
            rmse = results.get("metrics", {}).get(best, {}).get("RMSE", None)
            if rmse:
                st.metric("Best RMSE", f"{rmse:.1f}")
    return page


# ── Live Predictor page ───────────────────────────────────────────────────────
def page_predictor():
    st.title("🚦 Traffic Congestion Predictor")
    st.markdown("Enter the conditions below and get an instant congestion forecast.")
    st.markdown("---")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.subheader("⚙️  Input Parameters")

        c1, c2 = st.columns(2)
        with c1:
            hour = st.slider("🕐 Hour of Day", 0, 23, datetime.now().hour, format="%d:00")
        with c2:
            dow_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow_sel   = st.selectbox("📅 Day of Week", dow_names, index=datetime.now().weekday())
            dow       = dow_names.index(dow_sel)

        c3, c4 = st.columns(2)
        with c3:
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            month_sel   = st.selectbox("📆 Month", month_names, index=datetime.now().month - 1)
            month       = month_names.index(month_sel) + 1
        with c4:
            weather = st.selectbox("🌤 Weather", WEATHER_CATEGORIES)

        events = st.toggle("🎉 Special Event / Holiday", value=False)

        st.markdown("#### 📉 Recent Traffic Context")
        st.caption("Optional – set to 0 if unknown")
        cc1, cc2 = st.columns(2)
        with cc1:
            lag_1h  = st.number_input("Traffic 1h ago",  0, 8000, 1200, 50)
            lag_24h = st.number_input("Traffic 24h ago", 0, 8000, 1200, 50)
        with cc2:
            lag_2h   = st.number_input("Traffic 2h ago",  0, 8000, 1200, 50)
            lag_168h = st.number_input("Traffic 1wk ago", 0, 8000, 1200, 50)

        predict_btn = st.button("🔮  Predict Congestion", type="primary", use_container_width=True)

    with col_r:
        st.subheader("📊 Prediction Result")

        if predict_btn or "last_pred" in st.session_state:
            if predict_btn:
                inputs = {
                    "hour": hour, "day_of_week": dow, "month": month,
                    "weather": weather, "events": events,
                    "lag_1h": lag_1h, "lag_2h": lag_2h,
                    "lag_24h": lag_24h, "lag_168h": lag_168h,
                    "rolling_mean_3h": (lag_1h + lag_2h) / 2,
                    "rolling_mean_6h": (lag_1h + lag_2h + lag_24h) / 3,
                    "rolling_mean_24h": lag_24h,
                    "rolling_std_3h": abs(lag_1h - lag_2h),
                    "traffic_trend": (lag_1h + lag_24h + lag_168h) / 3,
                }
                with st.spinner("Running model inference …"):
                    time.sleep(0.3)
                    result = predict(inputs)
                st.session_state["last_pred"] = result
            else:
                result = st.session_state["last_pred"]

            vol   = result["volume"]
            level = result["congestion_level"]
            pct   = result["congestion_pct"]
            model_name = result.get("model_name", "Best Model")

            # Colour mapping
            color_map = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c", "Severe": "#8e0000"}
            badge_map = {"Low": "badge-low", "Moderate": "badge-moderate",
                         "High": "badge-high", "Severe": "badge-severe"}
            col = color_map.get(level, "#FFFFFF")

            # Volume gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=vol,
                delta={"reference": 1200, "valueformat": ".0f"},
                title={"text": "Vehicles / Hour", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 4000], "tickwidth": 1, "tickcolor": "white"},
                    "bar":  {"color": col, "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0,    700],  "color": "#1a472a"},
                        {"range": [700,  1100], "color": "#7d5a00"},
                        {"range": [1100, 1600], "color": "#641e16"},
                        {"range": [1600, 4000], "color": "#2c0e0e"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": vol},
                },
                number={"font": {"size": 48, "color": col}},
            ))
            fig_gauge.update_layout(
                height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Badge + bar
            st.markdown(
                f'<div style="text-align:center;margin:6px 0;">'
                f'<span class="{badge_map[level]}">⚡ {level.upper()} CONGESTION</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            bar_color = col.lstrip("#")
            st.markdown(
                f'<div class="progress-wrap">'
                f'<div class="progress-fill" style="width:{pct}%;background:{col};"></div>'
                f'</div>'
                f'<div style="text-align:right;font-size:12px;color:#9CA3AF">{pct}% severity</div>',
                unsafe_allow_html=True,
            )

            # Context metrics
            st.markdown("#### 📋 Prediction Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Volume",    f"{vol:,} veh/h")
            m2.metric("Level",     level)
            m3.metric("Severity",  f"{pct}%")

            st.markdown(f"<small style='color:#6B7280'>Model: {model_name}</small>",
                        unsafe_allow_html=True)

            # Advice box
            advice = {
                "Low":      "✅ Roads are clear. Smooth journey expected.",
                "Moderate": "⚠️  Light congestion. Allow an extra 5–10 min.",
                "High":     "🔴 Heavy traffic. Consider alternate routes.",
                "Severe":   "🚨 Gridlock conditions! Avoid if possible.",
            }
            st.info(advice[level])

        else:
            st.markdown(
                "<div style='text-align:center;padding:60px 20px;color:#6B7280'>"
                "<div style='font-size:64px'>🚗</div>"
                "<p>Set the input parameters and click <b>Predict Congestion</b> to get started.</p>"
                "</div>",
                unsafe_allow_html=True,
            )


# ── EDA Dashboard page ────────────────────────────────────────────────────────
def page_eda():
    st.title("📊 Exploratory Data Analysis")
    df = load_data()
    if df.empty:
        st.error("Feature data not found. Run `python main.py` first.")
        return

    st.markdown("---")

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Records",  f"{len(df):,}")
    k2.metric("Avg Volume",     f"{df[TARGET].mean():.0f}")
    k3.metric("Peak Volume",    f"{df[TARGET].max():.0f}")
    k4.metric("Event Hours",    f"{df['Events'].sum():,}")
    k5.metric("Unique Weathers", str(df["Weather"].nunique()))
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["⏰ Time Patterns", "🌤 Weather & Events", "🗓 Calendar Heatmap", "📉 Distribution"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            hourly = df.groupby("hour")[TARGET].mean().reset_index()
            fig = px.line(hourly, x="hour", y=TARGET, markers=True,
                          title="Average Traffic by Hour",
                          color_discrete_sequence=["#60A5FA"])
            fig.update_layout(template="plotly_dark", xaxis_title="Hour", yaxis_title="Avg Vehicles/h")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            dow_df = df.groupby("day_of_week")[TARGET].mean().reset_index()
            dow_df["day"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            fig = px.bar(dow_df, x="day", y=TARGET, title="Average Traffic by Day of Week",
                         color=TARGET, color_continuous_scale="Blues")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        monthly = df.groupby("month")[TARGET].mean().reset_index()
        fig = px.area(monthly, x="month", y=TARGET,
                      title="Monthly Traffic Trend (Seasonality)",
                      color_discrete_sequence=["#34D399"])
        fig.update_layout(template="plotly_dark", xaxis=dict(tickvals=list(range(1,13)),
            ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            w_stats = df.groupby("Weather")[TARGET].mean().reset_index()
            fig = px.bar(w_stats, x="Weather", y=TARGET, title="Traffic by Weather",
                         color="Weather", color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            e_stats = df.groupby("Events")[TARGET].agg(["mean","std","count"]).reset_index()
            e_stats["label"] = e_stats["Events"].map({0:"No Event",1:"Event"})
            fig = px.bar(e_stats, x="label", y="mean", error_y="std",
                         title="Traffic: Event vs No Event",
                         color="label", color_discrete_sequence=["#60A5FA","#F87171"])
            fig.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Weather distribution violin
        fig = px.violin(df, x="Weather", y=TARGET, box=True, title="Traffic Distribution by Weather",
                        color="Weather", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        pivot = df.pivot_table(values=TARGET, index="hour", columns="day_of_week", aggfunc="mean")
        pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig = px.imshow(pivot, color_continuous_scale="YlOrRd",
                        title="Traffic Heatmap: Hour × Day of Week",
                        labels={"x":"Day","y":"Hour","color":"Avg Vol"})
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=TARGET, nbins=60, title="Traffic Volume Distribution",
                               color_discrete_sequence=["#818CF8"])
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            df["Congestion"] = pd.cut(df[TARGET], bins=CONGESTION_BINS,
                                      labels=CONGESTION_LABELS)
            cong_counts = df["Congestion"].value_counts().reindex(CONGESTION_LABELS)
            fig = px.pie(values=cong_counts.values, names=cong_counts.index,
                         title="Congestion Level Breakdown",
                         color_discrete_sequence=["#2ecc71","#f39c12","#e74c3c","#8e0000"])
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ── Model Insights page ───────────────────────────────────────────────────────
def page_model_insights():
    st.title("🤖 Model Insights")
    results = load_results()
    if not results:
        st.error("No results found. Run `python main.py` first.")
        return

    metrics = results.get("metrics", {})
    best    = results.get("best", "")

    st.markdown("---")
    st.subheader("📊 Model Performance Comparison")

    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "RMSE":  round(m["RMSE"], 2),
            "MAE":   round(m["MAE"],  2),
            "R²":    round(m["R2"],   4),
            "MAPE%": round(m["MAPE"], 2),
            "Best":  "★" if name == best else "",
        })
    df_res = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df_res.style.highlight_min(["RMSE","MAE","MAPE%"], color="#1a472a")
                             .highlight_max(["R²"], color="#1a472a"), use_container_width=True)

    # Bar charts side by side
    col1, col2, col3 = st.columns(3)
    for col_w, metric, ascending in [
        (col1, "RMSE", True), (col2, "MAE", True), (col3, "R²", False)
    ]:
        with col_w:
            vals = {n: metrics[n][metric if metric != "R²" else "R2"] for n in metrics}
            fig = go.Figure(go.Bar(
                x=list(vals.keys()), y=list(vals.values()),
                marker_color=["#34D399" if k == best else "#60A5FA" for k in vals],
            ))
            fig.update_layout(title=metric, template="plotly_dark", height=300,
                               margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.subheader("🔍 Feature Importance")
    fi_model = st.selectbox("Select model", ["Random Forest", "XGBoost"])
    fi_df = load_feature_importance(fi_model)
    if not fi_df.empty:
        fi_top = fi_df.nlargest(20, "importance")
        fig = px.bar(fi_top, x="importance", y="feature", orientation="h",
                     title=f"Top 20 Features – {fi_model}",
                     color="importance", color_continuous_scale="Blues")
        fig.update_layout(template="plotly_dark", yaxis=dict(categoryorder="total ascending"), height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance file not found. Train the model first.")


# ── Forecast Simulation page ──────────────────────────────────────────────────
def page_forecast():
    st.title("📈 Forecast Simulation")
    st.markdown("Simulate hourly traffic for the next 24 hours given a starting scenario.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        start_hour = st.slider("Start Hour", 0, 23, 6)
        dow_names  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        start_dow  = st.selectbox("Day of Week", dow_names)
        dow_idx    = dow_names.index(start_dow)
    with c2:
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        month_sel   = st.selectbox("Month", month_names)
        month       = month_names.index(month_sel) + 1
        weather     = st.selectbox("Weather condition", WEATHER_CATEGORIES)
    with c3:
        events = st.toggle("Special Event", False)
        base_vol = st.slider("Base traffic volume", 200, 4000, 1200, 50)

    run_btn = st.button("▶  Run 24-Hour Simulation", type="primary")

    if run_btn:
        hours, volumes, levels = [], [], []
        lag1, lag2, lag24, lag168 = base_vol, base_vol, base_vol, base_vol

        progress = st.progress(0, text="Simulating …")
        for i in range(24):
            h = (start_hour + i) % 24
            inputs = {
                "hour": h, "day_of_week": dow_idx, "month": month,
                "weather": weather, "events": events,
                "lag_1h": lag1, "lag_2h": lag2,
                "lag_24h": lag24, "lag_168h": lag168,
                "rolling_mean_3h":  (lag1 + lag2) / 2,
                "rolling_mean_6h":  (lag1 + lag2 + lag24) / 3,
                "rolling_mean_24h": lag24,
                "rolling_std_3h":   abs(lag1 - lag2),
                "traffic_trend":    (lag1 + lag24 + lag168) / 3,
            }
            res = predict(inputs)
            vol = res["volume"]
            hours.append(f"{h:02d}:00")
            volumes.append(vol)
            levels.append(res["congestion_level"])
            lag2, lag1 = lag1, vol
            progress.progress((i + 1) / 24, text=f"Hour {h:02d}:00 → {vol:,} veh/h")

        progress.empty()

        # Plot
        color_seq = {
            "Low": "#2ecc71", "Moderate": "#f39c12",
            "High": "#e74c3c", "Severe": "#8e0000",
        }
        bar_colors = [color_seq[l] for l in levels]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=hours, y=volumes, marker_color=bar_colors,
                             name="Predicted Volume", showlegend=False))
        fig.add_trace(go.Scatter(x=hours, y=volumes, mode="lines+markers",
                                 line=dict(color="white", width=1.5),
                                 marker=dict(size=5), name="Trend"))
        for lvl, thresh in zip(CONGESTION_LABELS, CONGESTION_BINS[1:-1]):
            fig.add_hline(y=thresh, line_dash="dot",
                          annotation_text=lvl, annotation_position="top right",
                          line_color=color_seq.get(lvl, "gray"), opacity=0.5)
        fig.update_layout(
            title=f"24-Hour Traffic Forecast | {start_dow} | {weather}{' 🎉' if events else ''}",
            xaxis_title="Hour", yaxis_title="Vehicles / Hour",
            template="plotly_dark", height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        sim_df = pd.DataFrame({"Hour": hours, "Volume": volumes, "Level": levels})
        st.dataframe(sim_df.style.applymap(
            lambda x: f"color: {color_seq.get(x,'white')}" if x in color_seq else "",
            subset=["Level"]
        ), use_container_width=True, hide_index=True)

        # Stats
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Peak Volume",    f"{max(volumes):,}")
        s2.metric("Min Volume",     f"{min(volumes):,}")
        s3.metric("Avg Volume",     f"{int(np.mean(volumes)):,}")
        s4.metric("Severe Hours",   str(levels.count("Severe")))


# ── About page ────────────────────────────────────────────────────────────────
def page_about():
    st.title("ℹ️  About This Project")
    st.markdown("""
---
## 🚦 Traffic Congestion Prediction System

A production-level Machine Learning project built for college-level showcasing with
real-world data, rigorous modelling, and a polished interactive dashboard.

### 📁 Project Structure
```
traffic_congestion_prediction/
├── data/
│   ├── raw/         — Original dataset
│   └── processed/   — Cleaned & feature-engineered data
├── models/          — Saved model artifacts (.pkl) + metrics JSON
├── src/
│   ├── config.py           — Central configuration
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── visualization.py
│   └── predict.py          — Inference module
├── app/
│   └── streamlit_app.py    — This dashboard
├── reports/figures/        — Auto-generated charts
├── main.py                 — Master pipeline runner
└── requirements.txt
```

### 🧠 Models Trained
| Model | Description |
|---|---|
| **Linear Regression** | Baseline — fast, interpretable |
| **Random Forest** | Ensemble of 300 trees, handles non-linearity |
| **XGBoost** | Gradient-boosted trees — typically best performer |

### ✨ Feature Engineering
- **Cyclical encoding** — sin/cos for hour, day-of-week, month
- **Lag features** — 1h, 2h, 24h, 1-week look-back
- **Rolling statistics** — mean & std over 3/6/24h windows
- **Traffic trend** — 7-day centred rolling mean

### 📦 Tech Stack
`pandas` · `numpy` · `scikit-learn` · `xgboost` · `streamlit` · `plotly` · `joblib`

### 🔗 Dataset
Kaggle: *Metro Interstate Traffic Volume* / custom simulated annual dataset (2023).

### 🚀 How to Run
```bash
pip install -r requirements.txt
python main.py
streamlit run app/streamlit_app.py
```
""")


# ── Router ────────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()
    if   page == "🏠 Live Predictor":     page_predictor()
    elif page == "📊 EDA Dashboard":       page_eda()
    elif page == "🤖 Model Insights":      page_model_insights()
    elif page == "📈 Forecast Simulation": page_forecast()
    elif page == "ℹ️  About":              page_about()


if __name__ == "__main__":
    main()
