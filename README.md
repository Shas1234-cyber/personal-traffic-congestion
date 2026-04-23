# 🚦 Traffic Congestion Prediction System

> A production-level Machine Learning project — end-to-end pipeline from raw data
> to an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red?logo=streamlit)

---

## 📌 Overview

This system predicts **hourly traffic volume** (vehicles/hour) and classifies the result
into four congestion levels:

| Level    | Volume (veh/h) | Description                   |
|----------|---------------|-------------------------------|
| 🟢 Low      | < 700         | Free-flow traffic             |
| 🟡 Moderate | 700 – 1 100   | Light congestion              |
| 🔴 High     | 1 100 – 1 600 | Heavy traffic                 |
| ⚫ Severe   | > 1 600       | Gridlock / incident likely    |

---

## 📁 Project Structure

```
traffic_congestion_prediction/
│
├── data/
│   ├── raw/                    ← Original CSV dataset
│   └── processed/              ← Cleaned + feature-engineered data
│
├── models/                     ← Saved .pkl models + results.json
│
├── src/
│   ├── config.py               ← All paths, hyper-params, constants
│   ├── data_preprocessing.py   ← Cleaning, type coercion, outlier removal
│   ├── feature_engineering.py  ← 30+ features: cyclical, lag, rolling, OHE
│   ├── model_training.py       ← RF + XGBoost + LR training & evaluation
│   ├── visualization.py        ← 14 matplotlib/seaborn charts
│   └── predict.py              ← Inference module (used by UI)
│
├── app/
│   └── streamlit_app.py        ← Full-stack interactive dashboard
│
├── reports/
│   └── figures/                ← Auto-generated PNG charts
│
├── main.py                     ← Master pipeline runner
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing (`src/data_preprocessing.py`)
- Parse & sort timestamps
- Coerce `Events` to boolean, `Weather` to clean string
- Forward-fill missing values → median fallback
- Clip outliers at 3×IQR bounds

### 2. Feature Engineering (`src/feature_engineering.py`)
| Category        | Features |
|----------------|----------|
| Calendar        | hour, day_of_week, month, day_of_year |
| Cyclical        | hour_sin/cos, dow_sin/cos, month_sin/cos |
| Binary flags    | is_weekend, is_rush_hour, is_night |
| Weather OHE     | weather_Clear/Cloudy/Rain/Snow |
| Lag features    | lag_1h, lag_2h, lag_24h, lag_168h |
| Rolling stats   | rolling_mean_3h/6h/24h, rolling_std_3h |
| Trend           | traffic_trend (7-day centred mean) |

### 3. Models & Evaluation (`src/model_training.py`)

| Model             | RMSE     | MAE      | R²     |
|-------------------|----------|----------|--------|
| Linear Regression | ~280     | ~215     | ~0.85  |
| Random Forest     | ~180     | ~135     | ~0.94  |
| **XGBoost** ★     | **~165** | **~120** | **~0.95** |

> _Actual numbers may vary; results saved to `models/results.json`_

---

## 🌐 Streamlit Dashboard

The dashboard has five pages:

| Page                 | Description                                                  |
|----------------------|--------------------------------------------------------------|
| 🏠 Live Predictor     | Real-time prediction with gauge, badge & advice              |
| 📊 EDA Dashboard      | Interactive Plotly charts (hourly, weekly, heatmap, etc.)    |
| 🤖 Model Insights     | Performance table, bar charts, feature importance            |
| 📈 Forecast Simulation| 24-hour forward simulation with animated bar chart           |
| ℹ️  About             | Project docs, structure, tech stack                          |

---

## ▶️ How to Run

### Prerequisites
- Python 3.9 or higher
- pip

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Run the full ML pipeline
```bash
python main.py
```
This will:
1. Preprocess the raw CSV
2. Engineer all features
3. Train RF + XGBoost + Linear Regression
4. Evaluate models and save the best one
5. Generate 14 EDA + model charts in `reports/figures/`

### 3 — Launch the UI
```bash
streamlit run app/streamlit_app.py
```
Open the URL shown in your terminal (default: `http://localhost:8501`).

---

## 🔥 Advanced Features

- **Cyclical feature encoding** – prevents the model treating 23:00 → 00:00 as a large jump
- **Lag features** – the most recent traffic is the strongest predictor of future traffic
- **7-day trend** – captures weekly seasonality without an explicit date index
- **Forecast simulation** – rolls the model's own predictions forward as lag inputs
- **Modular architecture** – every step is independently runnable and testable

---

## 🚀 Deployment Options

### Streamlit Cloud (Free)
1. Push the project to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point to `app/streamlit_app.py`
4. Add `requirements.txt` to root
> Note: run `python main.py` locally first, commit the `models/` folder

### Render
1. Create a new **Web Service** on [render.com](https://render.com)
2. Build command: `pip install -r requirements.txt && python main.py --skip-viz`
3. Start command: `streamlit run app/streamlit_app.py --server.port $PORT`

---

## 📊 Dataset

**Source:** Hourly traffic records from a simulated smart-city dataset (2023).
- 8,736 rows — one per hour across a full year
- Columns: `Timestamp`, `Weather`, `Events`, `Traffic Volume`
- Weather categories: Clear, Cloudy, Rain, Snow

---

## 🔮 Future Improvements

- [ ] LSTM / Transformer model for multi-step forecasting
- [ ] Real traffic API integration (TomTom / HERE)
- [ ] Geospatial heatmap with Folium/Kepler
- [ ] Model retraining pipeline with drift detection
- [ ] Docker + CI/CD deployment

---

## 👤 Author

Built as a final-year college project demonstrating end-to-end ML engineering.
