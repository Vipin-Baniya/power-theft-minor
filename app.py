"""
app.py - Streamlit dashboard for Power Theft Detection (prototype)
Features:
 - Upload meter CSV or use sample data
 - Forecast expected load per meter (Prophet if installed, otherwise simple rolling mean)
 - Anomaly timeline visualization
 - Auto-refresh dashboard (every 5s) for demo (uses st.experimental_rerun with timer)
 - GIS Map view with folium (if available)
 - Multi-model comparison toggle (RandomForest, LogisticRegression, SVC)
 - SHAP explanation (if SHAP is installed)
 - Severity scoring and monthly bill estimator
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Grid Theft Detection", layout="wide")

ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, "data", "meters_data.csv")
MODEL_DIR = os.path.join(ROOT, "models")

st.title("⚡ AI-Based Power Theft Detection — Dashboard (Prototype)")

# Load sample data
@st.cache_data(ttl=600)
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        st.error("Sample data not found. Run src/generate_data.py to create data/meters_data.csv")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=['Timestamp'])
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload CSV (Meter_ID,Timestamp,Consumption_kWh,Latitude,Longitude)", type=['csv'])
use_sample = st.sidebar.checkbox("Use sample data", value=True)
refresh = st.sidebar.checkbox("Auto-refresh every 5 seconds (demo)", value=False)
model_choice = st.sidebar.selectbox("Choose classifier for batch prediction", ["RandomForest", "LogisticRegression", "SVC"])
show_map = st.sidebar.checkbox("Show GIS Map (requires streamlit-folium & folium)", value=False)

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['Timestamp'])

if df.empty:
    st.warning("No data available")
    st.stop()

st.sidebar.markdown("### Quick stats")
st.sidebar.metric("Meters", df['Meter_ID'].nunique())
st.sidebar.metric("Total Rows", len(df))

# Feature engineering (similar to train script)
def create_features(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    grouped = df.groupby('Meter_ID')
    features = []
    for meter_id, group in grouped:
        meter_data = {}
        meter_data['Meter_ID'] = meter_id
        meter_data['avg_consumption'] = group['Consumption_kWh'].mean()
        meter_data['std_consumption'] = group['Consumption_kWh'].std()
        meter_data['max_consumption'] = group['Consumption_kWh'].max()
        meter_data['min_consumption'] = group['Consumption_kWh'].min()
        meter_data['zero_consumption_hours'] = (group['Consumption_kWh'] == 0).sum()
        meter_data['low_consumption_hours'] = (group['Consumption_kWh'] < 0.1).sum()
        night_consumption = group[group['hour'].between(0,5)]['Consumption_kWh'].mean()
        day_consumption = group[group['hour'].between(8,20)]['Consumption_kWh'].mean()
        meter_data['night_day_ratio'] = (night_consumption / day_consumption) if day_consumption>0 else 0
        hourly_avg = group.groupby('hour')['Consumption_kWh'].mean()
        meter_data['hourly_profile_std'] = hourly_avg.std()
        weekday_avg = group[group['day_of_week']<5]['Consumption_kWh'].mean()
        weekend_avg = group[group['day_of_week']>=5]['Consumption_kWh'].mean()
        meter_data['weekend_weekday_ratio'] = (weekend_avg/weekday_avg) if weekday_avg>0 else 0
        # add simple signature stats
        meter_data['skewness'] = group['Consumption_kWh'].skew()
        meter_data['kurtosis'] = group['Consumption_kWh'].kurtosis()
        features.append(meter_data)
    feat_df = pd.DataFrame(features).set_index('Meter_ID').fillna(0)
    return feat_df

features = create_features(df)

# Load model artifacts if present
model_artifacts_exist = os.path.exists(os.path.join(MODEL_DIR, "power_theft_model.joblib"))
if model_artifacts_exist:
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "power_theft_model.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
        st.sidebar.success("Model artifacts loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading model artifacts: {e}")
        model = None
        scaler = None
        feat_cols = None
else:
    model = None
    scaler = None
    feat_cols = None
    st.sidebar.info("No trained model found. Run src/train_model.py to produce model artifacts.")

# Batch prediction using chosen model (if model artifacts exist)
results = None
if model is not None and scaler is not None:
    X = features.copy()
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    y_prob = model.predict_proba(Xs)[:,1] if hasattr(model, 'predict_proba') else np.zeros(len(Xs))
    results = X.reset_index().assign(Predicted_Theft=y_pred, Theft_Prob=y_prob)
    st.write("### Batch Predictions (sample)")
    st.dataframe(results.head())

# Show anomaly timeline for a selected meter
st.sidebar.markdown("---")
meter_select = st.sidebar.selectbox("Select meter for timeseries", df['Meter_ID'].unique())
meter_df = df[df['Meter_ID'] == meter_select].sort_values('Timestamp').reset_index(drop=True)

st.header(f"Meter: {meter_select} — Time Series & Anomaly Timeline")

# Forecasting: try Prophet if available, otherwise use rolling mean baseline
use_prophet = False
try:
    from prophet import Prophet  # type: ignore
    use_prophet = True
except Exception:
    try:
        from fbprophet import Prophet  # older name
        use_prophet = True
    except Exception:
        use_prophet = False

if use_prophet:
    ts = meter_df[['Timestamp', 'Consumption_kWh']].rename(columns={'Timestamp':'ds','Consumption_kWh':'y'})
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
    m.fit(ts)
    future = m.make_future_dataframe(periods=24*7, freq='H')
    forecast = m.predict(future)
    fig = m.plot(forecast)
    st.pyplot(fig)
else:
    # simple rolling mean forecast for demonstration
    meter_df['rolling_mean_24h'] = meter_df['Consumption_kWh'].rolling(window=24, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(meter_df['Timestamp'], meter_df['Consumption_kWh'], label='Observed', alpha=0.6)
    ax.plot(meter_df['Timestamp'], meter_df['rolling_mean_24h'], label='24h Rolling Mean (baseline)', linestyle='--')
    ax.set_xlabel('Timestamp'); ax.set_ylabel('kWh'); ax.legend(); ax.grid(True)
    st.pyplot(fig)

# Anomaly highlights: sudden dips, night spikes, zero streaks
st.subheader("Detected Anomalies (basic rules)")
anoms = []
# zero streaks > 48 hours
zeros = (meter_df['Consumption_kWh'] == 0)
zero_groups = (zeros != zeros.shift(1)).cumsum()
for _, g in meter_df.groupby(zero_groups):
    if g['Consumption_kWh'].eq(0).all() and len(g) >= 48:
        anoms.append({'type':'zero_streak', 'start':g['Timestamp'].iloc[0], 'end':g['Timestamp'].iloc[-1]})

# night spikes (avg night > 3x day average for that meter)
night_avg = meter_df[meter_df['Timestamp'].dt.hour.isin(range(0,6))]['Consumption_kWh'].mean()
day_avg = meter_df[meter_df['Timestamp'].dt.hour.isin(range(8,20))]['Consumption_kWh'].mean()
if day_avg>0 and (night_avg / day_avg) > 3:
    anoms.append({'type':'night_spike', 'description':f'Night avg {night_avg:.2f} kWh >> day avg {day_avg:.2f} kWh'})

# sudden dips: large negative drops > 90% from previous hour
deltas = meter_df['Consumption_kWh'].pct_change().fillna(0)
dips_idx = meter_df.index[deltas < -0.9].tolist()
for i in dips_idx:
    anoms.append({'type':'sudden_dip', 'time': meter_df.loc[i, 'Timestamp'], 'value': meter_df.loc[i,'Consumption_kWh']})

if anoms:
    for a in anoms:
        st.warning(a)
else:
    st.success("No simple-rule anomalies found for this meter")

# GIS Map (folium) if requested
if show_map:
    try:
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
        for mid, group in df.groupby('Meter_ID'):
            lat = group['Latitude'].iloc[0]; lon = group['Longitude'].iloc[0]
            popup = f"{mid}"
            folium.CircleMarker(location=[lat, lon], radius=4, popup=popup, color='red' if group['Consumption_kWh'].mean()<0.2 else 'blue').add_to(m)
        st_folium(m, width=700)
    except Exception as e:
        st.error("Folium or streamlit-folium not installed. Install with `pip install folium streamlit-folium`. " + str(e))

# Monthly bill estimator for the selected meter (last 30 days)
st.subheader("Monthly Bill Estimator (last 30 days)")
last30 = meter_df[meter_df['Timestamp'] >= (meter_df['Timestamp'].max() - pd.Timedelta(days=30))]
from math import isnan
total_kwh = last30['Consumption_kWh'].sum()
rate = st.sidebar.number_input("Rate per kWh (₹)", value=6.5, step=0.5)
est_bill = total_kwh * rate
st.metric("Estimated last 30d kWh", f"{total_kwh:.2f} kWh", delta=None)
st.metric("Estimated bill (last 30d)", f"₹{est_bill:.2f}")

# SHAP explanations (optional)
st.subheader("Model Explanation (SHAP)")
try:
    import shap
    if 'model' in globals() and model is not None:
        explainer = shap.Explainer(model.predict_proba, scaler.transform(features) if scaler is not None else features)
        shap_values = explainer(features)
        st.write("SHAP available — show summary for first 3 meters")
        st.pyplot(shap.plots.beeswarm(shap_values[:3]))
    else:
        st.info("No trained model to explain. Run training first.")
except Exception as e:
    st.info("SHAP not installed or failed to run. Install shap to enable per-meter explanations.")

# Multi-model comparison (placeholder)
st.subheader("Multi-model Comparison (baseline metrics)")
st.markdown("This section shows how to add multiple models and compare. Use `src/train_model.py` to produce metrics.")

# Severity scoring (simple)
st.subheader("Theft Severity Score (derived)")
if 'results' in globals() and results is not None:
    def severity_label(row):
        prob = row['Theft_Prob'] if 'Theft_Prob' in row else 0.0
        score = 0.0
        score += min(row.get('zero_consumption_hours',0)/24.0,1.0)*2.0
        score += min(row.get('night_day_ratio',0),5.0)*1.5
        score += (1.0 - min(row.get('hourly_profile_std',1.0),1.0))*1.0
        score += prob*2.0
        if score < 2.5: return 'Low'
        if score < 5.0: return 'Medium'
        if score < 7.5: return 'High'
        return 'Critical'
    results['Severity'] = results.apply(severity_label, axis=1)
    st.dataframe(results[['Meter_ID','Predicted_Theft','Theft_Prob','Severity']].sort_values('Theft_Prob', ascending=False).head(20))
else:
    st.info("No model results to show. Run training to enable this table.")

# Auto-refresh demo
if refresh:
    st.info("Auto-refreshing dashboard every 5 seconds (demo).")
    import time
    time.sleep(5)
    st.experimental_rerun()
