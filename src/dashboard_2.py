# import streamlit as st
# import pandas as pd
# import json
# from pathlib import Path
# from datetime import datetime
# import plotly.express as px

# # --------------------------
# # Page Settings
# st.set_page_config(
#     page_title="Karachi AQI Dashboard",
#     page_icon="🌫️",
#     layout="wide"
# )

# # --------------------------
# # Custom CSS Theme
# st.markdown("""

# <style>
# body { background: #eef3fa; }

# .main-title {
#     text-align:center;font-size:40px;font-weight:bold;
#     background:linear-gradient(90deg,#0072ff,#00c6ff);
#     -webkit-background-clip:text;color:transparent;
# }
# .sub-title { text-align:center;font-size:18px;color:#444;margin-bottom:25px; }

# div[data-testid="stMetric"]{
#     padding:18px;border-radius:14px;text-align:center;
#     box-shadow:0 4px 12px rgba(0,0,0,0.15);
# }
# div[data-testid="stMetricValue"] { font-size:22px;color:white !important;font-weight:700;}
# </style>
# """, unsafe_allow_html=True)

# # --------------------------
# # Dashboard Header
# st.markdown("<div class='main-title'>🌫️ Karachi Air Quality Dashboard</div>", unsafe_allow_html=True)
# st.markdown("<div class='sub-title'>📈 Live AQI • 🤖 ML Forecast • 🧪 Pollutant Trends • 📊 PM Ratios</div>", unsafe_allow_html=True)

# # --------------------------
# # File Paths
# LIVE_FILE = "data/processed/latest_aqi_data.csv"
# HISTORY_FILE = "data/karachi_aqi_historical_1year.csv"
# PRED_FILE = "data/predictions/3day_predictions.json"

# # --------------------------
# # Load Datasets
# def load_live():
#     df = pd.read_csv(LIVE_FILE)
#     df["pm25_pm10_ratio"] = df["pm2_5"] / df["pm10"]
#     df["datetime"] = pd.to_datetime(df["datetime"])
#     return df

# def load_history():
#     if not Path(HISTORY_FILE).exists(): return None
#     df = pd.read_csv(HISTORY_FILE)
#     df["datetime"] = pd.to_datetime(df["datetime"])
#     if "aqi_index" in df.columns:
#         df.rename(columns={"aqi_index": "aqi"}, inplace=True)
#     return df

# def load_predictions():
#     if not Path(PRED_FILE).exists(): return None
#     with open(PRED_FILE,"r") as f:
#         data = json.load(f)
#     df = pd.DataFrame({"datetime":data["dates"], "aqi":data["predictions"]})
#     df["datetime"] = pd.to_datetime(df["datetime"])
#     return df, data.get("model","N/A")

# live = load_live()
# history = load_history()
# forecast_data = load_predictions()
# forecast, model_used = forecast_data if forecast_data else (None, "N/A")

# # --------------------------
# # Metrics
# col1, col2, col3, col4 = st.columns(4)

# current_aqi = round(live["aqi_index"].iloc[-1],2)
# next_hour = round(forecast["aqi"].iloc[0],2) if forecast is not None else "N/A"
# avg_3day = round(forecast["aqi"].mean(),2) if forecast is not None else "N/A"

# col1.metric("Current AQI", current_aqi)
# col2.metric("Next Hour Forecast", next_hour)
# col3.metric("3-Day Avg AQI", avg_3day)
# col4.metric("Model Used", model_used)

# st.divider()

# # --------------------------
# # Live AQI Trend
# st.subheader("📈 Live AQI Trend")
# fig_live = px.line(live, x="datetime", y="aqi_index", title="Real-Time AQI")
# st.plotly_chart(fig_live, use_container_width=True)

# # PM Trends
# st.subheader("🌫 PM2.5 vs PM10 Trend")
# fig_pm = px.line(live, x="datetime", y=["pm2_5","pm10"], title="PM Levels")
# st.plotly_chart(fig_pm, use_container_width=True)

# # PM Ratio
# st.subheader("📊 PM2.5 / PM10 Ratio Trend")
# fig_ratio = px.line(live, x="datetime", y="pm25_pm10_ratio", title="PM Ratio")
# st.plotly_chart(fig_ratio, use_container_width=True)

# # --------------------------
# # Map
# st.subheader("🗺️ AQI Map")
# fig_map = px.scatter_mapbox(
#     live, lat="lat", lon="lon", color="aqi_index",
#     hover_data=["city"], zoom=10, height=350,
#     title="Sensor Locations"
# )
# fig_map.update_layout(mapbox_style="open-street-map")
# st.plotly_chart(fig_map, use_container_width=True)

# st.divider()

# # --------------------------
# # Historical + Forecast Combined Chart
# st.subheader("📉 1-Year AQI History + 3-Day Forecast")
# if history is not None and forecast is not None:
#     combined = pd.concat([history[["datetime","aqi"]], forecast], ignore_index=True)
#     fig_hist = px.line(combined, x="datetime", y="aqi", title="AQI: Past & Future")
#     st.plotly_chart(fig_hist, use_container_width=True)

# st.divider()

# # --------------------------
# # Tables
# with st.expander("📜 Latest Live Data"):
#     st.dataframe(live.tail())

# with st.expander("📊 Forecast Data"):
#     if forecast is not None:
#         st.dataframe(forecast)

# with st.expander("📚 1-Year Historical Data"):
#     if history is not None:
#         st.dataframe(history.tail(50))

# # --------------------------
# st.caption(f"✅ Live AQI + ML Forecast | Updated: {datetime.now():%Y-%m-%d %H:%M:%S}")







import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px

# --------------------------
# Page Settings
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# --------------------------
# Custom CSS Theme
st.markdown("""
<style>
body { background: #eef3fa; }

.main-title {
    text-align:center;font-size:40px;font-weight:bold;
    background:linear-gradient(90deg,#0072ff,#00c6ff);
    -webkit-background-clip:text;color:transparent;
}
.sub-title { text-align:center;font-size:18px;color:#444;margin-bottom:25px; }

div[data-testid="stMetric"]{
    padding:18px;border-radius:14px;text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.15);
}
div[data-testid="stMetricValue"] { font-size:22px;color:white !important;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Dashboard Header
st.markdown("<div class='main-title'>🌫️ Karachi Air Quality Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>📈 Live AQI • 🤖 ML Forecast • 🧪 Pollutant Trends • 📊 PM Ratios</div>", unsafe_allow_html=True)

# --------------------------
# File Paths
LIVE_FILE = "data/processed/latest_aqi_data.csv"
HISTORY_FILE = "data/karachi_aqi_historical_1year.csv"
PRED_FILE = "data/predictions/3day_predictions.json"

# --------------------------
# Load Datasets
def load_live():
    df = pd.read_csv(LIVE_FILE)
    df["pm25_pm10_ratio"] = df["pm2_5"] / df["pm10"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def load_history():
    if not Path(HISTORY_FILE).exists(): return None
    df = pd.read_csv(HISTORY_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if "aqi_index" in df.columns:
        df.rename(columns={"aqi_index": "aqi"}, inplace=True)
    return df

def load_predictions():
    if not Path(PRED_FILE).exists(): return None
    with open(PRED_FILE,"r") as f:
        data = json.load(f)
    df = pd.DataFrame({"datetime":data["dates"], "aqi":data["predictions"]})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df, data.get("model","N/A")

live = load_live()
history = load_history()
forecast_data = load_predictions()
forecast, model_used = forecast_data if forecast_data else (None, "N/A")

# --------------------------
# Metrics
col1, col2, col3, col4 = st.columns(4)

current_aqi = round(live["aqi_index"].iloc[-1],2)
next_hour = round(forecast["aqi"].iloc[0],2) if forecast is not None else "N/A"
avg_3day = round(forecast["aqi"].mean(),2) if forecast is not None else "N/A"

col1.metric("Current AQI", current_aqi)
col2.metric("Next Hour Forecast", next_hour)
col3.metric("3-Day Avg AQI", avg_3day)
col4.metric("Model Used", model_used)

st.divider()

# --------------------------
# Live AQI Trend
st.subheader("📈 Live AQI Trend")
fig_live = px.line(live, x="datetime", y="aqi_index", title="Real-Time AQI")
st.plotly_chart(fig_live, use_container_width=True)

# PM Trends
st.subheader("🌫 PM2.5 vs PM10 Trend")
fig_pm = px.line(live, x="datetime", y=["pm2_5","pm10"], title="PM Levels")
st.plotly_chart(fig_pm, use_container_width=True)

# PM Ratio
st.subheader("📊 PM2.5 / PM10 Ratio Trend")
fig_ratio = px.line(live, x="datetime", y="pm25_pm10_ratio", title="PM Ratio")
st.plotly_chart(fig_ratio, use_container_width=True)

# --------------------------
# Pollutant Individual Trends
st.subheader("🧪 Individual Pollutant Trends")

pollutants = {
    "co": "CO (ppm)",
    "no2": "NO₂ (ppm)",
    "o3": "O₃ (ppm)"
}

for col, label in pollutants.items():
    if col in live.columns:
        st.write(f"**{label} Trend**")
        fig = px.line(live, x="datetime", y=col, title=f"{label} Over Time")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Map
st.subheader("🗺️ AQI Map")
fig_map = px.scatter_mapbox(
    live, lat="lat", lon="lon", color="aqi_index",
    hover_data=["city"], zoom=10, height=350,
    title="Sensor Locations"
)
fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# --------------------------
# Historical + Forecast Combined Chart
st.subheader("📉 1-Year AQI History + 3-Day Forecast")
if history is not None and forecast is not None:
    combined = pd.concat([history[["datetime","aqi"]], forecast], ignore_index=True)
    fig_hist = px.line(combined, x="datetime", y="aqi", title="AQI: Past & Future")
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# --------------------------
# Tables
with st.expander("📜 Latest Live Data"):
    st.dataframe(live.tail())

with st.expander("📊 Forecast Data"):
    if forecast is not None:
        st.dataframe(forecast)

with st.expander("📚 1-Year Historical Data"):
    if history is not None:
        st.dataframe(history.tail(50))

# --------------------------
st.caption(f"✅ Live AQI + ML Forecast | Updated: {datetime.now():%Y-%m-%d %H:%M:%S}")
