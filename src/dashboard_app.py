# src/dashboard_app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="AQI Forecast", layout="wide")
st.title("AQI Forecast Dashboard")

city = st.text_input("City", "Karachi")
if st.button("Get Latest Prediction"):
    resp = requests.post("http://127.0.0.1:8080/predict", json={"city": city})
    if resp.ok:
        data = resp.json()
        st.metric("Predicted AQI", round(data["predicted_aqi"], 2))
    else:
        st.error(resp.text)

# Optional: show recent measured AQI from Hopsworks (not shown here)
