# src/ingest_live_openweather.py
import requests, os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from store_features import compute_all_features
from src.hopsworks_utils import login_project, get_feature_group

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT")
LON = os.getenv("LON")
CITY = os.getenv("CITY")

def fetch_openweather_now(lat, lon, api_key):
    air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    a = requests.get(air_url, timeout=10).json()
    w = requests.get(weather_url, timeout=10).json()
    item = a["list"][0]
    ts = datetime.utcfromtimestamp(item["dt"])
    components = item["components"]
    data = {
        "city": CITY,
        "event_timestamp": ts,
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "co": components.get("co"),
        "aqi": item["main"].get("aqi"), # note: improves if present
        "temperature": w["main"].get("temp"),
        "humidity": w["main"].get("humidity"),
        "wind_speed": w["wind"].get("speed") if "wind" in w else None,
        "pressure": w["main"].get("pressure")
    }
    return pd.DataFrame([data])

def ingest_and_append():
    df = fetch_openweather_now(LAT, LON, API_KEY)
    df = compute_all_features(df)
    project, fs = login_project()
    fg = get_feature_group(fs)
    fg.insert(df, overwrite=False)
    print("Inserted live row:", df.iloc[0].to_dict())

if __name__ == "__main__":
    ingest_and_append()
