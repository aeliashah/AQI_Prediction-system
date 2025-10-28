# src/fetch_aqi_data.py

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

load_dotenv()

OW_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

LAT = "24.8607"
LON = "67.0011"

url = "http://api.openweathermap.org/data/2.5/air_pollution"
params = {"lat": LAT, "lon": LON, "appid": OW_API_KEY}

r = requests.get(url, params=params)
data = r.json()

record = data["list"][0]
dt = datetime.utcfromtimestamp(record["dt"])

df = pd.DataFrame([{
    "datetime": dt,
    "aqi": record["main"]["aqi"],
    "pm2_5": record["components"].get("pm2_5"),
    "pm10": record["components"].get("pm10"),
    "co": record["components"].get("co"),
    "no2": record["components"].get("no2"),
    "o3": record["components"].get("o3"),
    "hour": dt.hour,
    "dayofweek": dt.weekday(),
    "is_weekend": dt.weekday() in [5,6],
    "month": dt.month,
}])

df["aqi"] = df["aqi"].astype(int)

os.makedirs("data", exist_ok=True)
df.to_csv("data/latest_aqi.csv", index=False)

print("✅ Live AQI saved")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

fg = fs.get_feature_group("aqi_features_karachi", version=1)
fg.insert(df)

print("🚀 Live AQI appended to Feature Store")
