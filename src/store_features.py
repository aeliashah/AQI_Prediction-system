# # src/features.py
# import pandas as pd

# def ensure_datetime(df, ts_col="event_timestamp"):
#     df[ts_col] = pd.to_datetime(df[ts_col])
#     return df

# def add_time_features(df, ts_col="event_timestamp"):
#     df = ensure_datetime(df, ts_col)
#     df["hour"] = df[ts_col].dt.hour
#     df["dayofweek"] = df[ts_col].dt.dayofweek
#     df["is_weekend"] = df["dayofweek"] >= 5
#     df["month"] = df[ts_col].dt.month
#     return df

# def add_lag_features(df, target_col="aqi", lags=[1,24]):
#     df = df.sort_values("event_timestamp")
#     for lag in lags:
#         df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)
#     return df

# def add_rolling_features(df, col="aqi", windows=[3,24]):
#     df = df.sort_values("event_timestamp")
#     for w in windows:
#         df[f"{col}_rolling_{w}h"] = df[col].rolling(window=w, min_periods=1).mean()
#     return df

# def add_rate_of_change(df, col="aqi", period=24):
#     df = df.sort_values("event_timestamp")
#     df[f"{col}_roc_{period}h"] = (df[col] - df[col].shift(period)) / period
#     return df

# def compute_all_features(df):
#     df = add_time_features(df)
#     if "aqi" in df.columns:
#         df = add_lag_features(df, "aqi", lags=[1,24,48])
#         df = add_rolling_features(df, "aqi", windows=[3,24])
#         df = add_rate_of_change(df, "aqi", 24)
#     # handle pm2_5/pm10 ratio
#     if "pm2_5" in df.columns and "pm10" in df.columns:
#         df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
#     df = df.fillna(method="ffill").fillna(0)
#     return df


import hopsworks
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

# Example: Load your historical data (1 year)
# If you already have it in a DataFrame, skip this
df = pd.read_csv("data/historical_aqi.csv")

# Optional — ensure datetime is parsed properly
df["datetime"] = pd.to_datetime(df["datetime"])

# Create Feature Group
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    primary_key=["datetime"],
    description="1-year historical AQI and weather data from Open-Meteo",
    online_enabled=True
)

# Insert data
aqi_fg.insert(df)
print("✅ Historical data successfully stored in Hopsworks Feature Store")
