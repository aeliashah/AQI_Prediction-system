# src/predict_aqi.py
import os
import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import joblib

# =============================
# STEP 1: Setup & Login
# =============================
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

print("\n🔗 Connected to Hopsworks successfully!")

# =============================
# STEP 2: Load latest Karachi data from Feature Store
# =============================
print("📥 Loading latest data from aqi_features_karachi v1 ...")
feature_group = fs.get_feature_group("aqi_features_karachi", version=1)
df = feature_group.read()

print(f"✅ Loaded {len(df)} historical records for Karachi.")

# Sort by datetime
df = df.sort_values("datetime")
latest_data = df.tail(72).copy()  # last 3 days (72 hours)
latest_features = latest_data[[
    "hour", "dayofweek", "is_weekend", "month",
    "pm2_5", "pm10", "co", "no2", "o3",
    "aqi_diff", "aqi_rolling_3h", "aqi_roc_24h", "pm25_pm10_ratio"
]].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"🧾 Prepared latest {len(latest_features)} feature rows for prediction.")

# =============================
# STEP 3: Load latest model from Model Registry
# =============================
mr = project.get_model_registry()
model = mr.get_model("Karachi_AQI_Predictor", version=1)
model_dir = model.download()
model_path = os.path.join(model_dir, "LinearRegression_karachi_aqi_model.pkl")

predictor = joblib.load(model_path)
print(f"🤖 Loaded model: Karachi_AQI_Predictor (LinearRegression)")

# =============================
# STEP 4: Make next 3-day AQI predictions
# =============================

# Assume hourly forecast for next 72 hours
last_timestamp = df["datetime"].max()
future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(72)]

# Predict
preds = predictor.predict(latest_features)

forecast_df = pd.DataFrame({
    "datetime": future_timestamps,
    "predicted_aqi": preds
})

forecast_df["predicted_aqi"] = forecast_df["predicted_aqi"].round(2)

print("\n🌤️  3-Day AQI Forecast for Karachi:")
print(forecast_df.head(10))

# =============================
# STEP 5: Save results locally
# =============================
os.makedirs("data", exist_ok=True)
forecast_path = "data/karachi_aqi_forecast.csv"
forecast_df.to_csv(forecast_path, index=False)

print(f"\n💾 Forecast saved to {forecast_path}")
print("🎉 Done — Real-time prediction pipeline ready for Karachi AQI!\n")

# =============================
# STEP 6: Optional — Upload to Hopsworks (if you want versioning)
# =============================
try:
    fg_pred = fs.get_or_create_feature_group(
        name="aqi_forecast_karachi",
        version=1,
        primary_key=["datetime"],
        description="3-day AQI forecast for Karachi"
    )
    fg_pred.insert(forecast_df)
    print("📤 Uploaded forecast to Hopsworks Feature Store successfully.")
except Exception as e:
    print(f"⚠️ Skipped Hopsworks upload: {e}")
