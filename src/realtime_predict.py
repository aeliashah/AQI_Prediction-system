# import os
# import json
# from datetime import datetime
# import requests
# import pandas as pd
# import hopsworks
# import joblib

# from dotenv import load_dotenv
# load_dotenv()

# # -------------------------------
# # 1️⃣ Connect to Hopsworks
# # -------------------------------
# project = hopsworks.login()
# mr = project.get_model_registry()

# print("🔍 Loading latest AQI model from Hopsworks...")
# model = mr.get_model("Karachi_AQI_Predictor", version=None)  # latest version automatically
# model_dir = model.download()
# model_file = [f for f in os.listdir(model_dir) if f.endswith(".pkl")][0]
# model = joblib.load(os.path.join(model_dir, model_file))
# print(f"✅ Loaded model: {model_file}")

# # -------------------------------
# # 2️⃣ Fetch Live AQI Data
# # -------------------------------
# API_KEY = os.getenv("OPENWEATHER_API_KEY")
# CITY = "Karachi"
# url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=24.8607&lon=67.0011&appid={API_KEY}"

# response = requests.get(url).json()

# aqi_live = response["list"][0]["main"]["aqi"]
# components = response["list"][0]["components"]

# print("🌍 Live AQI Data Fetched")

# # -------------------------------
# # 3️⃣ Prepare Data for Model
# # -------------------------------
# df = pd.DataFrame([{
#     "aqi": aqi_live,
#     "pm2_5": components["pm2_5"],
#     "pm10": components["pm10"],
#     "no2": components["no2"],
#     "so2": components["so2"],
#     "o3": components["o3"],
#     "co": components["co"],
# }])

# # Drop target if exists (some models expect only features)
# X = df.drop(columns=["aqi"], errors='ignore')  

# # -------------------------------
# # 4️⃣ Predict AQI
# # -------------------------------
# pred_aqi = model.predict(X)[0]

# print("\n🤖 Model Prediction Complete")
# print(f"📡 Live AQI    : {aqi_live}")
# print(f"🚀 Predicted AQI: {pred_aqi:.2f}")

# # -------------------------------
# # 5️⃣ Save latest prediction
# # -------------------------------
# output = {
#     "city": CITY,
#     "time": str(datetime.now()),
#     "live_aqi": float(aqi_live),
#     "predicted_aqi": float(pred_aqi)
# }

# os.makedirs("predictions", exist_ok=True)
# with open("predictions/latest_prediction.json", "w") as f:
#     json.dump(output, f, indent=4)

# print("\n✅ Saved latest prediction to predictions/latest_prediction.json")
# print("🎯 Real-Time AQI Pipeline Complete")


import os
import json
from datetime import datetime, timedelta
import requests
import pandas as pd
import hopsworks
import joblib
from dotenv import load_dotenv
load_dotenv()

# -------------------------------
# 1️⃣ Connect to Hopsworks
# -------------------------------
project = hopsworks.login()
mr = project.get_model_registry()

print("🔍 Loading latest AQI model from Hopsworks...")
model = mr.get_model("Karachi_AQI_Predictor", version=None)
model_dir = model.download()
model_file = [f for f in os.listdir(model_dir) if f.endswith(".pkl")][0]
model = joblib.load(os.path.join(model_dir, model_file))
print(f"✅ Loaded model: {model_file}")

# -------------------------------
# 2️⃣ Fetch Live AQI Data
# -------------------------------
API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Karachi"
LAT = "24.8607"
LON = "67.0011"

url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
response = requests.get(url).json()

aqi_live = response["list"][0]["main"]["aqi"]
components = response["list"][0]["components"]

now = datetime.utcnow()

# -------------------------------
# 3️⃣ Create feature row SAME as training
# -------------------------------
df = pd.DataFrame([{
    "hour": now.hour,
    "dayofweek": now.weekday(),
    "is_weekend": 1 if now.weekday() >= 5 else 0,
    "month": now.month,
    "pm2_5": components["pm2_5"],
    "pm10": components["pm10"],
    "co": components["co"],
    "no2": components["no2"],
    "o3": components["o3"],
    
    # placeholders — real values handled by feature store
    "aqi_diff": 0,
    "aqi_rolling_3h": aqi_live,
    "aqi_roc_24h": 0,

    "pm25_pm10_ratio": components["pm2_5"] / components["pm10"] if components["pm10"] != 0 else 0
}])

# -------------------------------
# 4️⃣ Predict
# -------------------------------
predicted_aqi = round(model.predict(df)[0], 2)

print(f"\n🌍 Live AQI: {aqi_live}")
print(f"🤖 Predicted AQI: {predicted_aqi}")

# -------------------------------
# 5️⃣ Save result
# -------------------------------
os.makedirs("predictions", exist_ok=True)
output = {
    "datetime": now.isoformat(),
    "city": CITY,
    "live_aqi": float(aqi_live),
    "predicted_aqi": float(predicted_aqi)
}

with open("predictions/latest_prediction.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Saved latest_prediction.json")
