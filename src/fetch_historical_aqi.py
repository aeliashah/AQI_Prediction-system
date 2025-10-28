# # # import os
# # # import requests
# # # import pandas as pd
# # # from datetime import datetime, timedelta
# # # from dotenv import load_dotenv

# # # # Load your .env variables
# # # load_dotenv()
# # # API_KEY = os.getenv("OPENWEATHER_API_KEY")
# # # LAT = "33.6844"   # Islamabad latitude
# # # LON = "73.0479"   # Islamabad longitude

# # # # Get 1 year range
# # # end_date = datetime.utcnow()
# # # start_date = end_date - timedelta(days=365)

# # # # OpenWeather Air Pollution Historical API
# # # url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
# # # params = {
# # #     "lat": LAT,
# # #     "lon": LON,
# # #     "start": int(start_date.timestamp()),
# # #     "end": int(end_date.timestamp()),
# # #     "appid": API_KEY
# # # }

# # # response = requests.get(url, params=params)

# # # # Check for errors
# # # if response.status_code != 200:
# # #     print("❌ API Error:", response.status_code, response.text)
# # #     exit()

# # # data = response.json()

# # # if "list" not in data:
# # #     print("❌ No 'list' found in response. Response was:", data)
# # #     exit()

# # # rows = []
# # # for record in data["list"]:
# # #     rows.append({
# # #         "datetime": datetime.utcfromtimestamp(record["dt"]),
# # #         "aqi": record["main"]["aqi"],
# # #         "pm2_5": record["components"].get("pm2_5"),
# # #         "pm10": record["components"].get("pm10"),
# # #         "co": record["components"].get("co"),
# # #         "no2": record["components"].get("no2"),
# # #         "o3": record["components"].get("o3")
# # #     })

# # # df_hist = pd.DataFrame(rows)
# # # os.makedirs("data", exist_ok=True)
# # # df_hist.to_csv("data/aqi_historical_1year.csv", index=False)

# # # print(f"✅ Saved {len(df_hist)} records of 1-year AQI history to CSV.")



# # import os
# # import requests
# # import pandas as pd
# # from datetime import datetime, timedelta
# # from dotenv import load_dotenv
# # import hopsworks

# # # ============================
# # # STEP 1: Load .env and Setup
# # # ============================
# # load_dotenv()
# # API_KEY = os.getenv("OPENWEATHER_API_KEY")
# # LAT = "33.6844"   # Islamabad latitude
# # LON = "73.0479"   # Islamabad longitude

# # # Hopsworks login
# # project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# # fs = project.get_feature_store()

# # # ============================
# # # STEP 2: Fetch 1-Year Historical AQI Data
# # # ============================
# # end_date = datetime.utcnow()
# # start_date = end_date - timedelta(days=365)

# # url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
# # params = {
# #     "lat": LAT,
# #     "lon": LON,
# #     "start": int(start_date.timestamp()),
# #     "end": int(end_date.timestamp()),
# #     "appid": API_KEY
# # }

# # print(f"🌍 Fetching AQI data from {start_date.date()} to {end_date.date()} ...")
# # response = requests.get(url, params=params)

# # if response.status_code != 200:
# #     print("❌ API Error:", response.status_code, response.text)
# #     exit()

# # data = response.json()
# # if "list" not in data:
# #     print("❌ No 'list' found in response. Response was:", data)
# #     exit()

# # rows = []
# # for record in data["list"]:
# #     rows.append({
# #         "datetime": datetime.utcfromtimestamp(record["dt"]),
# #         "aqi": record["main"]["aqi"],
# #         "pm2_5": record["components"].get("pm2_5"),
# #         "pm10": record["components"].get("pm10"),
# #         "co": record["components"].get("co"),
# #         "no2": record["components"].get("no2"),
# #         "o3": record["components"].get("o3")
# #     })

# # df_hist = pd.DataFrame(rows)
# # os.makedirs("data", exist_ok=True)
# # df_hist.to_csv("data/aqi_historical_1year.csv", index=False)
# # print(f"✅ Saved {len(df_hist)} records of 1-year AQI history to CSV.")

# # # ============================
# # # STEP 3: Feature Engineering
# # # ============================
# # df_hist = df_hist.sort_values("datetime").drop_duplicates(subset="datetime")
# # df_hist["hour"] = df_hist["datetime"].dt.hour
# # df_hist["dayofweek"] = df_hist["datetime"].dt.dayofweek
# # df_hist["is_weekend"] = df_hist["dayofweek"].isin([5,6]).astype(int)
# # df_hist["month"] = df_hist["datetime"].dt.month

# # # Extra features for model
# # df_hist["aqi_diff"] = df_hist["aqi"].diff()
# # df_hist["aqi_rolling_3h"] = df_hist["aqi"].rolling(3).mean()
# # df_hist["aqi_roc_24h"] = df_hist["aqi"].pct_change(periods=24)
# # df_hist["pm25_pm10_ratio"] = df_hist["pm2_5"] / (df_hist["pm10"] + 1e-6)

# # df_hist = df_hist.dropna().reset_index(drop=True)
# # print(f"✅ Engineered features added. Final rows: {len(df_hist)}")

# # # ============================
# # # STEP 4: Upload to Hopsworks (append mode)
# # # ============================
# # feature_group = fs.get_or_create_feature_group(
# #     name="aqi_features",
# #     version=2,
# #     primary_key=["datetime"],
# #     description="1-year historical hourly AQI features for training",
# #     online_enabled=True
# # )

# # feature_group.insert(df_hist, write_options={"wait_for_job": True})
# # print(f"✅ Uploaded {len(df_hist)} records to Hopsworks Feature Store (aqi_features v2)")



# import os
# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# import hopsworks

# # ============================
# # STEP 1: Load Environment Variables
# # ============================
# load_dotenv()
# API_KEY = os.getenv("OPENWEATHER_API_KEY")
# LAT = "24.8607"   # Karachi latitude
# LON = "67.0011"   # Karachi longitude


# # ============================
# # STEP 2: Fetch Historical Data
# # ============================
# end_date = datetime.utcnow()
# start_date = end_date - timedelta(days=365)

# print(f"🌍 Fetching AQI data from {start_date.date()} to {end_date.date()} ...")

# url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
# params = {
#     "lat": LAT,
#     "lon": LON,
#     "start": int(start_date.timestamp()),
#     "end": int(end_date.timestamp()),
#     "appid": API_KEY
# }

# response = requests.get(url, params=params)

# # Check for errors
# if response.status_code != 200:
#     print("❌ API Error:", response.status_code, response.text)
#     exit()

# data = response.json()
# if "list" not in data:
#     print("❌ No 'list' found in response. Response was:", data)
#     exit()

# # Convert API response into a DataFrame
# rows = []
# for record in data["list"]:
#     rows.append({
#         "datetime": datetime.utcfromtimestamp(record["dt"]),
#         "aqi": record["main"]["aqi"],
#         "pm2_5": record["components"].get("pm2_5"),
#         "pm10": record["components"].get("pm10"),
#         "co": record["components"].get("co"),
#         "no2": record["components"].get("no2"),
#         "o3": record["components"].get("o3")
#     })

# df_hist = pd.DataFrame(rows)
# os.makedirs("data", exist_ok=True)
# df_hist.to_csv("data/aqi_historical_1year.csv", index=False)
# print(f"✅ Saved {len(df_hist)} records of 1-year AQI history to CSV.")

# # ============================
# # STEP 3: Feature Engineering
# # ============================
# df_hist = df_hist.sort_values("datetime").drop_duplicates(subset="datetime")

# df_hist["hour"] = df_hist["datetime"].dt.hour
# df_hist["dayofweek"] = df_hist["datetime"].dt.dayofweek
# df_hist["is_weekend"] = df_hist["dayofweek"].isin([5, 6])  # boolean
# df_hist["month"] = df_hist["datetime"].dt.month

# # Add datetime_str for schema compatibility
# df_hist["datetime_str"] = df_hist["datetime"].astype(str)

# # Derived features
# df_hist["aqi_diff"] = df_hist["aqi"].diff()
# df_hist["aqi_rolling_3h"] = df_hist["aqi"].rolling(3).mean()
# df_hist["aqi_roc_24h"] = df_hist["aqi"].pct_change(periods=24)
# df_hist["pm25_pm10_ratio"] = df_hist["pm2_5"] / (df_hist["pm10"] + 1e-6)

# df_hist = df_hist.dropna().reset_index(drop=True)
# print(f"✅ Engineered features added. Final rows: {len(df_hist)}")

# # ============================
# # STEP 4: Upload to Hopsworks
# # ============================
# print("🔗 Connecting to Hopsworks...")
# project = hopsworks.login()
# fs = project.get_feature_store()

# # Ensure proper type
# df_hist["is_weekend"] = df_hist["is_weekend"].astype(bool)

# feature_group = fs.get_or_create_feature_group(
#     name="aqi_features",
#     version=2,
#     primary_key=["datetime"],
#     description="1-year historical hourly AQI features for training",
#     online_enabled=True
# )

# feature_group.insert(df_hist, write_options={"wait_for_job": True})
# print(f"✅ Uploaded {len(df_hist)} records to Hopsworks Feature Store (aqi_features v2)")

# print("🎉 All done — 1-year AQI history fetched, processed, and uploaded successfully!")


#here

# import os
# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# import hopsworks

# # ==============================
# # 1️⃣ Load environment variables
# # ==============================
# load_dotenv()
# API_KEY = os.getenv("OPENWEATHER_API_KEY")

# # Karachi coordinates
# LAT = "24.8607"
# LON = "67.0011"

# # ==============================
# # 2️⃣ Fetch historical AQI data (1 year)
# # ==============================
# end_date = datetime.utcnow()
# start_date = end_date - timedelta(days=365)

# print(f"🌍 Fetching AQI data for Karachi from {start_date.date()} to {end_date.date()} ...")

# url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
# params = {
#     "lat": LAT,
#     "lon": LON,
#     "start": int(start_date.timestamp()),
#     "end": int(end_date.timestamp()),
#     "appid": API_KEY
# }

# response = requests.get(url, params=params)
# if response.status_code != 200:
#     print("❌ API Error:", response.status_code, response.text)
#     exit()

# data = response.json()
# if "list" not in data:
#     print("❌ No 'list' in API response. Full response:", data)
#     exit()

# rows = []
# for record in data["list"]:
#     rows.append({
#         "datetime": datetime.utcfromtimestamp(record["dt"]),
#         "aqi": record["main"]["aqi"],
#         "pm2_5": record["components"].get("pm2_5"),
#         "pm10": record["components"].get("pm10"),
#         "co": record["components"].get("co"),
#         "no2": record["components"].get("no2"),
#         "o3": record["components"].get("o3")
#     })

# df_hist = pd.DataFrame(rows)
# print(f"✅ Saved {len(df_hist)} raw AQI records for Karachi.")

# # ==============================
# # 3️⃣ Feature Engineering
# # ==============================
# df_hist["hour"] = df_hist["datetime"].dt.hour
# df_hist["dayofweek"] = df_hist["datetime"].dt.dayofweek
# df_hist["is_weekend"] = df_hist["dayofweek"].isin([5, 6]).astype(bool)
# df_hist["month"] = df_hist["datetime"].dt.month

# # Derived metrics
# df_hist["aqi_diff"] = df_hist["aqi"].diff()
# df_hist["aqi_rolling_3h"] = df_hist["aqi"].rolling(window=3).mean()
# df_hist["aqi_roc_24h"] = df_hist["aqi"].pct_change(periods=24)
# df_hist["pm25_pm10_ratio"] = df_hist["pm2_5"] / df_hist["pm10"]
# df_hist["datetime_str"] = df_hist["datetime"].astype(str)

# # Clean data
# df_hist = df_hist.replace([float("inf"), -float("inf")], None).fillna(0)
# df_hist = df_hist.dropna()
# print(f"✅ Engineered features added. Final rows: {len(df_hist)}")

# # ==============================
# # 4️⃣ Save locally (optional)
# # ==============================
# os.makedirs("data", exist_ok=True)
# df_hist.to_csv("data/karachi_aqi_historical_1year.csv", index=False)
# print("💾 Saved to data/karachi_aqi_historical_1year.csv")

# # ==============================
# # 5️⃣ Upload to Hopsworks Feature Store
# # ==============================
# print("🔗 Connecting to Hopsworks...")
# project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# fs = project.get_feature_store()

# feature_group = fs.get_or_create_feature_group(
#     name="aqi_features_karachi",
#     version=1,
#     description="1-year Karachi AQI data with engineered features",
#     primary_key=["datetime"],
#     event_time="datetime"
# )

# # ✅ Ensure AQI type matches schema
# df_hist["aqi"] = df_hist["aqi"].astype(int)

# feature_group.insert(df_hist, write_options={"wait_for_job": True})

# print(f"✅ Uploaded {len(df_hist)} records to Hopsworks Feature Store (aqi_features_karachi v1)")
# print("🎉 All done — Karachi 1-year AQI history fetched, processed, and uploaded successfully!")


# import os
# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
# import hopsworks

# # ==============================
# # 1️⃣ Load environment variables
# # ==============================
# load_dotenv()

# API_KEY = os.getenv("OPENWEATHER_API_KEY")
# HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# if not API_KEY or not HOPSWORKS_API_KEY:
#     raise Exception("❌ Missing API Keys — Check .env or GitHub Secrets")

# LAT = "24.8607"
# LON = "67.0011"

# # ==============================
# # 2️⃣ Fetch 1 year AQI history
# # ==============================
# end_date = datetime.utcnow()
# start_date = end_date - timedelta(days=365)

# print(f"🌍 Fetching Karachi AQI data {start_date.date()} ➝ {end_date.date()}...")

# url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# params = {
#     "lat": LAT,
#     "lon": LON,
#     "start": int(start_date.timestamp()),
#     "end": int(end_date.timestamp()),
#     "appid": API_KEY
# }

# response = requests.get(url, params=params)

# if response.status_code != 200:
#     print("❌ API Error:", response.status_code, response.text)
#     exit()

# data = response.json()
# if "list" not in data:
#     print("❌ Invalid API response:", data)
#     exit()

# rows = []
# for r in data["list"]:
#     rows.append({
#         "datetime": datetime.utcfromtimestamp(r["dt"]),
#         "aqi": r["main"]["aqi"],
#         "pm2_5": r["components"].get("pm2_5"),
#         "pm10": r["components"].get("pm10"),
#         "co": r["components"].get("co"),
#         "no2": r["components"].get("no2"),
#         "o3": r["components"].get("o3")
#     })

# df = pd.DataFrame(rows)

# print(f"✅ Retrieved {len(df)} raw rows")

# # ==============================
# # 3️⃣ Feature Engineering
# # ==============================
# df["hour"] = df["datetime"].dt.hour
# df["dayofweek"] = df["datetime"].dt.dayofweek
# df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
# df["month"] = df["datetime"].dt.month

# df["aqi_diff"] = df["aqi"].diff()
# df["aqi_rolling_3h"] = df["aqi"].rolling(3).mean()
# df["aqi_roc_24h"] = df["aqi"].pct_change(24)
# df["pm25_pm10_ratio"] = df["pm2_5"] / df["pm10"]

# df["datetime"] = pd.to_datetime(df["datetime"])  # ✅ fix dtype

# df = df.replace([float("inf"), -float("inf")], None).dropna()

# print(f"✅ Features ready | Final rows: {len(df)}")

# # ==============================
# # 4️⃣ Save locally
# # ==============================
# os.makedirs("data", exist_ok=True)
# df.to_csv("data/karachi_aqi_historical_1year.csv", index=False)
# print("💾 Saved CSV")

# # ==============================
# # 5️⃣ Upload to Hopsworks FS
# # ==============================
# print("🔗 Connecting to Hopsworks...")

# project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
# fs = project.get_feature_store()

# feature_group = fs.get_or_create_feature_group(
#     name="aqi_features_karachi",
#     version=1,
#     primary_key=["datetime"],
#     event_time="datetime",
#     description="Karachi AQI history + features"
# )

# df["aqi"] = df["aqi"].astype(int)  # ✅ schema fix

# feature_group.insert(df, write_options={"wait_for_job": True})

# print(f"✅ Uploaded {len(df)} rows to Hopsworks")
# print("🎉 Done!")



import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks

# ==============================
# 1️⃣ Load environment variables
# ==============================
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not API_KEY or not HOPSWORKS_API_KEY:
    raise Exception("❌ Missing API Keys — Check .env or GitHub Secrets")

LAT = "24.8607"
LON = "67.0011"

# ==============================
# 2️⃣ Fetch 1 year AQI history
# ==============================
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=365)

print(f"🌍 Fetching Karachi AQI data {start_date.date()} ➝ {end_date.date()}...")

url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

params = {
    "lat": LAT,
    "lon": LON,
    "start": int(start_date.timestamp()),
    "end": int(end_date.timestamp()),
    "appid": API_KEY
}

response = requests.get(url, params=params)

if response.status_code != 200:
    print("❌ API Error:", response.status_code, response.text)
    exit()

data = response.json()
if "list" not in data:
    print("❌ Invalid API response:", data)
    exit()

rows = []
for r in data["list"]:
    rows.append({
        "datetime": datetime.utcfromtimestamp(r["dt"]),
        "aqi": r["main"]["aqi"],
        "pm2_5": r["components"].get("pm2_5"),
        "pm10": r["components"].get("pm10"),
        "co": r["components"].get("co"),
        "no2": r["components"].get("no2"),
        "o3": r["components"].get("o3")
    })

df = pd.DataFrame(rows)

print(f"✅ Retrieved {len(df)} raw rows")

# ==============================
# 3️⃣ Feature Engineering
# ==============================
df["datetime"] = pd.to_datetime(df["datetime"])

df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6])  # ✅ boolean not int
df["month"] = df["datetime"].dt.month

df["aqi_diff"] = df["aqi"].diff()
df["aqi_rolling_3h"] = df["aqi"].rolling(3).mean()
df["aqi_roc_24h"] = df["aqi"].pct_change(24)
df["pm25_pm10_ratio"] = df["pm2_5"] / df["pm10"]

# ✅ Replace inf and NaN
df = df.replace([float("inf"), -float("inf")], None).dropna()

# ✅ Deduplicate
df = df.drop_duplicates(subset=["datetime"])

# ✅ Correct dtypes for Hopsworks
df["aqi"] = df["aqi"].astype(int)
df["is_weekend"] = df["is_weekend"].astype(bool)

# ✅ Add missing required string column
df["datetime_str"] = df["datetime"].astype(str)

print(f"✅ Features ready | Final rows: {len(df)}")

# ==============================
# 4️⃣ Save locally
# ==============================
os.makedirs("data", exist_ok=True)
df.to_csv("data/karachi_aqi_historical_1year.csv", index=False)
print("💾 Saved CSV")

# ==============================
# 5️⃣ Upload to Hopsworks FS
# ==============================
print("🔗 Connecting to Hopsworks...")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

feature_group = fs.get_or_create_feature_group(
    name="aqi_features_karachi",
    version=1,
    primary_key=["datetime"],
    event_time="datetime",
    description="Karachi AQI history + features"
)

feature_group.insert(df, write_options={"wait_for_job": True})

print(f"✅ Uploaded {len(df)} rows to Hopsworks")
print("🎉 Done!")
