# # # import hopsworks
# # # import pandas as pd
# # # from datetime import datetime

# # # project = hopsworks.login()
# # # fs = project.get_feature_store()

# # # # --- STEP 1: Load your existing small CSV ---
# # # df_recent = pd.read_csv(r"C:\Users\User\projects\aqi-forecast\data\raw")
# # # df_recent["datetime"] = pd.to_datetime(df_recent["datetime"])

# # # # --- STEP 2: (Optional) Load 1-year historical data if you have it ---
# # # # For now, keep this empty; we'll fetch and add later
# # # df_historical = pd.DataFrame()

# # # # --- STEP 3: Combine ---
# # # df = pd.concat([df_historical, df_recent], ignore_index=True)
# # # df = df.sort_values("datetime")

# # # # --- STEP 4: Create Feature Group ---
# # # feature_group = fs.get_or_create_feature_group(
# # #     name="aqi_features",
# # #     version=1,
# # #     primary_key=["datetime"],
# # #     description="Historical and recent AQI + weather data",
# # #     online_enabled=True
# # # )

# # # # --- STEP 5: Insert data ---
# # # feature_group.insert(df, write_options={"wait_for_job": True})

# # # print("✅ Data successfully stored in Hopsworks Feature Store!")


# # # import hopsworks
# # # import pandas as pd
# # # from datetime import datetime

# # # # --- STEP 1: Connect to Hopsworks ---
# # # project = hopsworks.login()
# # # fs = project.get_feature_store()

# # # # --- STEP 2: Load your 1-year AQI dataset ---
# # # df = pd.read_csv("data/aqi_historical_1year.csv")
# # # df["datetime"] = pd.to_datetime(df["datetime"])
# # # df = df.sort_values("datetime")

# # # # --- STEP 3: Create or get Feature Group ---
# # # feature_group = fs.get_or_create_feature_group(
# # #     name="aqi_features",
# # #     version=1,
# # #     primary_key=["datetime"],
# # #     description="1-year historical AQI and pollutant data",
# # #     online_enabled=True
# # # )

# # # # --- STEP 4: Insert data into Feature Store ---
# # # feature_group.insert(df, write_options={"wait_for_job": True})

# # # print("✅ Data successfully stored in Hopsworks Feature Store!")


# # import os
# # import hopsworks
# # from dotenv import load_dotenv
# # import pandas as pd
# # from datetime import datetime

# # # --- STEP 0: Load .env and authenticate with API key ---
# # load_dotenv()
# # project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# # fs = project.get_feature_store()

# # # --- STEP 1: Load your 1-year AQI dataset ---
# # df = pd.read_csv("data/aqi_historical_1year.csv")
# # df["datetime"] = pd.to_datetime(df["datetime"])
# # df = df.sort_values("datetime")

# # # --- STEP 2: Create or get Feature Group ---
# # feature_group = fs.get_or_create_feature_group(
# #     name="aqi_features",
# #     version=1,
# #     primary_key=["datetime"],
# #     description="1-year historical AQI and pollutant data",
# #     online_enabled=True
# # )

# # # --- STEP 3: Insert data into Feature Store ---
# # feature_group.insert(df, write_options={"wait_for_job": True})

# # print("✅ Data successfully stored in Hopsworks Feature Store!")







# import os
# import hopsworks
# from dotenv import load_dotenv
# import pandas as pd
# from datetime import datetime

# # --- STEP 0: Load .env and connect ---
# load_dotenv()
# project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# fs = project.get_feature_store()

# # --- STEP 1: Load your 1-year AQI dataset ---
# df = pd.read_csv("data/aqi_historical_1year.csv")
# df["datetime"] = pd.to_datetime(df["datetime"])

# # Convert datetime to string (ISO format) for online feature store
# df["datetime_str"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

# df = df.sort_values("datetime")

# # --- STEP 2: Create or get Feature Group ---
# feature_group = fs.get_or_create_feature_group(
#     name="aqi_features",
#     version=1,
#     primary_key=["datetime_str"],   # use string version
#     description="1-year historical AQI and pollutant data",
#     online_enabled=True
# )

# # --- STEP 3: Insert data into Feature Store ---
# feature_group.insert(df, write_options={"wait_for_job": True})

# print("✅ Data successfully stored in Hopsworks Feature Store!")



import os
import hopsworks
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- STEP 0: Load API keys and connect to Hopsworks ---
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")   # Make sure this is in your .env file
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# --- STEP 1: Set Karachi coordinates ---
LAT = "24.8607"
LON = "67.0011"

# --- STEP 2: Set time range (last 24 hours) ---
end_date = datetime.utcnow()
start_date = end_date - timedelta(hours=24)

# --- STEP 3: Fetch latest AQI data from OpenWeather ---
url = f"http://api.openweathermap.org/data/2.5/air_pollution/history"
params = {
    "lat": LAT,
    "lon": LON,
    "start": int(start_date.timestamp()),
    "end": int(end_date.timestamp()),
    "appid": API_KEY
}

response = requests.get(url, params=params)
data = response.json().get("list", [])

if not data:
    print("⚠️ No new AQI data found for Karachi.")
    exit()

# --- STEP 4: Convert to DataFrame ---
rows = []
for record in data:
    rows.append({
        "datetime": datetime.utcfromtimestamp(record["dt"]),
        "aqi": record["main"]["aqi"],
        "pm2_5": record["components"].get("pm2_5", None),
        "pm10": record["components"].get("pm10", None),
        "co": record["components"].get("co", None),
        "no2": record["components"].get("no2", None),
        "o3": record["components"].get("o3", None)
    })

df_new = pd.DataFrame(rows)

# --- STEP 5: Prepare for insertion ---
df_new["datetime_str"] = df_new["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
df_new = df_new.sort_values("datetime")

# --- STEP 6: Get your existing feature group ---
feature_group = fs.get_feature_group("aqi_features", version=1)

# --- STEP 7: Insert new records ---
feature_group.insert(df_new, write_options={"wait_for_job": True})

print(f"✅ Successfully added {len(df_new)} new records for Karachi to Hopsworks!")



#new
# # features_group.py
# import os
# import hopsworks
# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# from dotenv import load_dotenv

# # --- STEP 0: Load API keys and connect to Hopsworks ---
# load_dotenv()
# API_KEY = os.getenv("OPENWEATHER_API_KEY")   # Make sure this is in your .env file
# project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# fs = project.get_feature_store()

# # --- STEP 1: Set Karachi coordinates ---
# LAT = "24.8607"
# LON = "67.0011"

# # --- STEP 2: Set time range (last 24 hours) ---
# end_date = datetime.utcnow()
# start_date = end_date - timedelta(hours=24)

# # --- STEP 3: Fetch latest AQI data from OpenWeather ---
# url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
# params = {
#     "lat": LAT,
#     "lon": LON,
#     "start": int(start_date.timestamp()),
#     "end": int(end_date.timestamp()),
#     "appid": API_KEY
# }

# response = requests.get(url, params=params)
# data = response.json().get("list", [])

# if not data:
#     print("⚠️ No new AQI data found for Karachi.")
#     exit()

# # --- STEP 4: Convert to DataFrame ---
# rows = []
# for record in data:
#     rows.append({
#         "datetime": datetime.utcfromtimestamp(record["dt"]),
#         "aqi": record["main"]["aqi"],
#         "pm2_5": record["components"].get("pm2_5", None),
#         "pm10": record["components"].get("pm10", None),
#         "co": record["components"].get("co", None),
#         "no2": record["components"].get("no2", None),
#         "o3": record["components"].get("o3", None)
#     })

# df_new = pd.DataFrame(rows)

# # --- STEP 5: Feature Engineering Functions ---
# def add_time_features(df, ts_col="datetime"):
#     df[ts_col] = pd.to_datetime(df[ts_col])
#     df["hour"] = df[ts_col].dt.hour
#     df["dayofweek"] = df[ts_col].dt.dayofweek
#     df["is_weekend"] = df["dayofweek"] >= 5
#     df["month"] = df[ts_col].dt.month
#     return df

# def add_trend_features(df, target_col="aqi"):
#     df = df.sort_values("datetime")
#     df[f"{target_col}_diff"] = df[target_col].diff()
#     df[f"{target_col}_rolling_3h"] = df[target_col].rolling(window=3, min_periods=1).mean()
#     df[f"{target_col}_roc_24h"] = (df[target_col] - df[target_col].shift(24)) / 24
#     return df

# def compute_all_features(df):
#     df = add_time_features(df)
#     if "aqi" in df.columns:
#         df = add_trend_features(df, "aqi")
#     if "pm2_5" in df.columns and "pm10" in df.columns:
#         df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
#     df = df.fillna(method="ffill").fillna(0)
#     return df

# # --- STEP 6: Prepare DataFrame for insertion ---
# df_new = df_new.sort_values("datetime")
# df_new = compute_all_features(df_new)
# df_new["datetime_str"] = df_new["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

# # --- STEP 7: Get your existing feature group ---
# feature_group = fs.get_feature_group("aqi_features", version=1)

# # --- STEP 8: Insert new records ---
# feature_group.insert(df_new, write_options={"wait_for_job": True})

# print(f"✅ Successfully added {len(df_new)} new records for Karachi to Hopsworks!")





#here
import os
import hopsworks
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- Load environment and connect ---
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# --- Coordinates for Karachi ---
LAT = "24.8607"
LON = "67.0011"

# --- Time range: last 24 hours ---
end_date = datetime.utcnow()
start_date = end_date - timedelta(hours=24)

# --- Fetch data ---
url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
params = {"lat": LAT, "lon": LON, "start": int(start_date.timestamp()), "end": int(end_date.timestamp()), "appid": API_KEY}
response = requests.get(url, params=params)
data = response.json().get("list", [])
if not data:
    print("⚠️ No new AQI data found for Karachi.")
    exit()

# --- Convert to DataFrame ---
rows = []
for record in data:
    rows.append({
        "datetime": datetime.utcfromtimestamp(record["dt"]),
        "aqi": record["main"]["aqi"],
        "pm2_5": record["components"].get("pm2_5"),
        "pm10": record["components"].get("pm10"),
        "co": record["components"].get("co"),
        "no2": record["components"].get("no2"),
        "o3": record["components"].get("o3"),
    })
df_new = pd.DataFrame(rows)
df_new = df_new.sort_values("datetime")

# --- Feature engineering ---
df_new["hour"] = df_new["datetime"].dt.hour
df_new["dayofweek"] = df_new["datetime"].dt.dayofweek
df_new["is_weekend"] = df_new["dayofweek"].isin([5, 6])
df_new["month"] = df_new["datetime"].dt.month
df_new["aqi_diff"] = df_new["aqi"].diff().fillna(0)
df_new["aqi_rolling_3h"] = df_new["aqi"].rolling(window=3, min_periods=1).mean()
df_new["aqi_roc_24h"] = df_new["aqi"].pct_change(periods=24).fillna(0)
df_new["pm25_pm10_ratio"] = df_new["pm2_5"] / df_new["pm10"]

# --- Add string version of datetime for PK ---
df_new["datetime_str"] = df_new["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

# --- Create NEW Feature Group version ---
feature_group = fs.create_feature_group(
    name="aqi_features",
    version=2,  # New version
    primary_key=["datetime_str"],  # ✅ string primary key works
    description="AQI features with engineered columns (time + trends)",
    online_enabled=True,
)

# --- Insert data ---
feature_group.insert(df_new, write_options={"wait_for_job": True})
print(f"✅ Created Feature Group v2 with {len(df_new)} records successfully inserted.")
