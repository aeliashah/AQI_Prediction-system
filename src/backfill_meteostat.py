# src/backfill_meteostat.py
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

from store_features import compute_all_features
from src.hopsworks_utils import login_project, get_feature_group

CITY = os.getenv("CITY")
LAT = float(os.getenv("LAT"))
LON = float(os.getenv("LON"))

def fetch_meteostat_history(lat, lon, days=365):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    loc = Point(lat, lon)
    data = Hourly(loc, start, end)
    df = data.fetch().reset_index()
    # rename columns to canonical names
    df = df.rename(columns={
        "time":"event_timestamp",
        "temp":"temperature",
        "rhum":"humidity",
        "wspd":"wind_speed",
        "pres":"pressure"
    })
    df["city"] = CITY
    # If no AQI in this dataset, keep aqi null and compute features accordingly
    return df

def backfill_to_hopsworks():
    df = fetch_meteostat_history(LAT, LON, days=365)
    # If you have separate AQI source, merge on timestamp here
    df = compute_all_features(df)
    project, fs = login_project()
    fg = get_feature_group(fs)
    # insert in batches
    fg.insert(df, overwrite=False)
    print("Backfill complete, inserted rows:", len(df))

if __name__ == "__main__":
    backfill_to_hopsworks()
