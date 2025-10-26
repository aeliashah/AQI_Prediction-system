# src/hopsworks_utils.py
import hopsworks
import os
from dotenv import load_dotenv
load_dotenv()

def login_project():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    return project, fs

def get_feature_group(fs, name="aqi_features", version=1, primary_key=["city"], event_time="event_timestamp", description="AQI features"):
    # Try to get existing FG, else create
    try:
        fg = fs.get_feature_group(name=name, version=version)
    except Exception:
        fg = fs.create_feature_group(name=name, version=version, primary_key=primary_key, event_time=event_time, description=description)
    return fg
