# src/serve_api.py
from flask import Flask, request, jsonify
import joblib, os
from dotenv import load_dotenv
load_dotenv()

from src.hopsworks_utils import login_project
from datetime import datetime
import pandas as pd

app = Flask(__name__)
# load model & scaler once
MODEL_PATH = "models/aqi_rf.pkl"
SCALER_PATH = "models/scaler.joblib"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    city = payload.get("city")
    as_of = payload.get("as_of")  # ISO string, optional
    if not as_of:
        as_of = datetime.utcnow().isoformat()
    # get features from Hopsworks online store (last available)
    project, fs = login_project()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    # reading batch by timeframe is Hopsworks-specific; we use get_batch or query
    df = fg.read()  # simple approach: read whole FG and filter
    df = df.sort_values("event_timestamp")
    df_city = df[df["city"]==city].copy()
    if df_city.empty:
        return jsonify({"error":"no features for city"}), 404
    latest_row = df_city.tail(1)
    X = latest_row.drop(columns=["city","event_timestamp","aqi"])
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    return jsonify({"city":city, "predicted_aqi":float(pred), "as_of":str(latest_row["event_timestamp"].values[0])})

if __name__ == "__main__":
    app.run(debug=True, port=8080)
