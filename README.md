Karachi Real-Time Air Quality Forecast System
AI-Powered Dashboard using Hopsworks Feature Store, Streamlit & Multiple ML Models
📌 Overview

This project predicts real-time Air Quality Index (AQI) for Karachi using a full MLOps pipeline.
It streams live air-quality data, stores features in Hopsworks, trains multiple ML models, compares them, and deploys an interactive dashboard for real-time forecasting.

The system uses machine learning, deep learning, time-series forecasting, and model registry automation.

🎯 Objectives

Collect & stream Karachi AQI data

Build Hopsworks Feature Store pipeline

Train & compare 5 models

Automatically select best performing model

Deploy Streamlit real-time dashboard

Visualize AQI predictions and trends

Maintain feature & model versioning

🏗️ System Architecture
Live AQI Data → Hopsworks Feature Store → Model Training
                                     ↓
                          Model Registry + Metrics
                                     ↓
                        Streamlit Dashboard (Real-Time)

🤖 Models Implemented
Model	Category	Purpose
Random Forest	ML	Baseline tree model
XGBoost	ML	Gradient boosting
LightGBM	ML	Fast scalable boosting
ARIMA	Time-Series	Classical baseline
LSTM	Deep Learning	Sequential AQI forecasting
📊 Final Model Performance
Metric	Score
RMSE	0.0070
MAE	0.0023
R² Score	0.9999

✅RGBOOSTis the best model
✅ Very strong prediction accuracy

🧾 Features Created
Type	Features
Raw	AQI, PM2.5, PM10, CO, NO2, O3
Time-based	hour, dayofweek, month, weekend flag
Engineered	rolling averages, differences, PM ratio

Total 16 engineered features used for prediction.

🗂️ Project Structure
src/
 ├── feature_pipeline.py   # Live data → Hopsworks
 ├── train_model.py        # Train & register models
 ├── predict.py            # Model inference
 └── utils.py

streamlit_app/
 └── app.py                # Real-time dashboard

model_registry/            # Saved models
data/                      

⚙️ Installation & Run
Install dependencies
pip install -r requirements.txt

Start feature pipeline
python src/feature_pipeline.py

Train all models
python src/train_model.py

Run dashboard
streamlit run streamlit_app/app.py

🖥️ Dashboard Features

Real-time AQI display & chart

ML vs Deep Learning comparison

Dropdown to manually choose model

Auto-fetch best model from Hopsworks

Forecast graph (future AQI)

🧠 MLOps Components

Hopsworks Feature Group: aqi_features_karachi

Model Registry with metric tracking

Automatic versioning

Online & Offline store integration

📬 Author

Aelia Taskeen Bibi

