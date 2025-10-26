import os
import hopsworks
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

# --- STEP 1: Connect to Hopsworks ---
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# --- STEP 2: Load data from Feature Store ---
feature_group = fs.get_feature_group(name="aqi_features", version=1)
df = feature_group.read()
df = df.sort_values("datetime")

print(f"✅ Loaded {len(df)} rows for evaluation")

# --- STEP 3: Prepare data ---
X = df.drop(columns=["AQI", "datetime", "datetime_str"], errors="ignore")
y = df["AQI"]

# Split by time (80% train, 20% test)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# --- STEP 4: Load model from Model Registry ---
mr = project.get_model_registry()
model = mr.get_model("aqi_forecast_randomforest", version=1)
model_dir = model.download()
model_path = os.path.join(model_dir, "RandomForest_aqi_model.pkl")
rf_model = joblib.load(model_path)

print("✅ Model loaded from Hopsworks")

# --- STEP 5: Evaluate model ---
y_pred = rf_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R²:   {r2:.3f}")

# --- STEP 6: Visualization ---
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual AQI", color="blue")
plt.plot(y_pred, label="Predicted AQI", color="red", linestyle="--")
plt.title("Actual vs Predicted AQI (Test Set)")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Evaluation complete!")
