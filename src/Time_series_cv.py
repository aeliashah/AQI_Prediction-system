# src/time_series_cv.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib

# --- Load the prepared dataset you trained on ---
# (Use the same cleaned feature file or pull from feature store)
df = pd.read_csv("data/processed/aqi_features.csv")

# --- Prepare features & target ---
X = df.drop(columns=["AQI"])   # Replace AQI with your actual target column
y = df["AQI"]

# --- Define time-series cross-validation ---
tscv = TimeSeriesSplit(n_splits=5)

model = RandomForestRegressor(n_estimators=200, random_state=42)

rmse_scores = []
fold = 1
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold}: RMSE = {rmse:.3f}")
    fold += 1

print("\n✅ Cross-validated RMSEs:", rmse_scores)
print(f"📊 Average RMSE across folds: {np.mean(rmse_scores):.3f}")

# Optional: save model from the last fold
joblib.dump(model, "models/random_forest_cv.pkl")
