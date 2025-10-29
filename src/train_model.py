# # # # # src/train_model.py
# # # # import joblib, os, json
# # # # from sklearn.ensemble import RandomForestRegressor
# # # # from sklearn.linear_model import Ridge
# # # # from sklearn.preprocessing import StandardScaler
# # # # from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# # # # from sklearn.model_selection import train_test_split
# # # # from dotenv import load_dotenv
# # # # load_dotenv()

# # # # from src.hopsworks_utils import login_project, get_feature_group

# # # # def load_data_from_hopsworks():
# # # #     project, fs = login_project()
# # # #     fg = fs.get_feature_group(name="aqi_features", version=1)
# # # #     df = fg.read()  # read whole FG into dataframe
# # # #     return df

# # # # def prepare_features(df):
# # # #     # Drop rows with no target
# # # #     df = df.dropna(subset=["aqi"])
# # # #     # sort by time
# # # #     df = df.sort_values("event_timestamp")
# # # #     # select features (exclude keys)
# # # #     drop_cols = ["city", "event_timestamp"]
# # # #     X = df.drop(columns=drop_cols + ["aqi"])
# # # #     y = df["aqi"].astype(float)
# # # #     return X, y

# # # # def train_and_register():
# # # #     df = load_data_from_hopsworks()
# # # #     X, y = prepare_features(df)
# # # #     # time-based split: last 20% as test
# # # #     n = len(X)
# # # #     split = int(n * 0.8)
# # # #     X_train, X_test = X.iloc[:split], X.iloc[split:]
# # # #     y_train, y_test = y.iloc[:split], y.iloc[split:]

# # # #     scaler = StandardScaler()
# # # #     X_train_s = scaler.fit_transform(X_train)
# # # #     X_test_s = scaler.transform(X_test)

# # # #     rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
# # # #     rf.fit(X_train_s, y_train)
# # # #     preds = rf.predict(X_test_s)

# # # #     rmse = mean_squared_error(y_test, preds, squared=False)
# # # #     mae = mean_absolute_error(y_test, preds)
# # # #     r2 = r2_score(y_test, preds)
# # # #     print("RF metrics:", rmse, mae, r2)

# # # #     # Save artifacts
# # # #     os.makedirs("models", exist_ok=True)
# # # #     joblib.dump(rf, "models/aqi_rf.pkl")
# # # #     joblib.dump(scaler, "models/scaler.joblib")
# # # #     meta = {"model":"rf","rmse":rmse,"mae":mae,"r2":r2}
# # # #     with open("models/metadata.json","w") as f:
# # # #         json.dump(meta, f)

# # # #     # Register model to Hopsworks model registry
# # # #     project, fs = login_project()
# # # #     mr = project.get_model_registry()
# # # #     # create model + save (this API may vary by hopsworks version)
# # # #     model = mr.python.create_model(name="aqi_rf", metrics=meta, model_dir="models")
# # # #     model.save()
# # # #     print("Model registered to Hopsworks model registry")

# # # # if __name__ == "__main__":
# # # #     train_and_register()



# # # # src/train_model.py
# # # import os
# # # import hopsworks
# # # import pandas as pd
# # # import numpy as np
# # # from dotenv import load_dotenv
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.metrics import mean_absolute_error, mean_squared_error
# # # from sklearn.linear_model import LinearRegression
# # # from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# # # import xgboost as xgb
# # # import joblib
# # # import math

# # # # --- STEP 1: Connect to Hopsworks ---
# # # load_dotenv()
# # # project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# # # fs = project.get_feature_store()

# # # # --- STEP 2: Load data from feature store ---
# # # feature_group = fs.get_feature_group(name="aqi_features", version=1)
# # # df = feature_group.read()
# # # print(f"✅ Loaded {len(df)} rows from Hopsworks Feature Store")

# # # # --- STEP 3: Data preparation ---
# # # df = df.dropna(subset=["aqi"])  # remove rows without target
# # # df = df.sort_values("datetime_str")

# # # # Feature engineering: use pollutants as features
# # # features = ["pm2_5", "pm10", "co", "no2", "o3"]
# # # target = "aqi"

# # # X = df[features]
# # # y = df[target]

# # # # Train-test split
# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # # # --- STEP 4: Define models ---
# # # models = {
# # #     "LinearRegression": LinearRegression(),
# # #     "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
# # #     "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
# # #     "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
# # # }

# # # results = {}

# # # # --- STEP 5: Train & evaluate models ---
# # # for name, model in models.items():
# # #     model.fit(X_train, y_train)
# # #     y_pred = model.predict(X_test)
# # #     rmse = math.sqrt(mean_squared_error(y_test, y_pred))
# # #     mae = mean_absolute_error(y_test, y_pred)
# # #     results[name] = {"RMSE": rmse, "MAE": mae}
# # #     print(f"📊 {name}: RMSE={rmse:.3f}, MAE={mae:.3f}")

# # # # --- STEP 6: Pick best model ---
# # # best_model_name = min(results, key=lambda x: results[x]["RMSE"])
# # # best_model = models[best_model_name]
# # # print(f"\n🏆 Best Model: {best_model_name} (RMSE={results[best_model_name]['RMSE']:.3f})")

# # # # --- STEP 7: Save model locally ---
# # # os.makedirs("models", exist_ok=True)
# # # model_path = f"models/{best_model_name}_aqi_model.pkl"
# # # joblib.dump(best_model, model_path)

# # # # --- STEP 8: Register model in Hopsworks ---
# # # mr = project.get_model_registry()
# # # model_meta = mr.python.create_model(
# # #     name=f"aqi_forecast_{best_model_name.lower()}",
# # #     metrics=results[best_model_name],
# # #     description=f"AQI prediction model trained on Karachi data ({best_model_name})"
# # # )

# # # model_meta.save(model_path)
# # # print(f"✅ Model '{best_model_name}' registered in Hopsworks Model Registry!")

# # # print("\n🎉 Training pipeline complete!")
# # # print("Model performance summary:")
# # # for name, metrics in results.items():
# # #     print(f"  - {name}: RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}")




# # # src/train_model.py
# # import os
# # import hopsworks
# # import pandas as pd
# # import numpy as np
# # from dotenv import load_dotenv
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import mean_absolute_error, mean_squared_error
# # from sklearn.linear_model import LinearRegression
# # from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# # import xgboost as xgb
# # import joblib
# # import math
# # import matplotlib.pyplot as plt

# # # --- STEP 1: Connect to Hopsworks ---
# # load_dotenv()
# # project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# # fs = project.get_feature_store()

# # # --- STEP 2: Load data from feature store ---
# # feature_group = fs.get_feature_group(name="aqi_features", version=1)
# # df = feature_group.read()
# # print(f"✅ Loaded {len(df)} rows from Hopsworks Feature Store")

# # # --- STEP 3: Data preparation ---
# # df = df.dropna(subset=["aqi"])  # remove rows without target
# # df = df.sort_values("datetime_str")

# # # Features and target
# # features = ["pm2_5", "pm10", "co", "no2", "o3"]
# # target = "aqi"

# # X = df[features]
# # y = df[target]

# # # Split chronologically (no shuffle!)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # # --- STEP 4: Define models ---
# # models = {
# #     "LinearRegression": LinearRegression(),
# #     "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
# #     "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
# #     "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
# # }

# # results = {}

# # # --- STEP 5: Train, evaluate & check overfitting ---
# # for name, model in models.items():
# #     model.fit(X_train, y_train)

# #     # Predictions
# #     y_train_pred = model.predict(X_train)
# #     y_test_pred = model.predict(X_test)

# #     # Metrics
# #     train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
# #     test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
# #     mae = mean_absolute_error(y_test, y_test_pred)

# #     # Save metrics
# #     results[name] = {"Train_RMSE": train_rmse, "Test_RMSE": test_rmse, "MAE": mae}

# #     print(f"\n📊 {name}")
# #     print(f"   Train RMSE={train_rmse:.3f}, Test RMSE={test_rmse:.3f}, MAE={mae:.3f}")
# #     if train_rmse < test_rmse / 2:
# #         print("⚠️ Potential overfitting detected!")

# #     # Plot actual vs predicted for test set
# #     plt.figure(figsize=(8, 4))
# #     plt.plot(y_test.values, label="Actual AQI", color="blue")
# #     plt.plot(y_test_pred, label="Predicted AQI", color="red", linestyle="--")
# #     plt.title(f"{name}: Actual vs Predicted AQI (Test Set)")
# #     plt.xlabel("Time")
# #     plt.ylabel("AQI")
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.show()

# # # --- STEP 6: Select best model ---
# # best_model_name = min(results, key=lambda x: results[x]["Test_RMSE"])
# # best_model = models[best_model_name]
# # print(f"\n🏆 Best Model: {best_model_name} (Test RMSE={results[best_model_name]['Test_RMSE']:.3f})")

# # # --- STEP 7: Save best model locally ---
# # os.makedirs("models", exist_ok=True)
# # model_path = f"models/{best_model_name}_aqi_model.pkl"
# # joblib.dump(best_model, model_path)

# # # --- STEP 8: Register model in Hopsworks ---
# # mr = project.get_model_registry()
# # model_meta = mr.python.create_model(
# #     name=f"aqi_forecast_{best_model_name.lower()}",
# #     metrics=results[best_model_name],
# #     description=f"AQI prediction model trained on Karachi data ({best_model_name})"
# # )
# # model_meta.save(model_path)
# # print(f"✅ Model '{best_model_name}' registered in Hopsworks Model Registry!")

# # # --- STEP 9: Print summary ---
# # print("\n🎉 Training pipeline complete!")
# # print("Model performance summary:")
# # for name, metrics in results.items():
# #     print(f"  - {name}: Train RMSE={metrics['Train_RMSE']:.3f}, Test RMSE={metrics['Test_RMSE']:.3f}, MAE={metrics['MAE']:.3f}")


# import os
# import hopsworks
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import joblib

# # --- STEP 1: Load API key and connect to Hopsworks ---
# load_dotenv()
# project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# fs = project.get_feature_store()

# # --- STEP 2: Load data from Feature Group version 2 ---
# print("📥 Loading data from Hopsworks Feature Store (version 2)...")
# feature_group = fs.get_feature_group("aqi_features", version=2)
# df = feature_group.read()
# print(f"✅ Loaded {len(df)} records from aqi_features v2")

# # --- STEP 3: Prepare data for model training ---
# df = df.sort_values("datetime")

# # Drop unnecessary columns (keep numeric + useful time features)
# X = df[[
#     "hour", "dayofweek", "is_weekend", "month",
#     "pm2_5", "pm10", "co", "no2", "o3",
#     "aqi_diff", "aqi_rolling_3h", "aqi_roc_24h", "pm25_pm10_ratio"
# ]]
# y = df["aqi"]

# # Handle missing or inf values
# X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # --- STEP 4: Train models ---
# models = {
#     "LinearRegression": LinearRegression(),
#     "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
#     "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
#     "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
# }

# results = {}

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     rmse = mean_squared_error(y_test, y_pred, squared=False)
#     mae = mean_absolute_error(y_test, y_pred)
#     results[name] = {"RMSE": rmse, "MAE": mae}
#     print(f"📊 {name}: RMSE={rmse:.2f}, MAE={mae:.2f}")

# # --- STEP 5: Select best model ---
# best_model_name = min(results, key=lambda k: results[k]["RMSE"])
# best_rmse = results[best_model_name]["RMSE"]
# best_mae = results[best_model_name]["MAE"]
# best_model = models[best_model_name]

# print(f"\n🏆 Best Model: {best_model_name}")
# print(f"   RMSE={best_rmse:.2f}, MAE={best_mae:.2f}")

# # --- STEP 6: Save and register model to Hopsworks ---
# model_dir = "models"
# os.makedirs(model_dir, exist_ok=True)
# model_path = f"{model_dir}/{best_model_name}_aqi_model.pkl"
# joblib.dump(best_model, model_path)

# mr = project.get_model_registry()
# model_meta = mr.python.create_model(
#     name="AQI_Predictor",
#     metrics={"RMSE": best_rmse, "MAE": best_mae},
#     description=f"{best_model_name} model trained on aqi_features v2"
# )
# model_meta.save(model_path)
# model_meta.register()

# print(f"✅ Model '{best_model_name}' logged to Hopsworks successfully!")


#here
# import os
# import hopsworks
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression, Ridge
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# #import tensorflow as tf
# import shap

# # =============================
# # STEP 1: Setup & Login
# # =============================
# load_dotenv()
# project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# fs = project.get_feature_store()

# # =============================
# # STEP 2: Load data from Feature Store
# # =============================
# print("📥 Loading data from Hopsworks Feature Store (version 2)...")
# feature_group = fs.get_feature_group("aqi_features", version=2)
# df = feature_group.read()
# print(f"✅ Loaded {len(df)} records from aqi_features v2")

# # =============================
# # STEP 3: Prepare dataset
# # =============================
# df = df.sort_values("datetime")

# X = df[[
#     "hour", "dayofweek", "is_weekend", "month",
#     "pm2_5", "pm10", "co", "no2", "o3",
#     "aqi_diff", "aqi_rolling_3h", "aqi_roc_24h", "pm25_pm10_ratio"
# ]]
# y = df["aqi"]

# X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, shuffle=False
# )

# # =============================
# # STEP 4: Train multiple models
# # =============================
# print("\n🚀 Training models...")

# models = {
#     "LinearRegression": LinearRegression(),
#     "RidgeRegression": Ridge(alpha=1.0),
#     "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
#     "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
#     "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
# }

# results = {}

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     rmse = mean_squared_error(y_test, y_pred, squared=False)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
#     print(f"📊 {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

# # =============================
# # STEP 5: Train TensorFlow (Deep Learning) model
# # =============================
# print("\n🧠 Training TensorFlow Neural Network...")

# tf_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# tf_model.compile(optimizer='adam', loss='mse')
# tf_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

# y_pred_tf = tf_model.predict(X_test).flatten()
# rmse_tf = mean_squared_error(y_test, y_pred_tf, squared=False)
# mae_tf = mean_absolute_error(y_test, y_pred_tf)
# r2_tf = r2_score(y_test, y_pred_tf)

# results["TensorFlowNN"] = {"RMSE": rmse_tf, "MAE": mae_tf, "R2": r2_tf}
# print(f"📊 TensorFlowNN: RMSE={rmse_tf:.2f}, MAE={mae_tf:.2f}, R2={r2_tf:.3f}")

# # =============================
# # STEP 6: Select best model
# # =============================
# best_model_name = min(results, key=lambda k: results[k]["RMSE"])
# best_model_metrics = results[best_model_name]

# print(f"\n🏆 Best Model: {best_model_name}")
# print(f"   RMSE={best_model_metrics['RMSE']:.2f}, MAE={best_model_metrics['MAE']:.2f}, R2={best_model_metrics['R2']:.3f}")

# # =============================
# # STEP 7: Save & Register Model
# # =============================
# model_dir = "models"
# os.makedirs(model_dir, exist_ok=True)
# model_path = f"{model_dir}/{best_model_name}_aqi_model.pkl"

# if best_model_name == "TensorFlowNN":
#     best_model = tf_model
#     model_path = f"{model_dir}/{best_model_name}_aqi_model.h5"
#     best_model.save(model_path)
# else:
#     best_model = models[best_model_name]
#     joblib.dump(best_model, model_path)

# mr = project.get_model_registry()
# model_meta = mr.python.create_model(
#     name="AQI_Predictor",
#     metrics=best_model_metrics,
#     description=f"{best_model_name} model trained on aqi_features v2"
# )
# model_meta.save(model_path)
# model_meta.register()

# print(f"✅ Model '{best_model_name}' logged to Hopsworks successfully!")

# # =============================
# # STEP 8: SHAP Feature Importance (for explainability)
# # =============================
# print("\n🔍 Computing SHAP feature importance (using RandomForest for baseline)...")

# try:
#     explainer = shap.TreeExplainer(models["RandomForest"])
#     shap_values = explainer.shap_values(X_test)

#     shap.summary_plot(shap_values, X_test, show=False)
#     print("✅ SHAP feature importance computed successfully.")
# except Exception as e:
#     print(f"⚠️ SHAP analysis skipped: {e}")

# # =============================
# # STEP 9: Summary Table
# # =============================
# results_df = pd.DataFrame(results).T.sort_values("RMSE")
# print("\n📈 Model Performance Summary:")
# print(results_df)




# import os
# import hopsworks
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression, Ridge
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# # import tensorflow as tf  # Disabled for now
# import shap
# from math import sqrt

# # =============================
# # STEP 1: Setup & Login
# # =============================
# load_dotenv()
# project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
# fs = project.get_feature_store()

# # =============================
# # STEP 2: Load data from Feature Store
# # =============================
# print("📥 Loading data from Hopsworks Feature Store (version 2)...")
# feature_group = fs.get_feature_group("aqi_features", version=2)
# df = feature_group.read()
# print(f"✅ Loaded {len(df)} records from aqi_features v2")

# # =============================
# # STEP 3: Prepare dataset
# # =============================
# df = df.sort_values("datetime")

# X = df[[
#     "hour", "dayofweek", "is_weekend", "month",
#     "pm2_5", "pm10", "co", "no2", "o3",
#     "aqi_diff", "aqi_rolling_3h", "aqi_roc_24h", "pm25_pm10_ratio"
# ]]
# y = df["aqi"]

# X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, shuffle=False
# )

# # =============================
# # STEP 4: Train multiple models
# # =============================
# print("\n🚀 Training models...")

# models = {
#     "LinearRegression": LinearRegression(),
#     "RidgeRegression": Ridge(alpha=1.0),
#     "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
#     "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
#     "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
# }

# results = {}

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     # --- Safe RMSE calculation ---
#     try:
#         rmse = mean_squared_error(y_test, y_pred, squared=False)
#     except TypeError:
#         rmse = sqrt(mean_squared_error(y_test, y_pred))

#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
#     print(f"📊 {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

# # =============================
# # STEP 5: (Optional) TensorFlow Neural Net
# # =============================
# """
# print("\n🧠 Training TensorFlow Neural Network...")

# tf_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# tf_model.compile(optimizer='adam', loss='mse')
# tf_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

# y_pred_tf = tf_model.predict(X_test).flatten()
# try:
#     rmse_tf = mean_squared_error(y_test, y_pred_tf, squared=False)
# except TypeError:
#     rmse_tf = sqrt(mean_squared_error(y_test, y_pred_tf))

# mae_tf = mean_absolute_error(y_test, y_pred_tf)
# r2_tf = r2_score(y_test, y_pred_tf)

# results["TensorFlowNN"] = {"RMSE": rmse_tf, "MAE": mae_tf, "R2": r2_tf}
# print(f"📊 TensorFlowNN: RMSE={rmse_tf:.2f}, MAE={mae_tf:.2f}, R2={r2_tf:.3f}")
# """

# # =============================
# # STEP 6: Select best model
# # =============================
# best_model_name = min(results, key=lambda k: results[k]["RMSE"])
# best_model_metrics = results[best_model_name]

# print(f"\n🏆 Best Model: {best_model_name}")
# print(f"   RMSE={best_model_metrics['RMSE']:.2f}, MAE={best_model_metrics['MAE']:.2f}, R2={best_model_metrics['R2']:.3f}")

# # =============================
# # STEP 7: Save & Register Model
# # =============================
# model_dir = "models"
# os.makedirs(model_dir, exist_ok=True)
# model_path = f"{model_dir}/{best_model_name}_aqi_model.pkl"

# best_model = models[best_model_name]
# joblib.dump(best_model, model_path)

# mr = project.get_model_registry()
# model_meta = mr.python.create_model(
#     name="AQI_Predictor",
#     metrics=best_model_metrics,
#     description=f"{best_model_name} model trained on aqi_features v2"
# )
# model_meta.save(model_path)
# model_meta.register()

# print(f"✅ Model '{best_model_name}' logged to Hopsworks successfully!")

# # =============================
# # STEP 8: SHAP Feature Importance
# # =============================
# print("\n🔍 Computing SHAP feature importance (using RandomForest for baseline)...")

# try:
#     explainer = shap.TreeExplainer(models["RandomForest"])
#     shap_values = explainer.shap_values(X_test)
#     shap.summary_plot(shap_values, X_test, show=False)
#     print("✅ SHAP feature importance computed successfully.")
# except Exception as e:
#     print(f"⚠️ SHAP analysis skipped: {e}")

# # =============================
# # STEP 9: Summary Table
# # =============================
# results_df = pd.DataFrame(results).T.sort_values("RMSE")
# print("\n📈 Model Performance Summary:")
# print(results_df)



import os
import hopsworks
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap
from math import sqrt

# =============================
# STEP 1: Setup & Login
# =============================
load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# =============================
# STEP 2: Load data from Feature Store (Karachi)
# =============================
print("📥 Loading Karachi AQI data from Hopsworks Feature Store (version 1)...")
feature_group = fs.get_feature_group("aqi_features_karachi", version=1)
df = feature_group.read()
print(f"✅ Loaded {len(df)} records from aqi_features_karachi v1")

# =============================
# STEP 3: Prepare dataset
# =============================
df = df.sort_values("datetime")

X = df[[
    "hour", "dayofweek", "is_weekend", "month",
    "pm2_5", "pm10", "co", "no2", "o3",
    "aqi_diff", "aqi_rolling_3h", "aqi_roc_24h", "pm25_pm10_ratio"
]]
y = df["aqi"]

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =============================
# STEP 4: Train multiple models
# =============================
print("\n🚀 Training models...")

models = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Safe RMSE calculation ---
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        rmse = sqrt(mean_squared_error(y_test, y_pred))

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"📊 {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

# =============================
# STEP 5: Select best model
# =============================
best_model_name = min(results, key=lambda k: results[k]["RMSE"])
best_model_metrics = results[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   RMSE={best_model_metrics['RMSE']:.2f}, MAE={best_model_metrics['MAE']:.2f}, R2={best_model_metrics['R2']:.3f}")

# =============================
# STEP 6: Save & Register Model
# =============================
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = f"{model_dir}/{best_model_name}_karachi_aqi_model.pkl"

best_model = models[best_model_name]
joblib.dump(best_model, model_path)

mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="Karachi_AQI_Predictor",
    metrics=best_model_metrics,
    description=f"{best_model_name} model trained on Karachi AQI data (v1)"
)
model_meta.save(model_path)

# 🛠️ Temporarily remove register() because it doesn’t exist anymore in hsfs>=3.8
# model_meta.register()

print(f"✅ Model '{best_model_name}' logged to Hopsworks successfully!")

# =============================
# STEP 7: SHAP Feature Importance
# =============================
print("\n🔍 Computing SHAP feature importance (using RandomForest for baseline)...")

try:
    explainer = shap.TreeExplainer(models["RandomForest"])
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    print("✅ SHAP feature importance computed successfully.")
except Exception as e:
    print(f"⚠️ SHAP analysis skipped: {e}")

# =============================
# STEP 8: Summary Table
# =============================
results_df = pd.DataFrame(results).T.sort_values("RMSE")
print("\n📈 Model Performance Summary:")
print(results_df)




#new

# import os
# import pandas as pd
# import numpy as np
# import hopsworks
# from dotenv import load_dotenv

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import math
# import json
# import joblib

# load_dotenv()

# # Load data
# df = pd.read_csv("data/processed/aqi_cleaned.csv")

# # Features & target
# X = df.drop(columns=["AQI"])
# y = df["AQI"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train model
# model = RandomForestRegressor(n_estimators=200, random_state=42)
# model.fit(X_train, y_train)

# # ============================
# # ✅ Evaluate model
# # ============================
# y_pred = model.predict(X_test)

# rmse = math.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("\n✅ Model Performance:")
# print(f"RMSE: {rmse:.2f}")
# print(f"MAE: {mae:.2f}")
# print(f"R² Score: {r2:.3f}")

# # Save metrics JSON
# metrics = {
#     "rmse": rmse,
#     "mae": mae,
#     "r2": r2
# }

# with open("model_metrics.json", "w") as f:
#     json.dump(metrics, f)
# print("📁 Metrics saved → model_metrics.json")

# # ===============================
# # ✅ Save model & scaler locally
# # ===============================
# os.makedirs("models", exist_ok=True)
# joblib.dump(model, "models/aqi_model.pkl")
# joblib.dump(scaler, "models/scaler.pkl")

# print("✅ Model & scaler saved locally")

# # ===============================
# # ✅ Upload to Hopsworks
# # ===============================
# project = hopsworks.login()
# mr = project.get_model_registry()

# model_dir = "models"
# model_meta = mr.python.create_model(
#     name="aqi_predictor_model",
#     metrics=metrics
# )

# model_meta.save(model_dir)
# print("🚀 Model & metrics registered in Hopsworks successfully!")
