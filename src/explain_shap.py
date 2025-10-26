# src/explain_shap.py
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def global_shap_summary(X_sample_path="models/X_sample.csv"):
    model = joblib.load("models/aqi_rf.pkl")
    X_sample = pd.read_csv(X_sample_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    os.makedirs("artifacts/shap", exist_ok=True)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("artifacts/shap/summary.png", bbox_inches="tight")
    plt.close()
    print("Saved SHAP summary to artifacts/shap/summary.png")

def local_shap_for_input(input_df):
    model = joblib.load("models/aqi_rf.pkl")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    return shap_values
