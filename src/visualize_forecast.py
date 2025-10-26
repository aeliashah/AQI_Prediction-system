# src/visualize_forecast.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ============================
# Step 1: Load data
# ============================
hist_path = "data/karachi_aqi_historical_1year.csv"
forecast_path = "data/karachi_aqi_forecast.csv"

if not os.path.exists(hist_path) or not os.path.exists(forecast_path):
    raise FileNotFoundError("❌ Missing historical or forecast CSV file in data/ directory.")

df_hist = pd.read_csv(hist_path, parse_dates=["datetime"])
df_forecast = pd.read_csv(forecast_path, parse_dates=["datetime"])

# ============================
# Step 2: Prepare data
# ============================
df_hist = df_hist.sort_values("datetime")
df_forecast = df_forecast.sort_values("datetime")

# Get last few days of history for context
df_recent = df_hist.tail(7 * 24)  # last 7 days (hourly data)

# ============================
# Step 3: Plot comparison
# ============================
plt.figure(figsize=(12, 6))
plt.plot(df_recent["datetime"], df_recent["aqi"], label="Historical AQI", color="skyblue", linewidth=2)
plt.plot(df_forecast["datetime"], df_forecast["predicted_aqi"], label="Predicted AQI (Next 3 Days)", color="orange", linewidth=2, linestyle="--")

plt.title("🌤️ Karachi Air Quality Index — Past 7 Days & 3-Day Forecast", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("AQI (Index Value)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Format x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)

# Save & show
output_path = "data/karachi_aqi_visualization.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"✅ Visualization saved to {output_path}")
