import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values("timestamp").reset_index(drop=True)

# Time-based features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["weekday"] = df["timestamp"].dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

print("\n=== Time Features Added ===")
print(df[["timestamp", "hour", "day", "weekday", "is_weekend"]].head(10))

# Lag features - capture past usage patterns
df["lag_1_usage"] = df["usage_units"].shift(1)
df["lag_7_usage"] = df["usage_units"].shift(7)

print("\n=== Lag Features Added ===")
print(df[["timestamp", "usage_units", "lag_1_usage", "lag_7_usage"]].head(10))

# Rolling mean - smooths noise and reveals trends
df["rolling_mean_3"] = df["usage_units"].rolling(window=3).mean()

print("\n=== Rolling Mean Added ===")
print(df[["timestamp", "usage_units", "rolling_mean_3"]].head(10))

# Spike detection - flag unusually high usage
threshold = df["usage_units"].mean() + df["usage_units"].std()
df["usage_spike"] = np.where(df["usage_units"] > threshold, 1, 0)

spike_count = df["usage_spike"].sum()
print(f"\n=== Spike Detection ===")
print(f"Threshold : {threshold:.2f}")
print(f"Spikes detected : {spike_count}")
print(df[["timestamp", "usage_units", "usage_spike"]].head(10))

# One-hot encoding - convert categorical to numeric
df = pd.get_dummies(df, columns=["region", "service_type"], drop_first=True)

print("\n=== Encoding Applied ===")
print("New columns after encoding:", [c for c in df.columns if c.startswith(("region_", "service_type_"))])

# Final validation
print("\n=== Final Dataset Info ===")
df.info()

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Missing Values After Feature Engineering ===")
print(df.isnull().sum())

# Visualization - usage over time with spike markers
plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["usage_units"], label="Usage Units", linewidth=0.8)
spikes = df[df["usage_spike"] == 1]
plt.scatter(spikes["timestamp"], spikes["usage_units"],
            color="red", s=10, label="Spike", zorder=5)
plt.axhline(threshold, color="orange", linestyle="--", label=f"Threshold ({threshold:.1f})")
plt.title("Azure Usage Demand – Spike Detection")
plt.xlabel("Timestamp")
plt.ylabel("Usage Units")
plt.legend()
plt.tight_layout()
plt.show()
