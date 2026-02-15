import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
print(df.head())

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

df['region'] = (
    df['region']
    .str.strip()
    .str.lower()
    .str.replace(" ", "-")
)

df['region'] = df['region'].replace({
    'us-east': 'US-East',
    'us-west': 'US-West',
    'india-south': 'India-South'
})

df = df.drop_duplicates()

df['usage_units'] = df['usage_units'].interpolate()

df['provisioned_capacity'] = df['provisioned_capacity'].interpolate()

df['cost_usd'] = df['cost_usd'].fillna(df['usage_units'] * 0.1)

df['availability_pct'] = df['availability_pct'].ffill()

df['is_holiday'] = df['is_holiday'].fillna(0)

df['economic_index'] = df['economic_index'].interpolate()

df['market_demand_index'] = df['market_demand_index'].fillna(
    df['market_demand_index'].rolling(7, min_periods=1).mean()
)

df['temperature_index'] = df['temperature_index'].interpolate()

print("\nMissing values after cleaning:\n")
print(df.isnull().sum())

plt.figure(figsize=(12,6))
for region in df['region'].unique():
    subset = df[df['region']==region]
    plt.plot(subset['timestamp'], subset['usage_units'], label=region)

plt.title("Azure Usage Demand by Region")
plt.legend()
plt.show()
