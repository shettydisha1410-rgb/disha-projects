# AQI Research Project - Data Cleaning
import pandas as pd
import numpy as np

df = pd.read_csv("delhi_ncr_aqi_dataset.csv")
print("Original Shape:", df.shape)
df.columns = df.columns.str.strip().str.lower()
df.replace(["-", "N/A", "na", "null", ""], np.nan, inplace=True)

num_cols = ["aqi", "pm2.5", "pm10", "o3", "no2", "so2", "co"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

time_cols = ["time", "datetime", "date", "timestamp"]
for col in time_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

df = df.drop_duplicates()
num_cols_present = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols_present] = df[num_cols_present].fillna(df[num_cols_present].median())
cat_cols = df.select_dtypes(include=["object"]).columns
if len(cat_cols):
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
if "aqi" in df.columns:
    df = df.dropna(subset=["aqi"])
for col in time_cols:
    if col in df.columns:
        df = df.dropna(subset=[col])

df.to_csv("cleaned_dataset.csv", index=False)
print("Cleaning complete. Final Shape:", df.shape)
