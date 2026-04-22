import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/tft/tft_ready_data.parquet"
OUTPUT_DIR = "data/processed/tft"
SANITIZED_DATA_FILE = os.path.join(OUTPUT_DIR, "tft_ready_data_sanitized.parquet")

def safe_imputation(df):
    print("Applying SAFE 15-minute imputation strategy...")

    df = df.sort_values(by=["station", "time_idx"]).reset_index(drop=True)

    target_cols = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']
    advection_cols = [f'adv_in_{col}' for col in target_cols]
    wind_cols = ['wind_x', 'wind_y']

    def fill_group(group):
        # Forward fill small sensor drops (limit=3 means max 45 minutes of ffill)
        group[target_cols] = group[target_cols].ffill(limit=3)
        group[wind_cols] = group[wind_cols].ffill(limit=6)
        group[advection_cols] = group[advection_cols].ffill(limit=6)
        return group

    # Apply the safe filling
    df = df.groupby("station", group_keys=False).apply(fill_group)

    # CRITICAL FIX: We DO NOT fillna(0) on pollutants! 
    # If a pollutant is still NaN, it means the sensor was truly offline.
    # We leave it as NaN so the downstream 'fix_station_time_range.py' can trim it correctly.
    
    # We only fillna(0) on advection features because missing wind math equals 0 transport
    df[advection_cols] = df[advection_cols].fillna(0.0)

    return df


def sanitize_dataset():
    print("INITIALIZING DATA SANITIZATION...\n")

    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    # -------------------------------
    # COLUMN SANITIZATION
    # -------------------------------
    print("Sanitizing column names...")
    df.columns = [
        c.replace('.', '_')
         .replace('/', '_')
         .replace('µ', 'u')
         .replace('³', '3')
        for c in df.columns
    ]

    # -------------------------------
    # SAFE IMPUTATION
    # -------------------------------
    df = safe_imputation(df)

    # -------------------------------
    # SAVE CLEAN DATA
    # -------------------------------
    print(f"Saving sanitized dataset to {SANITIZED_DATA_FILE}...")
    df.to_parquet(SANITIZED_DATA_FILE, index=False)

    print("\nSUCCESS: SANITIZED DATASET BUILT. (Legacy TFT objects removed).")


if __name__ == "__main__":
    sanitize_dataset()