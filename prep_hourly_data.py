import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/tft/tft_ready_data_sanitized.parquet"
OUTPUT_PATH = "data/processed/tft/tide_ready_1hr.parquet"

POLLUTANTS = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']
FEATURES = ['WS_m_s', 'wind_x', 'wind_y', 'adv_in_PM2_5_ug_m3', 'adv_in_PM10_ug_m3', 'adv_in_NO2_ug_m3', 'adv_in_SO2_ug_m3', 'adv_in_CO_mg_m3']
STATIC = ['latitude', 'longitude']

def build_hourly_data():
    print("Loading sanitized 15-min data...")
    df = pd.read_parquet(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # -------------------------------
    # 1. HOURLY RESAMPLING
    # -------------------------------
    print("Resampling to hourly grid...")
    df_hourly = (
        df.groupby(['city', 'station'])
          .resample('1h', on='timestamp')
          .mean(numeric_only=True)
          .reset_index()
    )

    KEEP_COLS = ['timestamp', 'city', 'station'] + POLLUTANTS + FEATURES + STATIC
    df_hourly = df_hourly[[c for c in KEEP_COLS if c in df_hourly.columns]]
    df_hourly['wind_observed'] = ((~df_hourly['wind_x'].isna()) | (~df_hourly['wind_y'].isna())).astype(int)

    # -------------------------------
    # 2. ENFORCE PERFECT GRID
    # -------------------------------
    print("Enforcing strict 1-hour continuity grid...")
    def enforce_grid(group):
        group = group.set_index('timestamp')
        full_idx = pd.date_range(group.index.min(), group.index.max(), freq='1h')
        group = group.reindex(full_idx)
        group.index.name = 'timestamp'
        
        # Restore static info safely
        group[['city', 'station', 'latitude', 'longitude']] = group[['city', 'station', 'latitude', 'longitude']].ffill().bfill()
        return group.reset_index()

    df_hourly = df_hourly.groupby(['city', 'station'], group_keys=False).apply(enforce_grid)

    # -------------------------------
    # 3. DESTROY FLATLINES
    # -------------------------------
    print("Hunting down and destroying hardware flatlines (> 24 hrs)...")
    for p in POLLUTANTS:
        # Check if the current value is exactly the same as the previous value
        is_same = df_hourly[p] == df_hourly.groupby(['city', 'station'])[p].shift()
        streak_id = (~is_same).cumsum()
        streak_lengths = df_hourly.groupby(['city', 'station', streak_id])[p].transform('size')
        
        # If it's flat for more than 24 hours, turn it into a NaN so we can slice it
        df_hourly.loc[(streak_lengths > 24) & df_hourly[p].notna(), p] = np.nan

    # -------------------------------
    # 4. SAFE PHYSICS & SHORT FILL
    # -------------------------------
    df_hourly[FEATURES] = df_hourly[FEATURES].fillna(0) # Missing wind physics defaults to 0

    print("Bridging short sensor drops (Max 3 hours)...")
    df_hourly[POLLUTANTS] = df_hourly.groupby(['city', 'station'])[POLLUTANTS].transform(
        lambda x: x.ffill(limit=3)
    )

    # -------------------------------
    # 5. CONTINUOUS CHUNKING (THE FIX)
    # -------------------------------
    print("Slicing timelines into 100% continuous chunks...")
    # If ANY pollutant is STILL NaN, the station is fundamentally broken at this timestamp.
    df_hourly['is_broken'] = df_hourly[POLLUTANTS].isna().any(axis=1)
    
    # Every time we hit a broken row, the chunk_id increments
    df_hourly['chunk_id'] = df_hourly.groupby(['city', 'station'])['is_broken'].cumsum()

    # NOW we drop the broken rows. What remains are perfectly continuous blocks.
    df_hourly = df_hourly[~df_hourly['is_broken']].copy()

    # Append the chunk_id to make Nixtla unique_ids (e.g., Delhi_ITO_c0, Delhi_ITO_c1)
    df_hourly['unique_id'] = df_hourly['city'] + "_" + df_hourly['station'] + "_c" + df_hourly['chunk_id'].astype(str)

    # -------------------------------
    # 6. DISCARD USELESS CHUNKS
    # -------------------------------
    # NeuralForecast needs 96-in + 96-out (192 hours). Throw away anything smaller.
    lengths = df_hourly['unique_id'].value_counts()
    keep_ids = lengths[lengths >= 192].index
    df_hourly = df_hourly[df_hourly['unique_id'].isin(keep_ids)].copy()

    # Time covariates
    df_hourly['hour'] = df_hourly['timestamp'].dt.hour
    df_hourly['day'] = df_hourly['timestamp'].dt.day
    df_hourly['weekday'] = df_hourly['timestamp'].dt.weekday
    df_hourly['month'] = df_hourly['timestamp'].dt.month

    # Cleanup
    df_hourly.drop(columns=['is_broken', 'chunk_id'], inplace=True)
    
    print(f"Saving {len(df_hourly):,} clean, perfectly continuous rows to {OUTPUT_PATH}...")
    df_hourly.to_parquet(OUTPUT_PATH, index=False)
    print("SUCCESS: ML Tensor Generated.")

if __name__ == "__main__":
    build_hourly_data()