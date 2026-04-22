import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# -------------------------------
# CONFIG
# -------------------------------
CPCB_PATH = "data/processed/clean_data_geo.parquet"
IMD_PATH = "data/raw/imd/imd_wind_15min.parquet"
OUTPUT_PATH = "data/processed/clean_data_with_wind.parquet"

TIME_COL = "timestamp"

# -------------------------------
# LOAD DATA
# -------------------------------
print("Loading datasets...")
df_cpcb = pd.read_parquet(CPCB_PATH)
df_imd = pd.read_parquet(IMD_PATH)

df_cpcb[TIME_COL] = pd.to_datetime(df_cpcb[TIME_COL])
df_imd[TIME_COL] = pd.to_datetime(df_imd[TIME_COL])

# -------------------------------
# STEP 1: UNIQUE STATION MATCHING
# -------------------------------
print("Extracting unique stations...")

unique_cpcb = df_cpcb[['station', 'latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
unique_imd = df_imd[['station', 'latitude', 'longitude']].drop_duplicates().reset_index(drop=True)

# Build BallTree ONLY on unique stations
print("Building BallTree on IMD stations...")
imd_coords = np.radians(unique_imd[['latitude', 'longitude']].values)
cpcb_coords = np.radians(unique_cpcb[['latitude', 'longitude']].values)

tree = BallTree(imd_coords, metric='haversine')
dist, idx = tree.query(cpcb_coords, k=1)

# Map CPCB → nearest IMD
unique_cpcb['imd_station'] = unique_imd.iloc[idx.flatten()]['station'].values

station_map = dict(zip(unique_cpcb['station'], unique_cpcb['imd_station']))

print("Station mapping created.")

# -------------------------------
# STEP 2: APPLY MAPPING
# -------------------------------
print("Applying mapping to full dataset...")
df_cpcb['imd_station'] = df_cpcb['station'].map(station_map)

# -------------------------------
# STEP 3: MERGE TEMPORAL WIND
# -------------------------------
print("Merging IMD wind data...")

df_merged = df_cpcb.merge(
    df_imd[['station', TIME_COL, 'wind_speed', 'wind_dir']],
    left_on=['imd_station', TIME_COL],
    right_on=['station', TIME_COL],
    how='left',
    suffixes=('', '_imd')
)

# -------------------------------
# STEP 4: HANDLE MISSING WIND
# -------------------------------
print("Interpolating wind data...")

df_merged = df_merged.sort_values(['station', TIME_COL])

df_merged[['wind_speed', 'wind_dir']] = df_merged.groupby('station')[
    ['wind_speed', 'wind_dir']
].transform(lambda x: x.interpolate(limit=3).ffill().bfill())

# -------------------------------
# STEP 5: WIND VECTOR (IMPORTANT FIX)
# -------------------------------
print("Converting to wind vectors...")

# Meteorological convention:
# wind_dir = direction wind COMES FROM
theta = np.radians(270 - df_merged['wind_dir'])

df_merged['wind_x'] = df_merged['wind_speed'] * np.cos(theta)
df_merged['wind_y'] = df_merged['wind_speed'] * np.sin(theta)

# -------------------------------
# CLEANUP
# -------------------------------
df_merged.drop(columns=['imd_station', 'station_imd'], errors='ignore', inplace=True)

print("Saving dataset...")
df_merged.to_parquet(OUTPUT_PATH, index=False)

print("SUCCESS: IMD wind merged correctly.")