import pandas as pd
import numpy as np

PATH = "data/processed/tft/tide_ready_1hr.parquet"

# TiDE Architecture Constraints
INPUT_CHUNK = 96
OUTPUT_CHUNK = 96
MIN_SEQ_LENGTH = INPUT_CHUNK + OUTPUT_CHUNK

print(f"Loading hourly dataset from {PATH}...")
df = pd.read_parquet(PATH)

print("\n==================================================")
print(" TiDE STRICT ARCHITECTURE VALIDATION SUITE ")
print("==================================================\n")

# -------------------------------
# TEST 1: ABSOLUTE NaN CHECK
# -------------------------------
print("=== TEST 1: GLOBAL NaN CHECK ===")
total_nans = df.isna().sum().sum()
if total_nans == 0:
    print("[SUCCESS] 0 NaNs found. Tensor matrix is perfectly dense.")
else:
    print(f"[FAIL] Found {total_nans} NaNs! NeuralForecast will crash.")
    print(df.isna().sum()[df.isna().sum() > 0])

# -------------------------------
# TEST 2: STRICT TIME CONTINUITY (ALL STATIONS)
# -------------------------------
print("\n=== TEST 2: 100% TEMPORAL CONTINUITY ===")
# NeuralForecast requires exactly 1H frequency with NO internal gaps
# We calculate the time difference between every row per station
time_diffs = df.groupby('unique_id')['timestamp'].diff().dropna()
invalid_gaps = time_diffs[time_diffs != pd.Timedelta(hours=1)]

if len(invalid_gaps) == 0:
    print("[SUCCESS] All 132 stations have perfect 1-hour continuity. No internal gaps.")
else:
    print(f"[FAIL] Found {len(invalid_gaps)} broken time jumps!")
    print(invalid_gaps.head())

# -------------------------------
# TEST 3: MINIMUM SEQUENCE LENGTH
# -------------------------------
print("\n=== TEST 3: TiDE SEQUENCE LENGTH CONSTRAINTS ===")
# To train 96-in/96-out, a station MUST have at least 192 hours of history
seq_lengths = df.groupby('unique_id').size()
short_stations = seq_lengths[seq_lengths < MIN_SEQ_LENGTH]

if len(short_stations) == 0:
    print(f"[SUCCESS] All stations meet the {MIN_SEQ_LENGTH} hour minimum requirement.")
else:
    print(f"[WARNING] Found {len(short_stations)} stations with too little data to train:")
    print(short_stations)

# -------------------------------
# TEST 4: STATIC COVARIATE INTEGRITY
# -------------------------------
print("\n=== TEST 4: STATIC EMBEDDING INTEGRITY ===")
# Latitude and Longitude MUST be perfectly static for each unique_id
lat_variance = df.groupby('unique_id')['latitude'].nunique().max()
lon_variance = df.groupby('unique_id')['longitude'].nunique().max()

if lat_variance == 1 and lon_variance == 1:
    print("[SUCCESS] Geospatial coordinates are perfectly static per station.")
else:
    print("[FAIL] A station moved? Multiple coordinates found for a single unique_id.")

# -------------------------------
# TEST 5: GLOBAL FLATLINE DETECTION (ALL STATIONS)
# -------------------------------
print("\n=== TEST 5: GLOBAL FLATLINE ARTIFACTS ===")
pollutants = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']

def max_flatline(series):
    # Vectorized consecutive value counting
    return (series != series.shift()).cumsum().value_counts().max()

flatlines = {}
for col in pollutants:
    # Get the max flatline length for this pollutant across the WHOLE dataset
    max_flat = df.groupby('unique_id')[col].apply(max_flatline).max()
    flatlines[col] = max_flat

print("Maximum consecutive hours of the exact same value:")
for pol, hours in flatlines.items():
    if hours > 24:
        print(f"  [WARNING] {pol}: {hours} hours (Suspiciously long)")
    else:
        print(f"  [SUCCESS] {pol}: {hours} hours (Normal physics)")

# -------------------------------
# TEST 6: NEGATIVE PHYSICS CHECK
# -------------------------------
print("\n=== TEST 6: NEGATIVE VALUE CHECK ===")
neg_pollutants = (df[pollutants] < 0).sum().sum()
if neg_pollutants == 0:
    print("[SUCCESS] 0 negative pollutants found.")
else:
    print(f"[FAIL] Found {neg_pollutants} negative pollutant values.")

# -------------------------------
# TEST 7: WIND PHYSICS FLAG
# -------------------------------
print("\n=== TEST 7: WIND OBSERVED FLAG ===")
observed_ratio = df['wind_observed'].mean() * 100
print(f"Active Wind Physics: {observed_ratio:.2f}% of the time")
if observed_ratio > 0:
    print("[SUCCESS] Advection layer will receive live wind signals.")

print("\n==================================================")
if total_nans == 0 and len(invalid_gaps) == 0 and neg_pollutants == 0:
    print(" ✅ DATASET IS GREEN-LIT FOR NIXTLA NEURALFORECAST ✅")
else:
    print(" ❌ DO NOT TRAIN. FIX ERRORS ABOVE. ❌")
print("==================================================\n")