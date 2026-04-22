import pandas as pd
import numpy as np

PATH = "data/processed/tft/tide_ready_1hr.parquet"

print("Loading hourly dataset...")
df = pd.read_parquet(PATH)

# -------------------------------
# BASIC INFO
# -------------------------------
print("\n=== TEST 1: SHAPE ===")
print(df.shape)

# -------------------------------
# REQUIRED COLUMNS
# -------------------------------
print("\n=== TEST 2: REQUIRED COLUMNS ===")
required_cols = [
    'timestamp', 'unique_id',
    'PM2_5_ug_m3', 'PM10_ug_m3',
    'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3'
]

missing = [c for c in required_cols if c not in df.columns]
print("Missing:", missing if missing else "[SUCCESS] All present")

# -------------------------------
# DUPLICATES
# -------------------------------
print("\n=== TEST 3: DUPLICATE ROWS ===")
dups = df.duplicated(subset=['unique_id', 'timestamp']).sum()
print(f"Duplicate rows: {dups}")

# -------------------------------
# TIME CONTINUITY
# -------------------------------
print("\n=== TEST 4: TIME CONTINUITY ===")
sample_ids = df['unique_id'].drop_duplicates().sample(3, random_state=42)

for uid in sample_ids:
    temp = df[df['unique_id'] == uid].sort_values('timestamp')
    gaps = temp['timestamp'].diff().value_counts().head(3)
    print(f"\n{uid}:")
    print(gaps)

# -------------------------------
# ZERO CHECK
# -------------------------------
print("\n=== TEST 5: ZERO RATIOS ===")
pollutants = [
    'PM2_5_ug_m3', 'PM10_ug_m3',
    'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3'
]

for col in pollutants:
    zero_ratio = (df[col] == 0).mean() * 100
    print(f"{col}: {zero_ratio:.2f}% zeros")

# -------------------------------
# NEGATIVE CHECK
# -------------------------------
print("\n=== TEST 6: NEGATIVE VALUES ===")
for col in pollutants:
    neg = (df[col] < 0).sum()
    print(f"{col}: {neg}")

# -------------------------------
# 🔥 NEW: CONSECUTIVE DUPLICATE CHECK
# -------------------------------
print("\n=== TEST 7: CONSECUTIVE FLATLINE CHECK ===")

def longest_flat_run(series):
    # compute longest consecutive identical values
    grp = (series != series.shift()).cumsum()
    counts = grp.value_counts()
    return counts.max()

results = []

for uid in df['unique_id'].drop_duplicates().sample(10, random_state=42):
    temp = df[df['unique_id'] == uid].sort_values('timestamp')

    for col in pollutants:
        run = longest_flat_run(temp[col])
        results.append((uid, col, run))

# Convert to DataFrame
flat_df = pd.DataFrame(results, columns=['unique_id', 'pollutant', 'max_flat_hours'])

print(flat_df.sort_values('max_flat_hours', ascending=False).head(10))

# -------------------------------
# FLAG SUSPICIOUS PATTERNS
# -------------------------------
print("\n=== TEST 8: SUSPICIOUS FLATLINES ===")

threshold = 6  # >6 hours flat = suspicious
suspicious = flat_df[flat_df['max_flat_hours'] > threshold]

if len(suspicious) == 0:
    print("[SUCCESS] No long flatlines detected.")
else:
    print("[WARNING] Possible flatline artifacts:")
    print(suspicious.sort_values('max_flat_hours', ascending=False).head(10))

# -------------------------------
# WIND FLAG CHECK
# -------------------------------
print("\n=== TEST 9: WIND FLAG CONSISTENCY ===")

if 'wind_x' in df.columns:
    mismatch = (
        (df['wind_observed'] == 0) &
        ((df['wind_x'] != 0) | (df['wind_y'] != 0))
    ).sum()

    print(f"Wind mismatch rows: {mismatch}")
else:
    print("Wind columns missing.")

# -------------------------------
# SAMPLE OUTPUT
# -------------------------------
print("\n=== TEST 10: SAMPLE ===")
print(df.head())

print("\n=== VERIFICATION COMPLETE ===")