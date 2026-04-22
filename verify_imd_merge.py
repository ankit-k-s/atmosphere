import pandas as pd
import numpy as np

PATH = "data/processed/clean_data_with_wind.parquet"

print("Loading merged dataset...")
df = pd.read_parquet(PATH)

print("\n=== TEST 1: BASIC SHAPE ===")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# -------------------------------
# TEST 2: WIND COMPLETENESS
# -------------------------------
print("\n=== TEST 2: WIND COMPLETENESS ===")

missing_wind_x = df['wind_x'].isna().mean() * 100
missing_wind_y = df['wind_y'].isna().mean() * 100

print(f"Missing wind_x: {missing_wind_x:.2f}%")
print(f"Missing wind_y: {missing_wind_y:.2f}%")

if missing_wind_x < 5:
    print("[SUCCESS] Wind coverage significantly improved.")
else:
    print("[WARNING] Wind still missing heavily.")

# -------------------------------
# TEST 3: VALUE RANGES
# -------------------------------
print("\n=== TEST 3: WIND VALUE RANGES ===")

print(f"Wind X: {df['wind_x'].min():.2f} to {df['wind_x'].max():.2f}")
print(f"Wind Y: {df['wind_y'].min():.2f} to {df['wind_y'].max():.2f}")

# -------------------------------
# TEST 4: NON-ZERO SIGNAL
# -------------------------------
print("\n=== TEST 4: NON-ZERO WIND SIGNAL ===")

zero_ratio = ((df['wind_x'] == 0) & (df['wind_y'] == 0)).mean() * 100
print(f"Zero wind vectors: {zero_ratio:.2f}%")

# -------------------------------
# TEST 5: TEMPORAL CONSISTENCY
# -------------------------------
print("\n=== TEST 5: TEMPORAL CONTINUITY ===")

sample_station = df['station'].iloc[0]
df_sample = df[df['station'] == sample_station].sort_values('timestamp')

gaps = df_sample['timestamp'].diff().value_counts().head(3)
print(gaps)

# -------------------------------
# TEST 6: WIND DISTRIBUTION
# -------------------------------
print("\n=== TEST 6: WIND DISTRIBUTION ===")

print(df[['wind_x', 'wind_y']].describe())

print("\n=== VERIFICATION COMPLETE ===")