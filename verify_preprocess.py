import pandas as pd
import sys

FILE_PATH = "data/processed/clean_data_15min.parquet"

def run_tests():
    print(" INITIALIZING PIPELINE VERIFICATION SUITE...\n")
    
    try:
        df = pd.read_parquet(FILE_PATH)
    except Exception as e:
        print(f" FAILED TO LOAD PARQUET: {e}")
        sys.exit(1)

    # TEST 1: Basic Sanity
    print("=== TEST 1: DATASET SHAPE & SANITY ===")
    print(f"Total Rows: {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    print(f"City Count: {df['city'].nunique()}")
    print(f"Station Count: {df['station'].nunique()}\n")

    # TEST 2: Time Continuity (Crucial for TFT)
    print("=== TEST 2: 15-MIN GRID CONTINUITY ===")
    # Taking a random highly-populated station as a sample
    sample_station = df['station'].iloc[0] 
    sample_city = df['city'].iloc[0]
    sample = df[(df["city"] == sample_city) & (df["station"] == sample_station)].sort_values("timestamp")
    
    diffs = sample["timestamp"].diff().value_counts().head(3)
    print(f"Time gaps for {sample_station}, {sample_city}:")
    print(diffs)
    if pd.Timedelta(minutes=15) in diffs.index and diffs.iloc[0] == diffs[pd.Timedelta(minutes=15)]:
        print(" Grid Alignment SUCCESS: Dominant gap is exactly 15 minutes.\n")
    else:
        print(" Grid Alignment WARNING: 15-minute gap is not dominant.\n")

    # TEST 3: Missing Value Check
    print("=== TEST 3: MISSING VALUE RATIOS ===")
    missing_ratios = df.isna().mean().sort_values(ascending=False).head(5)
    print("Top 5 sparsest columns:")
    print(missing_ratios.apply(lambda x: f"{x:.2%} missing"))
    print()

    # TEST 4 & 5: Physical Constraints
    print("=== TEST 4 & 5: PHYSICAL CONSTRAINTS ===")
    if 'PM2.5' in df.columns:
        pm25_min, pm25_max = df['PM2.5'].min(), df['PM2.5'].max()
        print(f"PM2.5 Bounds: Min = {pm25_min:.2f}, Max = {pm25_max:.2f}")
        if pm25_min < 0: print("❌ WARNING: Negative PM2.5 detected!")
    
    if 'wind_x' in df.columns and 'wind_y' in df.columns:
        print(f"Wind Vectors: X-bounds ({df['wind_x'].min():.2f} to {df['wind_x'].max():.2f})")
        print(f"Wind Vectors: Y-bounds ({df['wind_y'].min():.2f} to {df['wind_y'].max():.2f})")
    print()

    # TEST 6: Duplicates
    print("=== TEST 6: DUPLICATE TIMESTAMPS ===")
    duplicates = df.duplicated(subset=["city", "station", "timestamp"]).sum()
    if duplicates == 0:
        print(" SUCCESS: 0 duplicate timestamps found.\n")
    else:
        print(f" WARNING: Found {duplicates} duplicate rows!\n")

    # TEST 7: Station Coverage Map
    print("=== TEST 7: STATION COVERAGE BY CITY ===")
    coverage = df.groupby("city")["station"].nunique()
    print(coverage)
    print("\n VERIFICATION COMPLETE.")

if __name__ == "__main__":
    run_tests()