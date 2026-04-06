import pandas as pd
import sys
import os

FILE_PATH = "data/processed/clean_data_geo.parquet"

def run_final_verification():
    print("INITIALIZING FINAL SPATIAL PIPELINE VERIFICATION SUITE...\n")

    if not os.path.exists(FILE_PATH):
        print(f"[ERROR] File not found: {FILE_PATH}")
        print("Ensure you have run the final merge script successfully.")
        sys.exit(1)

    try:
        print(f"[INFO] Loading {FILE_PATH}. This may take a moment due to file size...")
        df = pd.read_parquet(FILE_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load Parquet file: {e}")
        sys.exit(1)

    # TEST 1: Schema and Shape
    print("\n=== TEST 1: DATASET SHAPE & SCHEMA ===")
    print(f"Total Rows: {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    
    required_cols = ['city', 'station', 'timestamp', 'latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns: {missing_cols}")
    else:
        print("[SUCCESS] All required core temporal and spatial columns are present.")

    # TEST 2: Spatial Completeness
    print("\n=== TEST 2: SPATIAL COMPLETENESS ===")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        missing_lat = df['latitude'].isna().sum()
        missing_lon = df['longitude'].isna().sum()
        print(f"Null Latitudes: {missing_lat}")
        print(f"Null Longitudes: {missing_lon}")
        
        if missing_lat == 0 and missing_lon == 0:
            print("[SUCCESS] 100% Spatial Completeness Achieved.")
        else:
            print(f"[WARNING] Spatial data is incomplete. {missing_lat} rows lack coordinates.")
    else:
        print("[ERROR] Spatial columns not found for Test 2.")

    # TEST 3: Geographic Bounds Validation (India Approximation)
    # India bounding box roughly: Lat 8.0 to 38.0, Lon 68.0 to 98.0
    print("\n=== TEST 3: GEOGRAPHIC BOUNDS VALIDATION ===")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
        min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
        
        print(f"Latitude Range:  {min_lat:.5f} to {max_lat:.5f}")
        print(f"Longitude Range: {min_lon:.5f} to {max_lon:.5f}")
        
        lat_valid = (8.0 <= min_lat) and (max_lat <= 38.0)
        lon_valid = (68.0 <= min_lon) and (max_lon <= 98.0)
        
        if lat_valid and lon_valid:
            print("[SUCCESS] All coordinates fall within expected geographic bounds for India.")
        else:
            print("[WARNING] Coordinates detected outside expected India bounding box.")

    # TEST 4: Duplicate Record Check
    print("\n=== TEST 4: DUPLICATE RECORD CHECK ===")
    duplicates = df.duplicated(subset=["city", "station", "timestamp"]).sum()
    if duplicates == 0:
        print("[SUCCESS] 0 duplicate records found. Time-series integrity maintained.")
    else:
        print(f"[WARNING] Found {duplicates} duplicate records.")

    # TEST 5: Station Coverage
    print("\n=== TEST 5: STATION COVERAGE BY CITY ===")
    coverage = df.groupby("city")["station"].nunique()
    print(coverage.to_string())
    
    print("\n[INFO] VERIFICATION COMPLETE.")

if __name__ == "__main__":
    run_final_verification()