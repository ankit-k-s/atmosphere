import pandas as pd
import numpy as np
import sys
import os

FILE_PATH = "data/processed/tft/tft_ready_data.parquet"

def verify_tft_preparation():
    print("INITIALIZING TFT DATASET VERIFICATION SUITE...\n")

    if not os.path.exists(FILE_PATH):
        print(f"[ERROR] File not found: {FILE_PATH}")
        sys.exit(1)

    try:
        print(f"[INFO] Loading {FILE_PATH}. This may take a moment...")
        df = pd.read_parquet(FILE_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load Parquet file: {e}")
        sys.exit(1)

    # TEST 1: Shape and Schema
    print("\n=== TEST 1: DATASET SHAPE & SCHEMA ===")
    print(f"Total Rows: {df.shape[0]:,}")
    if 'time_idx' not in df.columns:
        print("[ERROR] 'time_idx' column is missing. TFT cannot track sequence progression.")
        sys.exit(1)
    else:
        print("[SUCCESS] Continuous 'time_idx' tracking column is present.")

    # TEST 2: PyTorch Data Type Constraints
    print("\n=== TEST 2: PYTORCH DATA TYPE CONSTRAINTS ===")
    type_errors = 0
    
    # Categoricals must be strings/objects for Entity Embedding
    if not pd.api.types.is_string_dtype(df['city']) and not pd.api.types.is_object_dtype(df['city']):
        print("[ERROR] 'city' is not a string type.")
        type_errors += 1
    if not pd.api.types.is_string_dtype(df['station']) and not pd.api.types.is_object_dtype(df['station']):
        print("[ERROR] 'station' is not a string type.")
        type_errors += 1
        
    # Reals must be float32 to prevent GPU memory crashes
    float32_cols = [c for c in df.columns if c not in ['city', 'station', 'timestamp', 'time_idx']]
    float32_failures = 0
    for col in float32_cols:
        if df[col].dtype != np.float32:
            float32_failures += 1
    
    if type_errors == 0 and float32_failures == 0:
        print("[SUCCESS] Categoricals are strings. Continuous variables successfully downcast to float32.")
    else:
        print(f"[ERROR] Found {type_errors} string type errors and {float32_failures} float32 casting errors.")

    # TEST 3: Time Index Sorting and Continuity
    print("\n=== TEST 3: TIME INDEX SORTING CONSTRAINTS ===")
    # TFT requires strict autoregressive sorting (Station -> Time)
    is_sorted = df.groupby('station')['time_idx'].is_monotonic_increasing.all()
    if is_sorted:
        print("[SUCCESS] Dataset is strictly sorted by station and chronological time_idx.")
    else:
        print("[ERROR] Dataset is NOT properly sorted. PyTorch will mix future and past data.")

    # TEST 4: Null Values Check in Critical ML Identifiers
    print("\n=== TEST 4: NULL VALUE CHECK (CRITICAL IDENTIFIERS) ===")
    critical_cols = ['city', 'station', 'timestamp', 'time_idx', 'hour', 'day', 'month', 'weekday']
    missing_critical_cols = [c for c in critical_cols if c not in df.columns]
    
    if missing_critical_cols:
         print(f"[ERROR] Missing critical temporal columns: {missing_critical_cols}")
    else:
        null_critical = df[critical_cols].isna().sum().sum()
        if null_critical == 0:
            print("[SUCCESS] 0 Null values in critical ML identifiers and temporal features.")
        else:
            print(f"[ERROR] Found {null_critical} Null values in critical ML columns.")

    # INFO: Memory Footprint Validation
    print("\n=== INFO: MEMORY FOOTPRINT ===")
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Total Dataset RAM requirement: {memory_usage_mb:.2f} MB")
    if memory_usage_mb < 5000:
        print("[SUCCESS] Memory footprint is optimized for standard hardware loading.")
    else:
        print("[WARNING] Memory footprint is massive. Expect heavy RAM usage during DataLoader creation.")

    print("\n[INFO] VERIFICATION COMPLETE.")

if __name__ == "__main__":
    verify_tft_preparation()