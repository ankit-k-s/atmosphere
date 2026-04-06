import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/features/advection_features.parquet"
OUTPUT_DIR = "data/processed/tft"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tft_ready_data.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_tft_dataset():
    print("INITIALIZING TEMPORAL FUSION TRANSFORMER (TFT) DATA PREPARATION...\n")
    
    print("Loading physics-aware dataset...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 1. Sort strictly by Station and Time
    # This is mathematically critical for autoregressive modeling
    print("Sorting continuous temporal sequences...")
    df = df.sort_values(by=['station', 'timestamp']).reset_index(drop=True)
    
    # 2. Create the continuous Time Index (time_idx)
    # TFT requires a globally consistent integer step for time.
    # Since we have 15-min intervals, we map unique timestamps to ascending integers.
    print("Generating strictly continuous Time Index (time_idx)...")
    unique_times = np.sort(df['timestamp'].unique())
    time_map = {time: idx for idx, time in enumerate(unique_times)}
    df['time_idx'] = df['timestamp'].map(time_map)
    
    # 3. Enforce Strict Data Types for PyTorch
    print("Enforcing PyTorch-compatible data types...")
    
    # Categoricals must be strings
    categorical_cols = ['city', 'station']
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        
    # Temporal Reals (Known Future)
    known_reals = ['hour', 'day', 'weekday', 'month']
    for col in known_reals:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
            
    # Float casting for all continuous variables to save RAM and optimize GPU
    exclude_cols = categorical_cols + ['timestamp', 'time_idx']
    float_cols = [col for col in df.columns if col not in exclude_cols]
    
    for col in float_cols:
        # Downcast to float32 to prevent out-of-memory errors during tensor creation
        df[col] = df[col].astype(np.float32)
        
    # 4. Final Sanity Check
    print("\nDataset ML Schema Overview:")
    print(f"Total Rows: {len(df):,}")
    print(f"Time Index Range: {df['time_idx'].min()} to {df['time_idx'].max()}")
    print(f"Total Unique Timesteps: {len(unique_times):,}")
    
    print(f"\nSaving TFT-ready tensor dataset to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print("TFT DATA PREPARATION SUCCESSFULLY LOCKED.")

if __name__ == "__main__":
    prepare_tft_dataset()