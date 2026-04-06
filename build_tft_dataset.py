import pandas as pd
import numpy as np
import os
import pickle
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import MultiNormalizer, GroupNormalizer

INPUT_FILE = "data/processed/tft/tft_ready_data.parquet"
OUTPUT_DIR = "data/processed/tft"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tft_dataset_params.pkl")

# Define architecture horizons
ENCODER_LENGTH = 96  # Look-back 24 hours
DECODER_LENGTH = 96  # Predict next 24 hours

def configure_tft_dataset():
    print("INITIALIZING TFT ARCHITECTURE CONFIGURATION...\n")
    
    print(f"Loading float32 tensor dataset from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    # ---------------------------------------------------------
    # 1. PYTORCH SANITIZATION
    # ---------------------------------------------------------
    print("Sanitizing column names for PyTorch constraints...")
    df.columns = [c.replace('.', '_').replace('/', '_').replace('µ', 'u').replace('³', '3') for c in df.columns]
    
    # ---------------------------------------------------------
    # 2. TEMPORAL IMPUTATION (FIX FOR NaN TENSOR CRASH)
    # ---------------------------------------------------------
    print("Imputing missing sensor data (NaNs) for strict Tensor compatibility...")
    # Ensure strict sorting so forward-fill doesn't bleed across stations
    df = df.sort_values(by=["station", "time_idx"]).reset_index(drop=True)
    
    # Identify all continuous float columns (wind, pollutants, advection)
    float_cols = df.select_dtypes(include=[np.float32, np.float64]).columns
    
    # Group by station: Forward fill (carry last known value), then Backward fill
    df[float_cols] = df.groupby("station")[float_cols].transform(lambda x: x.ffill().bfill())
    
    # Absolute fallback: If a station never recorded a specific variable, fill with 0 to prevent crash
    df[float_cols] = df[float_cols].fillna(0.0)
    
    # ---------------------------------------------------------
    # 3. TFT CONFIGURATION
    # ---------------------------------------------------------
    target_cols = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']
    advection_cols = [f'adv_in_{pol}' for pol in target_cols]
    
    max_time_idx = df["time_idx"].max()
    validation_cutoff = max_time_idx - 2880
    
    print(f"\nSplitting Data. Training cutoff time_idx: {validation_cutoff}")
    train_df = df[df["time_idx"] <= validation_cutoff]
    
    print("Configuring Multi-Target Normalizers...")
    target_normalizers = [GroupNormalizer(groups=["station"], transformation="softplus") for _ in target_cols]
    multi_normalizer = MultiNormalizer(target_normalizers)
    
    print("\nBuilding PyTorch TimeSeriesDataSet (This may take a minute)...")
    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_cols,
        group_ids=["city", "station"],
        
        min_encoder_length=ENCODER_LENGTH // 2,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=DECODER_LENGTH,
        max_prediction_length=DECODER_LENGTH,
        
        static_categoricals=["city", "station"],
        time_varying_known_reals=["time_idx", "hour", "day", "month", "weekday"],
        time_varying_unknown_reals=["wind_x", "wind_y"] + advection_cols + target_cols,
        
        target_normalizer=multi_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    print("\n TimeSeriesDataSet built successfully!")
    print(f"Number of training sequences created: {len(training_dataset):,}")
    
    print(f"Saving dataset parameters to {OUTPUT_FILE}...")
    dataset_parameters = training_dataset.get_parameters()
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(dataset_parameters, f)
        
    SANITIZED_DATA_FILE = os.path.join(OUTPUT_DIR, "tft_ready_data_sanitized.parquet")
    print(f"Saving sanitized & imputed dataframe to {SANITIZED_DATA_FILE}...")
    df.to_parquet(SANITIZED_DATA_FILE, index=False)
        
    print("TFT ARCHITECTURE SUCCESSFULLY CONFIGURED AND LOCKED.")

if __name__ == "__main__":
    configure_tft_dataset()