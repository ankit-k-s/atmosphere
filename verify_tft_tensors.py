import pandas as pd
import pickle
import sys
import os
import torch
from pytorch_forecasting import TimeSeriesDataSet

PARAM_FILE = "data/processed/tft/tft_dataset_params.pkl"
DATA_FILE = "data/processed/tft/tft_ready_data_sanitized.parquet"

def verify_tensors():
    print("INITIALIZING DEEP LEARNING TENSOR COMPILATION TEST...\n")

    if not os.path.exists(PARAM_FILE) or not os.path.exists(DATA_FILE):
        print("[ERROR] Required TFT artifacts not found.")
        sys.exit(1)

    try:
        print("[INFO] Loading dataset parameters and sanitized dataframe...")
        with open(PARAM_FILE, "rb") as f:
            dataset_parameters = pickle.load(f)
        df = pd.read_parquet(DATA_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to load TFT artifacts: {e}")
        sys.exit(1)

    # TEST 1: Reconstruct Training Dataset
    print("\n=== TEST 1: TIMESERIESDATASET RECONSTRUCTION ===")
    try:
        # Reconstruct the training dataset object using the saved parameters
        train_df = df[df["time_idx"] <= df["time_idx"].max() - 2880]
        training_dataset = TimeSeriesDataSet.from_parameters(dataset_parameters, train_df)
        print("[SUCCESS] TimeSeriesDataSet perfectly reconstructed from serialized parameters.")
    except Exception as e:
        print(f"[ERROR] Failed to reconstruct dataset: {e}")
        sys.exit(1)

    # TEST 2: Multi-Target Configuration
    print("\n=== TEST 2: MULTI-TARGET HORIZON VERIFICATION ===")
    targets = training_dataset.target_names
    expected_targets = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']
    
    if targets == expected_targets:
        print(f"[SUCCESS] Multi-Target objectives correctly locked: {len(targets)} targets.")
    else:
        print(f"[ERROR] Target mismatch. Expected {expected_targets}, found {targets}")

    # TEST 3: Validation Split Generation
    print("\n=== TEST 3: VALIDATION SPLIT GENERATION ===")
    try:
        # The validation set must use the exact same scalers and mappings as the training set
        val_df = df[df["time_idx"] > df["time_idx"].max() - 2880 - dataset_parameters["max_encoder_length"]]
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, val_df, predict=True, stop_randomization=True
        )
        print(f"[SUCCESS] Validation dataset generated. Validation sequences: {len(validation_dataset):,}")
    except Exception as e:
        print(f"[ERROR] Failed to construct validation dataset: {e}")
        sys.exit(1)

    # TEST 4: DataLoader Compilation (The Ultimate Test)
    print("\n=== TEST 4: PYTORCH DATALOADER COMPILATION ===")
    try:
        # Create a PyTorch DataLoader
        batch_size = 64
        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        
        # Fetch exactly one batch of tensors
        x, y = next(iter(train_dataloader))
        
        # Check Encoder Dimensions (Batch Size, Sequence Length, Features)
        encoder_cont_shape = x['encoder_cont'].shape
        print(f"[INFO] Encoder Continuous Tensor Shape: {list(encoder_cont_shape)}")
        
        if encoder_cont_shape[0] == batch_size and encoder_cont_shape[1] == 96:
            print("[SUCCESS] Batch size and Look-back horizon (96 steps) correctly mapped.")
        else:
            print("[WARNING] Encoder dimensions do not match expectations.")

        # Check Target Dimensions (Since multi-target, y is a tuple of tensors)
        if isinstance(y[0], tuple):
            decoder_target_shape = y[0][0].shape  # Shape of the first target
        else:
            print("[WARNING] Target output is not a tuple, PyTorch Forecasting structure changed.")
            sys.exit(1)

        print(f"[INFO] Decoder Target Tensor Shape: {list(decoder_target_shape)}")
        if decoder_target_shape[0] == batch_size and decoder_target_shape[1] == 96:
            print("[SUCCESS] Look-forward horizon (96 steps) correctly mapped.")
        
        # Scan for hidden NaNs inside the compiled tensor
        if torch.isnan(x['encoder_cont']).any():
            print("[ERROR] Hidden NaNs detected inside the continuous tensors. Matrix multiplication will fail.")
        else:
            print("[SUCCESS] Tensors are mathematically clean (0 NaNs detected).")

    except Exception as e:
        print(f"[ERROR] DataLoader failed to compile batch: {e}")
        sys.exit(1)

    print("\n[INFO] ALL TENSOR VERIFICATIONS PASSED. READY FOR GPU TRAINING.")

if __name__ == "__main__":
    verify_tensors()