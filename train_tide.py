import pandas as pd
import os
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import TiDE
#  FIX 1: Multi-Quantile Loss imported
from neuralforecast.losses.pytorch import MQLoss

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/tft/tide_ready_1hr.parquet"
MODEL_DIR = "models/tide_global"
TEST_VAULT_DIR = "data/processed/tft/evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEST_VAULT_DIR, exist_ok=True)

INPUT_CHUNK = 96  # Look back 4 days
OUTPUT_CHUNK = 96 # Predict next 4 days

POLLUTANTS = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']

def prep_nixtla_format(df):
    print("Melting perfect tensor into NeuralForecast strict-long format...")
    
    hist_exog = ['wind_x', 'wind_y', 'wind_observed', 
                 'adv_in_PM2_5_ug_m3', 'adv_in_PM10_ug_m3', 
                 'adv_in_NO2_ug_m3', 'adv_in_SO2_ug_m3', 'adv_in_CO_mg_m3']
    
    futr_exog = ['hour', 'day', 'weekday', 'month', 'latitude', 'longitude']
    
    # Melt the 5 pollutants into a single 'y' target column
    id_vars = ['unique_id', 'timestamp'] + hist_exog + futr_exog
    df_melted = df.melt(
        id_vars=id_vars, value_vars=POLLUTANTS,
        var_name='pollutant_type', value_name='y'
    )
    
    # Create the final multi-target unique ID (e.g., Delhi_ITO_c0_PM2_5)
    df_melted['unique_id'] = df_melted['unique_id'] + "_" + df_melted['pollutant_type']
    df_melted = df_melted.drop(columns=['pollutant_type'])
    
    # Rename timestamp to 'ds' (Strict Nixtla requirement)
    df_melted = df_melted.rename(columns={'timestamp': 'ds'})
    df_melted = df_melted.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    
    return df_melted, hist_exog, futr_exog

def train_tide():
    print("Loading 1-hour ML Tensor...")
    df = pd.read_parquet(INPUT_FILE)
    
    df_nixtla, hist_exog, futr_exog = prep_nixtla_format(df)
    print(f"Total isolated training sequences: {df_nixtla['unique_id'].nunique():,}")
    
    # ---------------------------------------------------------
    # STRICT TEMPORAL SPLIT (TRAIN / VAL / TEST)
    # ---------------------------------------------------------
    print("\n[INITIATING STRICT TEMPORAL SPLIT]")
    
    # Internal Early Stopping Validation Window (4 days per chunk)
    VAL_HOURS = OUTPUT_CHUNK 
    # Absolute Unseen Evaluation Vault (30 days global)
    TEST_HOURS = 24 * 30  
    
    max_date = df_nixtla['ds'].max()
    test_cutoff = max_date - pd.Timedelta(hours=TEST_HOURS)
    
    print(f"Dataset Timeline: {df_nixtla['ds'].min()} to {max_date}")
    print(f"  -> Testing Vault:   {test_cutoff} to {max_date} (Global 30 Days)")
    
    # df_train_val goes into the model. Nixtla will carve out VAL_HOURS from each chunk.
    df_train_val = df_nixtla[df_nixtla['ds'] <= test_cutoff].reset_index(drop=True)
    
    # df_test is vaulted. The model NEVER sees this during training.
    df_test = df_nixtla[df_nixtla['ds'] > test_cutoff].reset_index(drop=True)
    
    # 🔥 FIX 2: The Guillotine Filter (Ensure no chunks are too short for PyTorch)
    REQUIRED_LEN = INPUT_CHUNK + VAL_HOURS
    print(f"Filtering fragmented train chunks shorter than {REQUIRED_LEN} hours...")
    train_lengths = df_train_val['unique_id'].value_counts()
    valid_train_ids = train_lengths[train_lengths >= REQUIRED_LEN].index
    df_train_val = df_train_val[df_train_val['unique_id'].isin(valid_train_ids)].reset_index(drop=True)
    
    print(f"Cleaned Train/Val sequences remaining for GPU: {df_train_val['unique_id'].nunique():,}")
    
    # ---------------------------------------------------------
    # TiDE ARCHITECTURE CONFIGURATION
    # ---------------------------------------------------------
    print("\nInstantiating Time-series Dense Encoder (TiDE) for H100 GPUs...")
    
    tide_model = TiDE(
        h=OUTPUT_CHUNK,
        input_size=INPUT_CHUNK,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog,
        hidden_size=512,          # Wide MLP to capture complex advection physics
        decoder_output_dim=32,    # Temporal projection dimension
        temporal_width=8,         # Width of temporal decoder
        
        #  FIX 3: Multi-Quantile Loss activated
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), # P10, P50, P90 Confidence Bands
        
        learning_rate=1e-3,
        max_steps=3000,           
        batch_size=512,           # Maximize H100 VRAM saturation
        scaler_type='robust',     # Crucial for ignoring massive atmospheric spikes
        early_stop_patience_steps=5, # Halts training if Validation loss stops improving
        
        #  FIX 4: Hardware arguments passed directly (Not nested!)
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed"    # Hopper native brain-float 16 for massive speedup
    )
    
    nf = NeuralForecast(models=[tide_model], freq='1h')
    
    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------
    print("\n[INITIATING GPU TRAINING SEQUENCE]")
    
    # Fire up PyTorch Lightning
    nf.fit(df=df_train_val, val_size=VAL_HOURS)
    
    # ---------------------------------------------------------
    # SECURE CHECKPOINTS & VAULT TEST DATA
    # ---------------------------------------------------------
    print("\n[TRAINING COMPLETE] Securing TiDE model weights and Test Data...")
    
    # 1. Save Model Weights
    nf.save(path=MODEL_DIR, model_index=None, overwrite=True, save_dataset=False)
    print(f"  Model Weights secured to: {MODEL_DIR}")
    
    # 2. Vault Test Set
    test_vault_path = os.path.join(TEST_VAULT_DIR, "test_holdout_set.parquet")
    df_test.to_parquet(test_vault_path, index=False)
    print(f"  Unseen Test Dataset vaulted to: {test_vault_path}")
    
    print("\nSUCCESS: The Foundation Model is ready for evaluation.")

if __name__ == "__main__":
    # Ensure PyTorch Matrix Cores are fully unthrottled
    torch.set_float32_matmul_precision('medium')
    train_tide()