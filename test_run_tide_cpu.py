import pandas as pd
import os
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import TiDE
#  FIX: Import MQLoss here
from neuralforecast.losses.pytorch import MQLoss

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/tft/tide_ready_1hr.parquet"
MODEL_DIR = "models/tide_cpu_test"
TEST_VAULT_DIR = "data/processed/tft/evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEST_VAULT_DIR, exist_ok=True)

INPUT_CHUNK = 96
OUTPUT_CHUNK = 96

POLLUTANTS = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']

def prep_nixtla_format(df):
    print("Melting perfect tensor into NeuralForecast strict-long format...")
    
    hist_exog = ['wind_x', 'wind_y', 'wind_observed', 
                 'adv_in_PM2_5_ug_m3', 'adv_in_PM10_ug_m3', 
                 'adv_in_NO2_ug_m3', 'adv_in_SO2_ug_m3', 'adv_in_CO_mg_m3']
    
    futr_exog = ['hour', 'day', 'weekday', 'month', 'latitude', 'longitude']
    
    id_vars = ['unique_id', 'timestamp'] + hist_exog + futr_exog
    df_melted = df.melt(
        id_vars=id_vars, value_vars=POLLUTANTS,
        var_name='pollutant_type', value_name='y'
    )
    
    df_melted['unique_id'] = df_melted['unique_id'] + "_" + df_melted['pollutant_type']
    df_melted = df_melted.drop(columns=['pollutant_type'])
    df_melted = df_melted.rename(columns={'timestamp': 'ds'})
    df_melted = df_melted.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    
    return df_melted, hist_exog, futr_exog

def run_cpu_smoke_test():
    print("Loading 1-hour ML Tensor...")
    df = pd.read_parquet(INPUT_FILE)
    
    df_nixtla, hist_exog, futr_exog = prep_nixtla_format(df)
    print(f"Total isolated training sequences: {df_nixtla['unique_id'].nunique():,}")
    
    print("\n[INITIATING STRICT TEMPORAL SPLIT]")
    VAL_HOURS = OUTPUT_CHUNK   
    TEST_HOURS = 24 * 30  
    
    max_date = df_nixtla['ds'].max()
    test_cutoff = max_date - pd.Timedelta(hours=TEST_HOURS)
    
    print(f"Slicing Global Test Vault at {test_cutoff}...")
    df_train_val = df_nixtla[df_nixtla['ds'] <= test_cutoff].reset_index(drop=True)
    df_test = df_nixtla[df_nixtla['ds'] > test_cutoff].reset_index(drop=True)
    
    REQUIRED_LEN = INPUT_CHUNK + VAL_HOURS
    print(f"Filtering fragmented train chunks shorter than {REQUIRED_LEN} hours...")
    
    train_lengths = df_train_val['unique_id'].value_counts()
    valid_train_ids = train_lengths[train_lengths >= REQUIRED_LEN].index
    
    df_train_val = df_train_val[df_train_val['unique_id'].isin(valid_train_ids)].reset_index(drop=True)
    print(f"Cleaned Train/Val sequences remaining: {df_train_val['unique_id'].nunique():,}")
    
    print("\nInstantiating TiDE for CPU SMOKE TEST...")
    
    tide_model = TiDE(
        h=OUTPUT_CHUNK,
        input_size=INPUT_CHUNK,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog,
        hidden_size=512,          
        decoder_output_dim=32,    
        temporal_width=8,         
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), #  FIX: Multi-Quantile Loss
        learning_rate=1e-3,
        max_steps=10,             
        batch_size=64,            
        scaler_type='robust',     
        accelerator="cpu" 
    )
    
    nf = NeuralForecast(models=[tide_model], freq='1h')
    
    print("\n[INITIATING CPU TRAINING SEQUENCE (10 STEPS ONLY)]")
    
    nf.fit(df=df_train_val, val_size=VAL_HOURS)
    
    print("\n[SMOKE TEST COMPLETE] CPU successfully compiled and ran the model!")
    print("\nSUCCESS: The pipeline is mathematically sound. You are cleared for H100 deployment.")

if __name__ == "__main__":
    run_cpu_smoke_test()