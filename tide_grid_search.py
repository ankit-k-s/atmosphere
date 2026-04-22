import pandas as pd
import numpy as np
import os
import itertools
from neuralforecast import NeuralForecast
from neuralforecast.models import TiDE
from neuralforecast.losses.pytorch import MQLoss
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/tft/tide_ready_1hr.parquet"
BEST_MODEL_DIR = "models/tide_optimized_best" # Brand new folder for the winner
TEMP_DIR = "models/tide_temp" # Temporary folder for current loop

os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

INPUT_CHUNK = 96
OUTPUT_CHUNK = 96
POLLUTANTS = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']

hist_exog = ['wind_x', 'wind_y', 'wind_observed', 
             'adv_in_PM2_5_ug_m3', 'adv_in_PM10_ug_m3', 
             'adv_in_NO2_ug_m3', 'adv_in_SO2_ug_m3', 'adv_in_CO_mg_m3']
futr_exog = ['hour', 'day', 'weekday', 'month', 'latitude', 'longitude']

def prep_nixtla_format(df):
    id_vars = ['unique_id', 'timestamp'] + hist_exog + futr_exog
    df_melted = df.melt(
        id_vars=id_vars, value_vars=POLLUTANTS,
        var_name='pollutant_type', value_name='y'
    )
    df_melted['unique_id'] = df_melted['unique_id'] + "_" + df_melted['pollutant_type']
    df_melted = df_melted.drop(columns=['pollutant_type'])
    df_melted = df_melted.rename(columns={'timestamp': 'ds'})
    df_melted = df_melted.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    return df_melted

def run_grid_search():
    print("Loading 1-hour ML Tensor for Grid Search...")
    df = pd.read_parquet(INPUT_FILE)
    df_nixtla = prep_nixtla_format(df)
    
    # ---------------------------------------------------------
    # 3-WAY STRICT SPLIT (Train / Grid Validation / Unseen Test)
    # ---------------------------------------------------------
    TEST_HOURS = 24 * 30  
    GRID_VAL_HOURS = 24 * 30  
    
    max_date = df_nixtla['ds'].max()
    test_cutoff = max_date - pd.Timedelta(hours=TEST_HOURS)
    val_cutoff = test_cutoff - pd.Timedelta(hours=GRID_VAL_HOURS)
    
    # Train on history up to val_cutoff
    df_train = df_nixtla[df_nixtla['ds'] <= val_cutoff].reset_index(drop=True)
    
    # Validate the grid search on this 30 day window
    df_val = df_nixtla[(df_nixtla['ds'] > val_cutoff) & (df_nixtla['ds'] <= test_cutoff)].reset_index(drop=True)
    
    # Guillotine filter to protect PyTorch from broken chunks
    REQUIRED_LEN = INPUT_CHUNK + OUTPUT_CHUNK
    valid_train_ids = df_train['unique_id'].value_counts()[lambda x: x >= REQUIRED_LEN].index
    df_train = df_train[df_train['unique_id'].isin(valid_train_ids)].reset_index(drop=True)
    
    # ---------------------------------------------------------
    # THE HYPERPARAMETER GRID
    # ---------------------------------------------------------
    # Testing combinations of width, depth, and learning rates
    grid = {
        'hidden_size': [512, 768],         # 768 is a massive, highly expressive MLP
        'temporal_width': [4, 8],          # How wide the temporal decoder is
        'learning_rate': [1e-3, 5e-4]      # Fast vs Smooth convergence
    }
    
    keys, values = zip(*grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n[STARTING GRID SEARCH] Testing {len(permutations)} total combinations...")
    
    best_global_mae = float('inf')
    best_params_record = None
    
    for i, params in enumerate(permutations):
        print(f"\n==================================================")
        print(f" RUN {i+1}/{len(permutations)} | Params: {params}")
        print(f"==================================================")
        
        tide_model = TiDE(
            h=OUTPUT_CHUNK,
            input_size=INPUT_CHUNK,
            hist_exog_list=hist_exog,
            futr_exog_list=futr_exog,
            
            # --- Injected Grid Params ---
            hidden_size=params['hidden_size'],          
            temporal_width=params['temporal_width'],         
            learning_rate=params['learning_rate'],
            # ----------------------------
            
            decoder_output_dim=32,    
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), 
            max_steps=2000,           
            batch_size=512,           
            scaler_type='robust',     
            early_stop_patience_steps=3, 
            
            # H100 Flags
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed"
        )
        
        nf = NeuralForecast(models=[tide_model], freq='1h')
        
        # 1. Train the model
        nf.fit(df=df_train, val_size=OUTPUT_CHUNK)
        
        # 2. Evaluate against the Grid Validation set
        print("Generating Validation Forecast...")
        futr_df = nf.make_future_dataframe(df=df_train)
        futr_df['hour'] = futr_df['ds'].dt.hour
        futr_df['day'] = futr_df['ds'].dt.day
        futr_df['weekday'] = futr_df['ds'].dt.weekday
        futr_df['month'] = futr_df['ds'].dt.month
        
        static_coords = df_train[['unique_id', 'latitude', 'longitude']].drop_duplicates()
        futr_df = pd.merge(futr_df, static_coords, on='unique_id', how='left')
        
        forecasts = nf.predict(df=df_train, futr_df=futr_df).reset_index()
        median_col = next((c for c in forecasts.columns if '0.5' in c or '50' in c or 'median' in c.lower()), None)
        forecasts = forecasts.rename(columns={median_col: 'prediction'})
        
        # Calculate MAE for this run (Using PM2.5 as the primary benchmark)
        eval_df = pd.merge(forecasts[['unique_id', 'ds', 'prediction']], df_val[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')
        pm25_df = eval_df[eval_df['unique_id'].str.contains("PM2_5")]
        
        run_mae = mean_absolute_error(pm25_df['y'], pm25_df['prediction'])
        print(f"  Run {i+1} PM2.5 MAE Score: {run_mae:.2f}")
        
        # 3. Check if it's the new champion
        if run_mae < best_global_mae:
            print(f" NEW BEST MODEL DETECTED! (Improved from {best_global_mae:.2f} to {run_mae:.2f})")
            best_global_mae = run_mae
            best_params_record = params
            
            # Save it to the Best Directory
            nf.save(path=BEST_MODEL_DIR, model_index=None, overwrite=True, save_dataset=False)
            print(f" Champion weights secured in {BEST_MODEL_DIR}")
        else:
            print(" Did not beat the champion. Discarding weights.")

    print("\n==================================================")
    print("  GRID SEARCH COMPLETE  ")
    print(f" Winning Score (PM2.5 MAE): {best_global_mae:.2f}")
    print(f" Winning Params: {best_params_record}")
    print(f" The optimized model is safely stored in '{BEST_MODEL_DIR}'")
    print("==================================================")

if __name__ == "__main__":
    import torch
    torch.set_float32_matmul_precision('medium')
    run_grid_search()