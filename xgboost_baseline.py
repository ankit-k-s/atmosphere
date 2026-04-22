import pandas as pd
import numpy as np
import os
import re
import warnings
import xgboost as xgb
from mlforecast import MLForecast
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Suppress expected XGBoost lag-dropping warnings for short chunks
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/tft/tide_ready_1hr.parquet"

POLLUTANTS = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']
# XGBoost Baseline: We drop the unknown future weather, relying purely on coordinates and time.
static_exog = ['latitude', 'longitude'] 

def prep_nixtla_format(df):
    print("Melting perfect tensor into XGBoost strict-long format...")
    id_vars = ['unique_id', 'timestamp'] + static_exog
    df_melted = df.melt(
        id_vars=id_vars, value_vars=POLLUTANTS,
        var_name='pollutant_type', value_name='y'
    )
    df_melted['unique_id'] = df_melted['unique_id'] + "_" + df_melted['pollutant_type']
    df_melted = df_melted.drop(columns=['pollutant_type'])
    df_melted = df_melted.rename(columns={'timestamp': 'ds'})
    df_melted = df_melted.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    return df_melted

def run_xgboost_pipeline():
    print("=== XGBOOST BASELINE PIPELINE ===")
    
    print("Loading 1-hour ML Tensor...")
    df = pd.read_parquet(INPUT_FILE)
    df_nixtla = prep_nixtla_format(df)
    
    # ---------------------------------------------------------
    # STRICT TEMPORAL SPLIT (Exact same logic as TiDE)
    # ---------------------------------------------------------
    print("\n[INITIATING STRICT TEMPORAL SPLIT]")
    TEST_HOURS = 24 * 30  
    
    max_date = df_nixtla['ds'].max()
    test_cutoff = max_date - pd.Timedelta(hours=TEST_HOURS)
    
    df_train = df_nixtla[df_nixtla['ds'] <= test_cutoff].reset_index(drop=True)
    df_test = df_nixtla[df_nixtla['ds'] > test_cutoff].reset_index(drop=True)
    
    # ---------------------------------------------------------
    # XGBOOST & MLFORECAST CONFIGURATION
    # ---------------------------------------------------------
    print("\nInstantiating XGBoost with MLForecast Wrapper...")
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        tree_method='hist', 
        device='cuda',       # Train the trees on the H100
        random_state=42
    )
    
    mlf = MLForecast(
        models=[xgb_model],
        freq='1h',
        # Explicitly define the lag structure for XGBoost to look back in time
        lags=[1, 2, 3, 24, 48, 96], 
        # MLForecast will automatically generate these time features for the future!
        date_features=['hour', 'day', 'dayofweek', 'month'] 
    )
    
    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------
    print("\n[INITIATING GPU TRAINING SEQUENCE]")
    mlf.fit(
        df_train,
        id_col='unique_id',
        time_col='ds',
        target_col='y',
        static_features=static_exog
    )
    
    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------
    print("\n[TRAINING COMPLETE] Generating 96-hour Zero-Shot Forecast...")
    
    # No manual X_df required! MLForecast uses date_features to auto-generate the future.
    forecasts = mlf.predict(h=96)
    forecasts = forecasts.rename(columns={'XGBRegressor': 'prediction'})
    
    print("Aligning Forecasts with Real Ground Truth...")
    evaluation_df = pd.merge(
        forecasts[['unique_id', 'ds', 'prediction']],
        df_test[['unique_id', 'ds', 'y']],
        on=['unique_id', 'ds'],
        how='inner'
    )
    
    evaluation_df['pollutant'] = evaluation_df['unique_id'].apply(lambda x: re.split(r'_c\d+_', x)[-1])

    print("\n==================================================")
    print("  XGBOOST 96-HOUR FORECAST ACCURACY REPORT ")
    print("==================================================\n")
    
    for pol in POLLUTANTS:
        pol_data = evaluation_df[evaluation_df['pollutant'] == pol].copy()
        
        if len(pol_data) > 0:
            pol_data.loc[pol_data['prediction'] < 0, 'prediction'] = 0
            
            mae = mean_absolute_error(pol_data['y'], pol_data['prediction'])
            rmse = root_mean_squared_error(pol_data['y'], pol_data['prediction'])
            
            print(f"--- {pol} ---")
            print(f"  MAE  (Avg Hourly Error): {mae:.2f}")
            print(f"  RMSE (Big Spike Error):  {rmse:.2f}")
            print(f"  Average Real Baseline:   {pol_data['y'].mean():.2f}")
            print(f"  Average Predicted:       {pol_data['prediction'].mean():.2f}\n")

if __name__ == "__main__":
    run_xgboost_pipeline()