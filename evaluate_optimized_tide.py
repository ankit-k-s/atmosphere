import pandas as pd
import numpy as np
import os
import re
from neuralforecast import NeuralForecast
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# --- UPDATED PATHS FOR OPTIMIZED MODEL ---
MODEL_DIR = "models/tide_optimized_best"
INPUT_FILE = "data/processed/tft/tide_ready_1hr.parquet"
TEST_VAULT = "data/processed/tft/evaluation/test_holdout_set.parquet"
OUTPUT_FILE = "data/processed/evaluation/tide_optimized_96hr_results.parquet" # Saved separately!

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

def evaluate_optimized_model():
    print(f"Loading OPTIMIZED TiDE Model from {MODEL_DIR}...")
    nf = NeuralForecast.load(path=MODEL_DIR)
    
    print("Loading Historical Data to prime the model's memory...")
    df = pd.read_parquet(INPUT_FILE)
    df_nixtla = prep_nixtla_format(df)
    
    # Re-create the exact December cutoff point for an apples-to-apples comparison
    TEST_HOURS = 24 * 30  
    max_date = df_nixtla['ds'].max()
    test_cutoff = max_date - pd.Timedelta(hours=TEST_HOURS)
    
    # History strictly up to Nov 30th
    df_train_val = df_nixtla[df_nixtla['ds'] <= test_cutoff].reset_index(drop=True)
    
    print("Loading Vaulted December Ground Truth Test Data...")
    df_test = pd.read_parquet(TEST_VAULT)
    
    print("Generating exact future exogenous template using Nixtla...")
    futr_df = nf.make_future_dataframe(df=df_train_val)
    futr_df['hour'] = futr_df['ds'].dt.hour
    futr_df['day'] = futr_df['ds'].dt.day
    futr_df['weekday'] = futr_df['ds'].dt.weekday
    futr_df['month'] = futr_df['ds'].dt.month
    
    static_coords = df_train_val[['unique_id', 'latitude', 'longitude']].drop_duplicates()
    futr_df = pd.merge(futr_df, static_coords, on='unique_id', how='left')
    
    print("\nGenerating 96-hour Zero-Shot Forecast...")
    forecasts = nf.predict(df=df_train_val, futr_df=futr_df)
    forecasts = forecasts.reset_index()
    
    # Dynamic column selection for the median
    tide_cols = [c for c in forecasts.columns if 'TiDE' in c]
    median_col = next((c for c in tide_cols if '0.5' in c or '50' in c or 'median' in c.lower()), tide_cols[0])
    forecasts = forecasts.rename(columns={median_col: 'prediction'})
    
    print("Aligning Forecasts with Real Ground Truth...")
    evaluation_df = pd.merge(
        forecasts[['unique_id', 'ds', 'prediction']],
        df_test[['unique_id', 'ds', 'y']],
        on=['unique_id', 'ds'],
        how='inner'
    )
    
    evaluation_df['pollutant'] = evaluation_df['unique_id'].apply(lambda x: re.split(r'_c\d+_', x)[-1])

    print("\n=========================================================")
    print("  OPTIMIZED TiDE 96-HOUR FORECAST ACCURACY REPORT  ")
    print("=========================================================\n")
    
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
            
    os.makedirs("data/processed/evaluation", exist_ok=True)
    evaluation_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Optimized evaluation data securely saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    evaluate_optimized_model()