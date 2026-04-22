import pandas as pd
import numpy as np
import requests
import time
import os

# --- INFERENCE CONFIGURATION ---
STATION_CSV_PATH = "data/processed/stations_geocoded.csv" 
OUTPUT_DIR = "data/processed/wind"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetching the wind for the TiDE evaluation window
START_DATE = "2025-12-01"
END_DATE = "2025-12-07" # Grabbing a full week to be safe!

def calculate_uv_vectors(df):
    """Decomposes Wind Speed & Direction into U (East-West) and V (North-South) vectors."""
    dir_rad = df['wind_direction_10m'] * np.pi / 180.0
    df['wind_x'] = -df['wind_speed_10m'] * np.sin(dir_rad)
    df['wind_y'] = -df['wind_speed_10m'] * np.cos(dir_rad)
    return df

def fetch_inference_wind():
    print(f"Loading stations from {STATION_CSV_PATH}...")
    stations_df = pd.read_csv(STATION_CSV_PATH)
    all_wind_data = []
    
    print(f"Fetching Visualizer Wind Data: {START_DATE} to {END_DATE}")
    
    for index, row in stations_df.iterrows():
        city = row['city']
        station = row['station']
        unique_id = f"{city}_{station}".replace(" ", "_")
        
        # Open-Meteo Historical API (It handles recent/forecast data seamlessly)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "timezone": "Asia/Kolkata" 
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                df_station = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['hourly']['time']),
                    'wind_speed_10m': data['hourly']['wind_speed_10m'],
                    'wind_direction_10m': data['hourly']['wind_direction_10m'],
                    'unique_id': unique_id,
                    'city': city
                })
                
                df_station = calculate_uv_vectors(df_station)
                df_station = df_station[['unique_id', 'city', 'timestamp', 'wind_x', 'wind_y']]
                all_wind_data.append(df_station)
        except Exception as e:
            print(f" Error for {unique_id}: {e}")
            
        time.sleep(0.5) # Rate limit protection

    master_wind_df = pd.concat(all_wind_data, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR, "inference_wind_2026.parquet")
    master_wind_df.to_parquet(output_path, index=False)
    print(f"\n Success! 2026 Wind Tensor saved to: {output_path}")

if __name__ == "__main__":
    fetch_inference_wind()