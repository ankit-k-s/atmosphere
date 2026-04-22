import pandas as pd
import numpy as np
import os
import json
import re
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.interpolate import griddata

# --- CONFIGURATION ---
CITY_TARGET = "Delhi"
POLLUTANT = "PM2_5_ug_m3"

# --- REAL DATA PATHS ---
PREDICTIONS_FILE = "data/processed/evaluation/tide_96hr_results.parquet" 
WIND_FILE = "data/processed/wind/inference_wind_2026.parquet"
STATIONS_FILE = "data/processed/stations_geocoded.csv"

OUTPUT_DIR = "data/processed/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data():
    """
    Loads all datasets, normalizes IDs, and prepares them for the temporal loop.
    """
    print(f"=== INITIALIZING 96-HOUR DIGITAL TWIN: {CITY_TARGET} ===")
    
    # 1. Load Stations
    stations_df = pd.read_csv(STATIONS_FILE)
    stations_df = stations_df[stations_df['city'] == CITY_TARGET].copy()
    stations_df['base_id'] = stations_df.apply(lambda row: f"{row['city']}_{row['station']}".replace(" ", "_"), axis=1)

    # 2. Load ML Predictions
    df_pol = pd.read_parquet(PREDICTIONS_FILE)
    df_pol['ds'] = pd.to_datetime(df_pol['ds'])
    df_pol = df_pol[df_pol['pollutant'] == POLLUTANT].copy()
    
    #  FIX 2: Robust Regex Base ID Extraction
    # Splits exactly at the Nixtla chunk identifier (e.g., "_c0_") 
    df_pol['base_id'] = df_pol['unique_id'].apply(lambda x: re.split(r'_c\d+_', x)[0])

    # 3. Load Wind Data
    df_wind = pd.read_parquet(WIND_FILE)
    df_wind['timestamp'] = pd.to_datetime(df_wind['timestamp'])
    df_wind = df_wind[df_wind['city'] == CITY_TARGET].copy()
    df_wind = df_wind.rename(columns={'unique_id': 'base_id'})

    #  FIX 3: Validation Logging (Pre-Join)
    print(f" Unique Stations in TiDE Predictions: {df_pol['base_id'].nunique()}")
    print(f" Unique Stations in Wind Data: {df_wind['base_id'].nunique()}")
    print(f" Unique Stations in Geocodes: {stations_df['base_id'].nunique()}")

    return stations_df, df_pol, df_wind

def process_timestamp(target_dt, stations_df, df_pol, df_wind):
    """Aligns data and builds the Kriging grid for a single hour."""
    
    pol_slice = df_pol[df_pol['ds'] == target_dt]
    wind_slice = df_wind[df_wind['timestamp'] == target_dt]
    
    # Left Join + Explicit DropNA
    master_df = pd.merge(stations_df, pol_slice[['base_id', 'prediction']], on='base_id', how='left')
    master_df = pd.merge(master_df, wind_slice[['base_id', 'wind_x', 'wind_y']], on='base_id', how='left')
    
    master_df = master_df.dropna(subset=['prediction', 'wind_x', 'wind_y']).reset_index(drop=True)
    
    if len(master_df) < 3: 
        print(f" Warning: Insufficient data at {target_dt}. Skipping...")
        return None

    # Define bounding box
    lat_min, lat_max = master_df['latitude'].min() - 0.05, master_df['latitude'].max() + 0.05
    lon_min, lon_max = master_df['longitude'].min() - 0.05, master_df['longitude'].max() + 0.05
    
    grid_res = 50
    grid_lon, grid_lat = np.meshgrid(np.linspace(lon_min, lon_max, grid_res), np.linspace(lat_min, lat_max, grid_res))
    
    points = np.column_stack((master_df['longitude'].values, master_df['latitude'].values))
    
    #  FIX: Interpolate Pollution (Cubic with Nearest fallback)
    grid_pm25 = griddata(points, master_df['prediction'].values, (grid_lon, grid_lat), method='cubic')
    grid_pm25_nearest = griddata(points, master_df['prediction'].values, (grid_lon, grid_lat), method='nearest')
    grid_pm25 = np.where(np.isnan(grid_pm25), grid_pm25_nearest, grid_pm25)
    
    #  FIX: Interpolate Wind with Nearest fallback instead of fill_value=0!
    grid_u = griddata(points, master_df['wind_x'].values, (grid_lon, grid_lat), method='cubic')
    grid_u_nearest = griddata(points, master_df['wind_x'].values, (grid_lon, grid_lat), method='nearest')
    grid_u = np.where(np.isnan(grid_u), grid_u_nearest, grid_u)
    
    grid_v = griddata(points, master_df['wind_y'].values, (grid_lon, grid_lat), method='cubic')
    grid_v_nearest = griddata(points, master_df['wind_y'].values, (grid_lon, grid_lat), method='nearest')
    grid_v = np.where(np.isnan(grid_v), grid_v_nearest, grid_v)
    
    return {
        "timestamp": str(target_dt),
        "pm25": np.round(grid_pm25.flatten(), 2).tolist(),
        "wind_u": np.round(grid_u.flatten(), 2).tolist(),
        "wind_v": np.round(grid_v.flatten(), 2).tolist()
    }, grid_lon, grid_lat, grid_pm25, grid_u, grid_v, master_df


def render_html_snapshot(ts_str, glon, glat, gpm25, gu, gv, master_df):
    """Renders a static HTML snapshot for presentation purposes."""
    
    #  FIX: The Epsilon Patch. Add 1e-5 to prevent divide-by-zero crashes in Plotly.
    gu_safe = gu + 1e-5
    gv_safe = gv + 1e-5
    
    try:
        fig = ff.create_streamline(x=glon[0], y=glat[:, 0], u=gu_safe, v=gv_safe, arrow_scale=0.05, density=1.5, line=dict(color='rgba(255,255,255,0.7)'))
    except Exception as e:
        print(f"\n Plotly Streamline failed for {ts_str} due to math error. Rendering heatmap only.")
        fig = go.Figure() # Fallback to empty figure if wind math still fails
        
    fig.add_trace(go.Contour(
        x=glon[0], y=glat[:, 0], z=gpm25, colorscale='Inferno', opacity=0.6, showscale=True,
        colorbar=dict(title=f"{POLLUTANT} Concentration"), hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=master_df['longitude'], y=master_df['latitude'], mode='markers',
        marker=dict(size=8, color='cyan', line=dict(width=1, color='black')),
        text=[f"{name}: {val:.1f} µg/m³" for name, val in zip(master_df['station'], master_df['prediction'])], hoverinfo="text"
    ))
    
    fig.update_layout(
        title=f"TiDE Digital Twin: {CITY_TARGET} | Timestamp: {ts_str}",
        plot_bgcolor='rgb(20, 20, 20)', paper_bgcolor='rgb(20, 20, 20)', font=dict(color='white'), width=1000, height=800
    )
    
    # Only reverse data if streamline was actually added successfully
    if len(fig.data) > 2: 
        fig.data = fig.data[::-1] 
    
    html_path = os.path.join(OUTPUT_DIR, f"{CITY_TARGET}_{ts_str.replace(':', '').replace(' ', '_')}.html")
    fig.write_html(html_path)

if __name__ == "__main__":
    stations_df, df_pol, df_wind = load_and_prepare_data()
    
    #  FIX 4: The 96-Hour Loop
    # Extract unique timestamps from the prediction set and sort them
    timestamps = sorted(df_pol['ds'].unique())
    print(f"\n Processing {len(timestamps)} temporal frames for Advection Animation...")
    
    timeline_payload = []
    global_grid_bounds = None
    
    for i, ts in enumerate(timestamps):
        print(f"[{i+1}/{len(timestamps)}] Interpolating physics for {ts}...", end="\r")
        result = process_timestamp(ts, stations_df, df_pol, df_wind)
        
        if result:
            payload_data, glon, glat, gpm25, gu, gv, master_df = result
            timeline_payload.append(payload_data)
            
            if global_grid_bounds is None:
                global_grid_bounds = {
                    "grid_resolution": [glon.shape[0], glon.shape[1]],
                    "lon_min": float(np.min(glon)), "lon_max": float(np.max(glon)),
                    "lat_min": float(np.min(glat)), "lat_max": float(np.max(glat))
                }
            
            # Save HTML snapshots for the very first, middle, and last hours for visual proof
            if i == 0 or i == 48 or i == len(timestamps) - 1:
                render_html_snapshot(str(ts), glon, glat, gpm25, gu, gv, master_df)

    print("\n\nPackaging Final 96-Hour Frontend API Payload...")
    final_payload = {
        "city": CITY_TARGET,
        "pollutant": POLLUTANT,
        "spatial_meta": global_grid_bounds,
        "timeline": timeline_payload # Array of 96 hours!
    }
    
    json_path = os.path.join(OUTPUT_DIR, f"{CITY_TARGET}_96hr_digital_twin.json")
    with open(json_path, 'w') as f:
        json.dump(final_payload, f)
        
    print(f" Mission Complete. Massive 96-hour JSON saved to: {json_path}")
    print(" 3 HTML Snapshot validations saved to the visualizations folder.")