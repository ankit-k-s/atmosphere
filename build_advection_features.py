import pandas as pd
import numpy as np
import math
import os
import gc

INPUT_DATA = "data/processed/clean_data_geo.parquet"
INPUT_EDGES = "data/processed/spatial/spatial_graph_edges.parquet"
OUTPUT_DIR = "data/processed/features"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "advection_features.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pollutant-Specific Dispersion Coefficients (Sigma in km)
# Approximates dry deposition velocities and chemical lifetimes
POLLUTANT_PHYSICS = {
    'PM2.5_µg/m³': 10.0,  # Fine particles: Moderate travel
    'PM10_µg/m³': 5.0,    # Coarse particles: Settles fast (gravity)
    'NO2_µg/m³': 8.0,     # Reactive gas: Short-medium lifetime
    'SO2_µg/m³': 8.0,     # Reactive gas: Short-medium lifetime
    'CO_mg/m³': 20.0      # Stable gas: Long-range transport
}

def calculate_multipollutant_advection():
    print("INITIALIZING MULTI-POLLUTANT ADVECTION PHYSICS LAYER...\n")
    
    pollutant_cols = list(POLLUTANT_PHYSICS.keys())
    required_cols = ['city', 'station', 'timestamp', 'wind_x', 'wind_y'] + pollutant_cols
    
    print("Loading temporal and spatial datasets...")
    df = pd.read_parquet(INPUT_DATA, columns=required_cols)
    edges = pd.read_parquet(INPUT_EDGES)
    
    print("Pre-computing Bearing Unit Vectors...")
    edges['dx'] = np.sin(np.radians(edges['bearing_degrees']))
    edges['dy'] = np.cos(np.radians(edges['bearing_degrees']))
    
    # Pre-compute Gaussian decay for each specific pollutant
    for pol, sigma in POLLUTANT_PHYSICS.items():
        decay_col = f'decay_{pol}'
        edges[decay_col] = np.exp(-(edges['distance_km']**2) / (2 * (sigma**2)))
    
    cities = df['city'].unique()
    all_city_features = []
    
    for city in cities:
        print(f"Computing Fluid Dynamics for {city}...")
        
        city_data = df[df['city'] == city]
        city_edges = edges[edges['source_city'] == city]
        
        if city_edges.empty:
            continue
            
        # Create rename mapping dynamically for the merge
        rename_map = {'station': 'source_station', 'wind_x': 'src_wind_x', 'wind_y': 'src_wind_y'}
        for pol in pollutant_cols:
            rename_map[pol] = f'src_{pol}'
            
        merged = pd.merge(
            city_edges,
            city_data.rename(columns=rename_map),
            on=['source_station']
        )
        
        # 1. Vector Projection & Upwind Filter
        merged['directed_wind_speed'] = (merged['src_wind_x'] * merged['dx']) + (merged['src_wind_y'] * merged['dy'])
        merged['directed_wind_speed'] = np.maximum(0, merged['directed_wind_speed'])
        
        # 2. Compute Advection for EACH Pollutant
        advection_feature_cols = []
        for pol in pollutant_cols:
            flux_col = f'flux_{pol}'
            adv_col = f'adv_in_{pol}'
            decay_col = f'decay_{pol}'
            
            # Flux = Concentration * Directed Velocity
            merged[flux_col] = merged[f'src_{pol}'] * merged['directed_wind_speed']
            
            # Incoming Mass = Flux * Specific Gaussian Decay
            merged[adv_col] = merged[flux_col] * merged[decay_col]
            advection_feature_cols.append(adv_col)
            
        # 3. Target Aggregation
        agg_dict = {col: 'sum' for col in advection_feature_cols}
        target_features = merged.groupby(['target_station', 'timestamp']).agg(agg_dict).reset_index()
        target_features.rename(columns={'target_station': 'station'}, inplace=True)
        target_features['city'] = city
        
        all_city_features.append(target_features)
        
        del merged
        del city_data
        gc.collect()

    print("\nCompiling unified multi-pollutant advection matrix...")
    final_advection_df = pd.concat(all_city_features, ignore_index=True)
    
    print("Merging spatiotemporal features with main observational grid...")
    main_df = pd.read_parquet(INPUT_DATA)
    main_df = main_df.merge(final_advection_df, on=["city", "station", "timestamp"], how="left")
    
    # Fill missing values with 0 for all generated advection columns
    generated_cols = [f'adv_in_{pol}' for pol in pollutant_cols]
    main_df[generated_cols] = main_df[generated_cols].fillna(0)
    
    print(f"Saving finalized physics-aware matrix to {OUTPUT_FILE}...")
    main_df.to_parquet(OUTPUT_FILE, index=False)
    
    print("MULTI-POLLUTANT ADVECTION PHYSICS LAYER SUCCESSFULLY LOCKED.")

if __name__ == "__main__":
    calculate_multipollutant_advection()