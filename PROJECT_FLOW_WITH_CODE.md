## Project Flow with Code — atmosphere

This file embeds the key pipeline scripts with short explanations so you can read the code in one place and understand how each script links to the next.

IMPORTANT: These are verbatim copies of the repository scripts at the time this file was created. To run, use the original `.py` files in the repo.

---

### 1) preprocess_cpcb.py — load raw CSVs, clean, align to 15-min grid, interpolate

Purpose: read raw CPCB CSVs from `data/cpcb_downloads/`, clean column names, parse station names, normalize timestamps, align to a strict 15-minute grid (creating NaNs for missing readings), interpolate missing values, add time & wind features, and save `data/processed/clean_data_15min.parquet`.

```python
import os
import pandas as pd
import numpy as np

DATA_PATH = r"data/cpcb_downloads"
OUTPUT_PATH = r"data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# -------------------------------
# CLEAN COLUMN NAMES
# -------------------------------
def clean_columns(df):
    df.columns = [
        c.strip()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("Â", "")
        for c in df.columns
    ]
    return df

# -------------------------------
# PARSE STATION NAME
# -------------------------------
def parse_station_name(folder_name):
    return folder_name.split(",")[0].strip()

# -------------------------------
# LOAD ALL DATA
# -------------------------------
def load_all_data():
    all_data = []
    print(f"\nChecking path: {DATA_PATH}")

    for city in os.listdir(DATA_PATH):
        city_path = os.path.join(DATA_PATH, city)

        if not os.path.isdir(city_path):
            continue

        print(f"\nProcessing city: {city}")

        for station_folder in os.listdir(city_path):
            station_path = os.path.join(city_path, station_folder)

            if not os.path.isdir(station_path):
                continue

            print(f"   Station: {station_folder}")
            station_name = parse_station_name(station_folder)
            csv_files = [f for f in os.listdir(station_path) if f.lower().endswith(".csv")]

            if len(csv_files) == 0:
                print("      No CSV files found")
                continue

            for file in csv_files:
                file_path = os.path.join(station_path, file)
                print(f"      Loading: {file}")

                try:
                    df = pd.read_csv(file_path, encoding="latin1")

                    if df.empty:
                        print("      Empty file skipped")
                        continue

                    df = clean_columns(df)
                    df["city"] = city
                    df["station"] = station_name

                    all_data.append(df)

                except Exception as e:
                    print(f"      Error: {e}")

    print(f"\nTotal loaded files: {len(all_data)}")

    if len(all_data) == 0:
        raise Exception("No valid CSV files found! Check your DATA_PATH.")

    return pd.concat(all_data, ignore_index=True)

# -------------------------------
# FIX TIMESTAMP
# -------------------------------
def fix_timestamp(df):
    print("\nFixing timestamp...")

    for col in df.columns:
        if "Timestamp" in col or "timestamp" in col:
            df = df.rename(columns={col: "timestamp"})
            break

    if "timestamp" not in df.columns:
        raise Exception("Timestamp column not found!")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True, format="mixed")

    before = len(df)
    df = df.dropna(subset=["timestamp"])
    after = len(df)

    print(f"   Removed {before - after} invalid timestamps")

    return df

# -------------------------------
# CONVERT NUMERIC
# -------------------------------
def convert_numeric(df):
    print("\nConverting numeric columns...")

    for col in df.columns:
        if col not in ["timestamp", "city", "station"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# -------------------------------
# ALIGN TO STRICT 15-MIN GRID
# -------------------------------
def align_time_grid(df):
    print("\nAligning to strict 15-minute grid to expose missing sensor drops...")

    df = df.set_index("timestamp")
    
    # This forces the data onto a perfect 15-min grid. 
    # If a sensor skipped a reading, this creates a row with NaNs so we can interpolate it next.
    df = df.groupby(["city", "station"]).resample("15min").mean(numeric_only=True)
    
    df = df.reset_index()

    return df

# -------------------------------
# HANDLE MISSING
# -------------------------------
def handle_missing(df):
    print("\nHandling missing values...")

    df = df.sort_values(["city", "station", "timestamp"])
    df = df.set_index("timestamp")

    # Prevent Pandas FutureWarnings
    df = df.infer_objects(copy=False)

    df = df.groupby(["city", "station"], group_keys=False).apply(
        lambda g: g.interpolate(method="time", limit=3)
    )

    df = df.reset_index()

    return df

# -------------------------------
# FEATURES
# -------------------------------
def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    return df

def add_wind_features(df):
    ws_col = None
    wd_col = None

    for col in df.columns:
        if "WS" in col:
            ws_col = col
        if "WD" in col:
            wd_col = col

    if ws_col and wd_col:
        df["wind_x"] = df[ws_col] * np.cos(np.radians(df[wd_col]))
        df["wind_y"] = df[ws_col] * np.sin(np.radians(df[wd_col]))
    else:
        print("   Wind columns not found")

    return df

# -------------------------------
# MAIN
# -------------------------------
def main():
    print("\nStarting preprocessing pipeline...")

    df = load_all_data()
    df = fix_timestamp(df)
    df = convert_numeric(df)
    
    # 1. Align the grid first to create empty rows where sensors failed
    df = align_time_grid(df)
    
    # 2. Interpolate over the newly created empty rows
    df = handle_missing(df)
    
    df = add_time_features(df)
    df = add_wind_features(df)

    output_file = os.path.join(OUTPUT_PATH, "clean_data_15min.parquet")

    print(f"\nSaving to {output_file}")
    df.to_parquet(output_file, index=False)

    print("\nDONE — 15-MIN DATA READY")

if __name__ == "__main__":
    main()
```

---

### 2) spatial_geocode.py — geocode unique station names and merge lat/lon

Purpose: collect station list, call Nominatim (OpenStreetMap) to get lat/lon, save `stations_geocoded.csv`. Can also merge the CSV into `clean_data_geo.parquet`.

```python
import pandas as pd
import time
import os
from geopy.geocoders import Nominatim

INPUT_PARQUET = "data/processed/clean_data_15min.parquet"
GEO_CSV = "data/processed/stations_geocoded.csv"
OUTPUT_PARQUET = "data/processed/clean_data_geo.parquet"

def build_geocoding_csv():
    print(" Extracting unique stations...")
    df = pd.read_parquet(INPUT_PARQUET, columns=["city", "station"])
    stations = df.drop_duplicates().reset_index(drop=True)
    
    print(f"Found {len(stations)} stations. Connecting to OpenStreetMap...")
    # Nominatim requires a custom user agent to avoid being blocked
    geolocator = Nominatim(user_agent="bits_pilani_atmos_track_research")
    
    latitudes = []
    longitudes = []
    
    for i, row in stations.iterrows():
        query = f"{row['station']}, {row['city']}, India"
        try:
            location = geolocator.geocode(query, timeout=10)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
                print(f"   [{i+1}/{len(stations)}] {query} → {location.latitude:.2f}, {location.longitude:.2f}")
            else:
                latitudes.append(None)
                longitudes.append(None)
                print(f"   [{i+1}/{len(stations)}] Not found: {query}")
        except Exception as e:
            latitudes.append(None)
            longitudes.append(None)
            print(f"   [{i+1}/{len(stations)}] Error: {query} → {e}")
        
        # STRICT ToS LIMIT: Nominatim will permanently ban your IP if you go faster than 1 sec
        time.sleep(1.2) 
        
    stations["latitude"] = latitudes
    stations["longitude"] = longitudes
    stations.to_csv(GEO_CSV, index=False)
    print(f"\n Geocoding complete. Saved to {GEO_CSV}")
    print(" PLEASE OPEN THIS CSV AND CHECK FOR MISSING (BLANK) LAT/LON VALUES.")
    print("Fix any abbreviations (e.g., 'DTU' -> 'Delhi Technological University') and run this script again.")

def merge_to_parquet():
    print(f"\n Reading {GEO_CSV} and merging with main dataset...")
    df = pd.read_parquet(INPUT_PARQUET)
    geo = pd.read_csv(GEO_CSV)
    
    # Merge based on city and station
    df = df.merge(geo, on=["city", "station"], how="left")
    
    missing = df["latitude"].isna().sum()
    if missing > 0:
        print(f" WARNING: {missing:,} rows in your dataset still lack spatial coordinates!")
        print("You must fix the missing rows in the CSV before proceeding to Kriging.")
    
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f" Spatial Dataset Locked: {OUTPUT_PARQUET}")

def main():
    # Intelligent Workflow:
    # If the CSV doesn't exist, build it.
    if not os.path.exists(GEO_CSV):
        build_geocoding_csv()
    else:
        # If it exists, ask the user if they want to merge or re-download
        choice = input(f" '{GEO_CSV}' found. Have you fixed the missing coordinates? \nType 'merge' to integrate into Parquet, or 'rebuild' to hit the API again: ").strip().lower()
        
        if choice == 'merge':
            merge_to_parquet()
        elif choice == 'rebuild':
            build_geocoding_csv()
        else:
            print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
```

---

### 3) fix_csv_coordinates.py — manual overrides for missing geocodes

Purpose: apply a small whitelist of manual lat/lon overrides to common station names that Nominatim fails to locate.

```python
import pandas as pd
import os

# --- PATHS ---
GEO_CSV = "data/processed/stations_geocoded.csv"

# --- THE CHEAT SHEET (7 DECIMAL PLACES) ---
MANUAL_OVERRIDES = {
    "Raikhad": (23.0238000, 72.5797000),
    "SAC ISRO Bopal": (23.0294000, 72.4646000),
    "SAC ISRO Satellite": (23.0250000, 72.5200000),
    "SVPI Airport Hansol": (23.0674000, 72.6320000),
    "Sardar Vallabhbhai Patel Stadium": (23.0916000, 72.5976000),
    "BWSSB Kadabesanahalli": (12.9354000, 77.6946000),
    "Sanegurava Halli": (12.9856000, 77.5342000),
    "Alandur Bus Depot": (13.0031000, 80.2014000),
    "Manali Village": (13.1770000, 80.2630000),
    "Velachery Res. Area": (12.9750000, 80.2200000),
    "IHBAS": (28.6823000, 77.3183000), 
    "IHBAS, Dilshad Garden": (28.6823000, 77.3183000),
    "DTU": (28.7501000, 77.1177000),
    "CRRI Mathura Road": (28.5535000, 77.2790000),
    "NSIT Dwarka": (28.6091000, 77.0350000),
    "Sonia Vihar": (28.7329000, 77.2472000),
    "Bollaram Industrial Area": (17.5501000, 78.3314000),
    "ICRISAT Patancheru": (17.5111000, 78.2752000),
    "IDA Pashamylaram": (17.5285000, 78.1901000),
    "IITH Kandi": (17.5947000, 78.1230000),
    "Kompally Municipal Office": (17.5457000, 78.4831000),
    "Nacharam_TSIIC IALA": (17.4243000, 78.5714000),
    "Chhatrapati Shivaji Intl. Airport (T2)": (19.0974000, 72.8743000),
    "Siddharth Nagar-Worli": (19.0146000, 72.8173000),
    "Mhada Colony": (18.6015000, 73.8214000),
    "Panchawati_Pashan": (18.5363000, 73.8054000),
    "Vasai West": (19.3840000, 72.8258000),
}

def fix_csv():
    print(" Initializing CSV Patch Script...")
    
    if not os.path.exists(GEO_CSV):
        print(f" ERROR: Cannot find {GEO_CSV}.")
        return

    # Load the CSV generated by the web scraper
    geo_df = pd.read_csv(GEO_CSV)
    
    missing_mask = geo_df['latitude'].isna()
    missing_count = missing_mask.sum()
    print(f"Found {missing_count} stations missing coordinates.")
    
    patched_count = 0
    for idx, row in geo_df[missing_mask].iterrows():
        station_name = str(row['station']).strip()
        
        if station_name in MANUAL_OVERRIDES:
            lat, lon = MANUAL_OVERRIDES[station_name]
            geo_df.at[idx, 'latitude'] = lat
            geo_df.at[idx, 'longitude'] = lon
            print(f"   PATCHED: {station_name} -> {lat:.7f}, {lon:.7f}")
            patched_count += 1
        else:
            print(f"   UNKNOWN: '{station_name}' is not in the cheat sheet!")

    # Save the patched CSV back with exactly 7 decimal places for consistency
    geo_df.to_csv(GEO_CSV, index=False, float_format='%.7f')
    print(f"\n Saved {patched_count} patches to {GEO_CSV}.")

    # Verify if we fixed everything
    remaining_missing = geo_df['latitude'].isna().sum()
    if remaining_missing == 0:
        print("\n SUCCESS: stations_geocoded.csv is 100% complete and perfectly formatted.")
    else:
        print(f"\n WARNING: {remaining_missing} stations are still missing.")

if __name__ == "__main__":
    fix_csv()
```

---

### 4) merge_final.py — merge geocoded CSV into main 15-min parquet

Purpose: straightforward merge of `clean_data_15min.parquet` with `stations_geocoded.csv` producing `clean_data_geo.parquet` used by spatial graph & advection.

```python
import pandas as pd

INPUT_PARQUET = "data/processed/clean_data_15min.parquet"
GEO_CSV = "data/processed/stations_geocoded.csv"
OUTPUT_PARQUET = "data/processed/clean_data_geo.parquet"

print(f" Reading {GEO_CSV} and merging with main dataset...")
main_df = pd.read_parquet(INPUT_PARQUET)
geo_df = pd.read_csv(GEO_CSV)

# Merge based on city and station
final_df = main_df.merge(geo_df[['city', 'station', 'latitude', 'longitude']], 
                         on=["city", "station"], 
                         how="left")

# Save to the final geospatial parquet
final_df.to_parquet(OUTPUT_PARQUET, index=False)
print(f" SPATIAL DATASET SUCCESSFULLY LOCKED: {OUTPUT_PARQUET}")
```

---

### 5) build_spatial_graph.py — compute edges (distance, bearing) under radius

Purpose: compute pairwise distances and bearings for stations and save edges under the MAX_RADIUS_KM to `data/processed/spatial/spatial_graph_edges.parquet`.

```python
import pandas as pd
import os
import math

INPUT_PARQUET = "data/processed/clean_data_geo.parquet"
OUTPUT_DIR = "data/processed/spatial"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "spatial_graph_edges.parquet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_RADIUS_KM = 50.0 

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1) / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def build_borderless_graph():
    print(" INITIALIZING BORDERLESS SPATIAL GRAPH (50km Limit)...")
    
    df = pd.read_parquet(INPUT_PARQUET, columns=['city', 'station', 'latitude', 'longitude'])
    stations = df.drop_duplicates().reset_index(drop=True)
    
    edges = []
    num_stations = len(stations)
    
    # O(N^2) for N=132 is only ~17k iterations. Executes instantly.
    for i in range(num_stations):
        for j in range(num_stations):
            if i == j: continue 
            
            src = stations.iloc[i]
            tgt = stations.iloc[j]
            
            dist_km = haversine_distance(src['latitude'], src['longitude'], tgt['latitude'], tgt['longitude'])
            
            if dist_km <= MAX_RADIUS_KM:
                bearing_deg = calculate_bearing(src['latitude'], src['longitude'], tgt['latitude'], tgt['longitude'])
                
                edges.append({
                    'source_city': src['city'],
                    'source_station': src['station'],
                    'target_city': tgt['city'],
                    'target_station': tgt['station'],
                    'distance_km': round(dist_km, 3),
                    'bearing_degrees': round(bearing_deg, 2)
                })
                
    edges_df = pd.DataFrame(edges)
    print(f"\n Generated {len(edges_df)} borderless physical edges.")
    edges_df.to_parquet(OUTPUT_FILE, index=False)
    print(f" SPATIAL GRAPH LOCKED: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_borderless_graph()
```

---

### 6) build_advection_features.py — compute pollutant-specific incoming advection

Purpose: using `clean_data_geo.parquet` and `spatial_graph_edges.parquet`, compute directed flux and Gaussian-decayed incoming mass per pollutant (`adv_in_*`) and merge into the main data matrix saved at `data/processed/features/advection_features.parquet`.

```python
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
```

---

### 7) prep_tft_dataset.py — create strict time_idx and cast dtypes

Purpose: read advection features parquet, sort, create `time_idx` mapping of timestamps to integers (global), cast categorical & float dtypes suitable for PyTorch, and write `data/processed/tft/tft_ready_data.parquet`.

```python
import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/processed/features/advection_features.parquet"
OUTPUT_DIR = "data/processed/tft"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tft_ready_data.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_tft_dataset():
    print("INITIALIZING TEMPORAL FUSION TRANSFORMER (TFT) DATA PREPARATION...\n")
    
    print("Loading physics-aware dataset...")
    df = pd.read_parquet(INPUT_FILE)
    
    # 1. Sort strictly by Station and Time
    # This is mathematically critical for autoregressive modeling
    print("Sorting continuous temporal sequences...")
    df = df.sort_values(by=['station', 'timestamp']).reset_index(drop=True)
    
    # 2. Create the continuous Time Index (time_idx)
    # TFT requires a globally consistent integer step for time.
    # Since we have 15-min intervals, we map unique timestamps to ascending integers.
    print("Generating strictly continuous Time Index (time_idx)...")
    unique_times = np.sort(df['timestamp'].unique())
    time_map = {time: idx for idx, time in enumerate(unique_times)}
    df['time_idx'] = df['timestamp'].map(time_map)
    
    # 3. Enforce Strict Data Types for PyTorch
    print("Enforcing PyTorch-compatible data types...")
    
    # Categoricals must be strings
    categorical_cols = ['city', 'station']
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        
    # Temporal Reals (Known Future)
    known_reals = ['hour', 'day', 'weekday', 'month']
    for col in known_reals:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
            
    # Float casting for all continuous variables to save RAM and optimize GPU
    exclude_cols = categorical_cols + ['timestamp', 'time_idx']
    float_cols = [col for col in df.columns if col not in exclude_cols]
    
    for col in float_cols:
        # Downcast to float32 to prevent out-of-memory errors during tensor creation
        df[col] = df[col].astype(np.float32)
        
    # 4. Final Sanity Check
    print("\nDataset ML Schema Overview:")
    print(f"Total Rows: {len(df):,}")
    print(f"Time Index Range: {df['time_idx'].min()} to {df['time_idx'].max()}")
    print(f"Total Unique Timesteps: {len(unique_times):,}")
    
    print(f"\nSaving TFT-ready tensor dataset to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print("TFT DATA PREPARATION SUCCESSFULLY LOCKED.")

if __name__ == "__main__":
    prepare_tft_dataset()
```

---

### 8) build_tft_dataset.py — impute and build TimeSeriesDataSet parameters

Purpose: sanitize column names, impute missing float columns per station (ffill/bfill and absolute fallback), build pytorch-forecasting TimeSeriesDataSet for training and pickle the dataset parameters. Also writes sanitized parquet `tft_ready_data_sanitized.parquet`.

```python
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
```

---

### 9) prep_hourly_data.py — downsample sanitized TFT-ready data to 1-hour

Purpose: downsample the sanitized TFT-ready parquet to 1-hour intervals (averaging physics variables and advection inflows), merge station lat/lon metadata, add simple temporal covariates and integer station IDs for embedding. Produces `data/processed/tide_ready_1hr.parquet`.

```python
import pandas as pd
import os

# --- PATHS (Aligned with your directory structure) ---
INPUT_FILE = "data/processed/tft/tft_ready_data_sanitized.parquet"
GEO_FILE = "data/processed/stations_geocoded.csv"
OUTPUT_FILE = "data/processed/tide_ready_1hr.parquet"

def build_hourly_dataset():
    print(f"Loading master 15-minute dataset from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    # Ensure standard datetime column
    if 'datetime' in df.columns:
        df['ds'] = pd.to_datetime(df['datetime'])
    else:
        df['ds'] = pd.to_datetime(df['time_idx']) 
    
    print(f"Loading spatial metadata from {GEO_FILE}...")
    geo_df = pd.read_csv(GEO_FILE)
    meta = geo_df[['city', 'station', 'latitude', 'longitude']].drop_duplicates(subset=['station'])
    
    print("Downsampling to 1-Hour intervals (Averaging Physics)...")
    df = df.set_index('ds')
    
    # Safely average the scalars and vectors
    agg_dict = {
        'PM2.5': 'mean', 'PM10': 'mean', 'NO2': 'mean', 'SO2': 'mean', 'CO': 'mean',
        'wind_x': 'mean', 'wind_y': 'mean',
        'adv_in_PM2.5': 'mean', 'adv_in_PM10': 'mean', 
        'adv_in_NO2': 'mean', 'adv_in_SO2': 'mean', 'adv_in_CO': 'mean'
    }
    
    df_hourly = df.groupby('station').resample('1H').agg(agg_dict).reset_index()
    
    print("Interpolating missing sensor gaps safely per station...")
    df_hourly = df_hourly.groupby('station').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
    
    print("Merging spatial metadata...")
    df_hourly = df_hourly.merge(meta, on='station', how='left')
    
    # Generate temporal covariates for TiDE
    df_hourly['hour'] = df_hourly['ds'].dt.hour
    df_hourly['day'] = df_hourly['ds'].dt.dayofweek
    df_hourly['month'] = df_hourly['ds'].dt.month
    df_hourly['is_weekend'] = (df_hourly['ds'].dt.dayofweek >= 5).astype(int)
    
    # Create integer embeddings for the neural network
    df_hourly['station_id_code'] = df_hourly['station'].astype('category').cat.codes

    print(f"Saving TiDE-Ready 1-Hour Dataset ({len(df_hourly)} rows) to {OUTPUT_FILE}...")
    df_hourly.to_parquet(OUTPUT_FILE, index=False)
    print("Data Engineering Complete.")

if __name__ == "__main__":
    build_hourly_dataset()
```

---

### 10) train_tft.py — model training pipeline

Purpose: load dataset parameters and sanitized parquet, build DataLoaders, instantiate TemporalFusionTransformer and train with Lightning; saves best model to `models/tft`.

```python
import pandas as pd
import pickle
import os
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

PARAM_FILE = "data/processed/tft/tft_dataset_params.pkl"
DATA_FILE = "data/processed/tft/tft_ready_data_sanitized.parquet"
MODEL_DIR = "models/tft"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# H100 GPU PRODUCTION CONFIGURATION
# ---------------------------------------------------------
BATCH_SIZE = 1024         # Massive batch size to saturate 80GB VRAM
MAX_EPOCHS = 30           # Full training cycle
LEARNING_RATE = 0.03
NUM_WORKERS = 8           # High CPU thread count to keep the GPU fed
DATASET_FRACTION = 1.0    # 100% of the 13.4 million row dataset

def train_tft_model():
    print("INITIALIZING H100 GPU PRODUCTION TRAINING PROTOCOL...\n")
    
    print("Loading serialized tensors and parameters...")
    with open(PARAM_FILE, "rb") as f:
        dataset_parameters = pickle.load(f)
        
    df = pd.read_parquet(DATA_FILE)
    
    val_cutoff = df["time_idx"].max() - 2880
    train_df = df[df["time_idx"] <= val_cutoff]
    val_df = df[df["time_idx"] > val_cutoff - dataset_parameters["max_encoder_length"]]
    
    if DATASET_FRACTION < 1.0:
        print(f"Sub-sampling active. Reducing training data to {DATASET_FRACTION * 100}%...")
        unique_stations = train_df['station'].unique()
        sampled_train_list = []
        for station in unique_stations:
            station_data = train_df[train_df['station'] == station]
            cutoff_idx = int(len(station_data) * (1.0 - DATASET_FRACTION))
            sampled_train_list.append(station_data.iloc[cutoff_idx:])
        train_df = pd.concat(sampled_train_list, ignore_index=True)

    print("Constructing Multi-Threaded DataLoaders...")
    training_dataset = TimeSeriesDataSet.from_parameters(dataset_parameters, train_df)
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, predict=True, stop_randomization=True
    )
    
    # Enable persistent workers to prevent data loading bottlenecks between epochs
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS, persistent_workers=True)
    
    num_targets = len(training_dataset.target_names)
    multi_loss = MultiLoss([QuantileLoss()] * num_targets)
    
    print("\nInitializing Temporal Fusion Transformer Architecture...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=64,             
        attention_head_size=4,      
        dropout=0.1,                
        hidden_continuous_size=32,  
        loss=multi_loss,
        log_interval=50,            
        reduce_on_plateau_patience=4
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR, monitor="val_loss", save_top_k=1, mode="min", filename="best-tft-model"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger(save_dir=MODEL_DIR, name="tft_multipollutant")
    
    print("\nConfiguring Lightning GPU Trainer...")
    
    # Ensure torch matrix math takes advantage of the Hopper Tensor Cores
    torch.set_float32_matmul_precision("medium")
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",  # BFloat16 Mixed Precision for H100 acceleration
        gradient_clip_val=0.1,   
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger,
        fast_dev_run=False
    )
    
    print("\n[INITIATING NEURAL NETWORK TRAINING LOOP]")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    print("\n[TRAINING COMPLETE] Best model weights have been secured in the models/tft directory.")

if __name__ == "__main__":
    train_tft_model()
```

---

## How the scripts link together (summary)

- `preprocess_cpcb.py` → writes `clean_data_15min.parquet`.
- `spatial_geocode.py` (+ `fix_csv_coordinates.py`) → produce `stations_geocoded.csv` and `clean_data_geo.parquet` (or use `merge_final.py`).
- `build_spatial_graph.py` reads `clean_data_geo.parquet` → writes `spatial_graph_edges.parquet`.
- `build_advection_features.py` reads `clean_data_geo.parquet` + `spatial_graph_edges.parquet` → writes `features/advection_features.parquet`.
- `prep_tft_dataset.py` reads features → writes `tft_ready_data.parquet` (creates `time_idx`).
- `build_tft_dataset.py` reads `tft_ready_data.parquet` → writes sanitized parquet + `tft_dataset_params.pkl`.
- `prep_hourly_data.py` reads sanitized parquet + `stations_geocoded.csv` → writes `tide_ready_1hr.parquet` (your 1-hour dataset).
- `train_tft.py` reads sanitized parquet + params → trains model.

---

If you'd like, I can:
- commit this file (already created),
- add a small orchestrator to run the pipeline in order with flags and safety checks, or
- make `prep_hourly_data.py` accept CLI args and perform defensive checks for missing columns before running.

Which would you like next?
