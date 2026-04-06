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