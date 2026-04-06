import pandas as pd
import sys
import os

FILE_PATH = "data/processed/spatial/spatial_graph_edges.parquet"

def verify_spatial_graph():
    print("INITIALIZING SPATIAL GRAPH VERIFICATION SUITE...\n")

    if not os.path.exists(FILE_PATH):
        print(f"[ERROR] File not found: {FILE_PATH}")
        sys.exit(1)

    try:
        df = pd.read_parquet(FILE_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load Parquet file: {e}")
        sys.exit(1)

    # TEST 1: Schema and Shape
    print("=== TEST 1: DATASET SHAPE & SCHEMA ===")
    print(f"Total Edges: {df.shape[0]:,}")
    
    required_cols = ['source_city', 'source_station', 'target_city', 'target_station', 'distance_km', 'bearing_degrees']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns: {missing_cols}")
    else:
        print("[SUCCESS] All required graph columns are present.")

    # TEST 2: Distance Constraints (The 50km Rule)
    print("\n=== TEST 2: DISTANCE CONSTRAINTS ===")
    if 'distance_km' in df.columns:
        max_dist = df['distance_km'].max()
        min_dist = df['distance_km'].min()
        print(f"Distance Range: {min_dist:.3f} km to {max_dist:.3f} km")
        
        # Adding a tiny 0.01 buffer for floating point arithmetic
        if max_dist <= 50.01:
            print("[SUCCESS] All edges strictly obey the 50km threshold.")
        else:
            print(f"[ERROR] Found edges exceeding 50km (Max: {max_dist}). The threshold failed.")

    # TEST 3: Bearing Constraints
    print("\n=== TEST 3: BEARING CONSTRAINTS ===")
    if 'bearing_degrees' in df.columns:
        max_bearing = df['bearing_degrees'].max()
        min_bearing = df['bearing_degrees'].min()
        print(f"Bearing Range: {min_bearing:.2f} to {max_bearing:.2f} degrees")
        
        if min_bearing >= 0.0 and max_bearing <= 360.0:
            print("[SUCCESS] All bearings are valid compass angles (0-360).")
        else:
            print("[ERROR] Bearings outside valid 0-360 degree range detected.")

    # TEST 4: Self-Loop Check
    print("\n=== TEST 4: SELF-LOOP CHECK ===")
    self_loops = df[df['source_station'] == df['target_station']].shape[0]
    if self_loops == 0:
        print("[SUCCESS] 0 self-loops found. Graph is strictly pairwise.")
    else:
        print(f"[ERROR] Found {self_loops} self-loops. Stations should not connect to themselves.")

    # TEST 5: Cross-City Connections (The Borderless Test)
    print("\n=== TEST 5: CROSS-CITY CONNECTIONS ===")
    cross_city_edges = df[df['source_city'] != df['target_city']].shape[0]
    print(f"Total cross-city edges found: {cross_city_edges}")
    
    if cross_city_edges > 0:
        print("[SUCCESS] Borderless logic confirmed. Stations are connecting across city limits.")
    else:
        print("[INFO] No cross-city edges found. This is acceptable if all target cities are strictly >50km apart.")

    print("\n[INFO] VERIFICATION COMPLETE.")

if __name__ == "__main__":
    verify_spatial_graph()