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