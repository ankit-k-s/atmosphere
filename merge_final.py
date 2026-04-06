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