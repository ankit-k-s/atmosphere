import pandas as pd

PRED_PATH = "data/processed/evaluation/tide_optimized_96hr_results.parquet"
WIND_PATH = "data/processed/wind/inference_wind_2026.parquet"

KNOWN_POLLUTANTS = ['PM2_5_ug_m3', 'PM10_ug_m3', 'NO2_ug_m3', 'SO2_ug_m3', 'CO_mg_m3']

def extract_pollutant(uid):
    for p in KNOWN_POLLUTANTS:
        if uid.endswith(p):
            return p
    return "UNKNOWN"

def load_data():
    df = pd.read_parquet(PRED_PATH)
    wind = pd.read_parquet(WIND_PATH)

    # 1. Safely extract pollutant and base_id
    df["pollutant"] = df["unique_id"].apply(extract_pollutant)
    df["base_id"] = df.apply(lambda row: row["unique_id"].replace(f"_{row['pollutant']}", ""), axis=1)

    # 2. Pivot to wide format
    df_wide = df.pivot_table(
        index=["base_id", "ds"],
        columns="pollutant",
        values="prediction"
    ).reset_index()

    # 3. Extract city and station (Splits ONLY on the first underscore)
    split = df_wide["base_id"].str.split("_", n=1, expand=True)
    df_wide["city"] = split[0]
    df_wide["station"] = split[1]

    # 4. Merge wind and geo
    wind = wind.rename(columns={"timestamp": "ds"})
    df_wide = df_wide.merge(
        wind[["unique_id", "ds", "wind_x", "wind_y", "latitude", "longitude"]],
        left_on=["base_id", "ds"],
        right_on=["unique_id", "ds"],
        how="left"
    )

    df_wide = df_wide.drop(columns=["unique_id"])
    return df_wide