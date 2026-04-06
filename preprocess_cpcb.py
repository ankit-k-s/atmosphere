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