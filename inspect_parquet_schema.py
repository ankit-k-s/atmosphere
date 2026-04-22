import pandas as pd
import os

# -------------------------------
# CONFIG: ROOT DIRECTORY
# -------------------------------
BASE_PATH = "data/processed"

# -------------------------------
# HELPER FUNCTION
# -------------------------------
def inspect_parquet(file_path):
    print("\n" + "="*80)
    print(f"FILE: {file_path}")
    print("="*80)

    try:
        df = pd.read_parquet(file_path)

        print("\n--- SHAPE ---")
        print(df.shape)

        print("\n--- COLUMNS ---")
        for col in df.columns:
            print(col)

        print("\n--- DTYPES ---")
        print(df.dtypes)

        print("\n--- SAMPLE ROWS ---")
        print(df.head(3))

        print("\n--- MISSING % ---")
        missing = df.isna().mean() * 100
        print(missing.sort_values(ascending=False).head(10))

        print("\n--- NUMERIC SUMMARY ---")
        print(df.describe().T.head(10))

        # Key checks
        important_cols = [
            'timestamp', 'station', 'city',
            'latitude', 'longitude',
            'wind_x', 'wind_y'
        ]

        print("\n--- IMPORTANT COLUMN CHECK ---")
        for col in important_cols:
            print(f"{col}: {'YES' if col in df.columns else 'NO'}")

    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")


# -------------------------------
# WALK THROUGH DIRECTORY
# -------------------------------
for root, dirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith(".parquet"):
            full_path = os.path.join(root, file)
            inspect_parquet(full_path)

print("\nDONE: All parquet files inspected.")