## Project flow — atmosphere

This document explains the data flow, which scripts produce which artifacts, and how scripts are linked together. It is intended to help you pick up the pipeline quickly and to know what to run to create the 1-hour dataset you mentioned.

---

## High-level overview

- Raw data (CPCB device CSVs) are in `data/cpcb_downloads/` organized by city and station.
- A preprocessing pipeline ingests these CSVs, cleans timestamps and numeric types, aligns to a 15-minute grid, interpolates missing values and adds basic features → `data/processed/clean_data_15min.parquet`.
- Spatial geocoding (station lat/lon) is produced to `data/processed/stations_geocoded.csv` and merged into the main dataset → `data/processed/clean_data_geo.parquet`.
- A spatial graph (edges between stations within ~50 km) is created → `data/processed/spatial/spatial_graph_edges.parquet`.
- Physics-aware advection features are computed using the spatial graph and the gridded observations → `data/processed/features/advection_features.parquet`.
- The advection-aware matrix is prepared for the Temporal Fusion Transformer (TFT) model (time index, dtypes) → `data/processed/tft/tft_ready_data.parquet`.
- A TFT dataset configuration step sanitizes/imputes and serializes dataset parameters → `data/processed/tft/tft_ready_data_sanitized.parquet` and `data/processed/tft/tft_dataset_params.pkl`.
- From the TFT-ready sanitized parquet you can (optionally) downsample to hour frequency for TiDE / other models → `data/processed/tide_ready_1hr.parquet` (produced by `prep_hourly_data.py`).
- The final model training script reads the sanitized parquet & params and trains (saves best model to `models/tft`).

---

## Script-by-script mapping

- `preprocess_cpcb.py`
  - Purpose: Load raw CPCB CSV files from `data/cpcb_downloads/`, clean columns, parse station names, fix timestamps, align to a strict 15-minute grid, interpolate missing values, add time & wind features.
  - Inputs: all CSV files under `data/cpcb_downloads/`
  - Output: `data/processed/clean_data_15min.parquet`

- `spatial_geocode.py`
  - Purpose: Build `stations_geocoded.csv` by geocoding unique station names (via Nominatim) and/or merge that CSV into the parquet.
  - Inputs: `data/processed/clean_data_15min.parquet`; optionally re-downloads via the geocoding API
  - Outputs: `data/processed/stations_geocoded.csv` (when building) and `data/processed/clean_data_geo.parquet` (when merged)
  - Notes: The script will pause to avoid violating Nominatim ToS (1+ second sleep). If `stations_geocoded.csv` has blanks, run `fix_csv_coordinates.py` to patch common stations.

- `fix_csv_coordinates.py`
  - Purpose: Apply manual coordinate overrides for station names that failed geocoding.
  - Inputs: `data/processed/stations_geocoded.csv`
  - Output: updated `data/processed/stations_geocoded.csv`

- `merge_final.py`
  - Purpose: Simple merge of `clean_data_15min.parquet` with `stations_geocoded.csv` into `clean_data_geo.parquet`.
  - Inputs: `data/processed/clean_data_15min.parquet`, `data/processed/stations_geocoded.csv`
  - Output: `data/processed/clean_data_geo.parquet`

- `build_spatial_graph.py`
  - Purpose: Build a borderless spatial graph by calculating pairwise distances and bearings between stations found in `clean_data_geo.parquet` and emitting edges under a radius threshold (50 km by default).
  - Inputs: `data/processed/clean_data_geo.parquet`
  - Output: `data/processed/spatial/spatial_graph_edges.parquet`

- `build_advection_features.py`
  - Purpose: Use the spatial edges + wind vectors + pollutant concentrations (from `clean_data_geo.parquet`) to compute pollutant-specific advection (incoming flux) features per station and timestamp.
  - Inputs: `data/processed/clean_data_geo.parquet`, `data/processed/spatial/spatial_graph_edges.parquet`
  - Output: `data/processed/features/advection_features.parquet` (also writes combined main DF with adv features)

- `prep_tft_dataset.py`
  - Purpose: Read the advection-feature-augmented matrix, create a strict `time_idx` (integer index), cast dtypes, add / ensure time covariates. Produces a TFT-ready parquet.
  - Inputs: `data/processed/features/advection_features.parquet`
  - Output: `data/processed/tft/tft_ready_data.parquet`

- `build_tft_dataset.py`
  - Purpose: Sanitize column names, impute missing values per station, build PyTorch Forecasting TimeSeriesDataSet and save dataset parameters (pickle). Also writes a sanitized parquet ready for model ingestion.
  - Inputs: `data/processed/tft/tft_ready_data.parquet`
  - Outputs: `data/processed/tft/tft_ready_data_sanitized.parquet`, `data/processed/tft/tft_dataset_params.pkl`

- `prep_hourly_data.py` (the script you're currently editing)
  - Purpose: Downsample the TFT sanitized parquet to 1-hour intervals (averaging physics variables and advective inflows), merge spatial metadata, add temporal covariates and `station_id_code` for integer embeddings used by TiDE.
  - Inputs: `data/processed/tft/tft_ready_data_sanitized.parquet`, `data/processed/stations_geocoded.csv`
  - Output: `data/processed/tide_ready_1hr.parquet`
  - When to run: After `build_tft_dataset.py` finishes (so sanitized data exists). This is the step you asked about for 1-hour data.

- `train_tft.py`
  - Purpose: Load dataset parameters and sanitized parquet, create dataloaders and train a Temporal Fusion Transformer via Lightning. Saves best model weights under `models/tft`.
  - Inputs: `data/processed/tft/tft_dataset_params.pkl`, `data/processed/tft/tft_ready_data_sanitized.parquet`
  - Outputs: model checkpoints in `models/tft`

- `verify_*.py` scripts
  - Purpose: Small sanity-check utilities for verifying expected outputs exist and some basic integrity checks (e.g., `verify_preprocess.py`, `verify_spatial_graph.py`, `verify_tft_dataset.py`, `verify_tft_tensors.py`, `verify_final_dataset.py`, `verify_advection.py`). Run these after each major step to catch regressions.

---

## Recommended run order (full pipeline)

1. `python preprocess_cpcb.py`
   - Produces `data/processed/clean_data_15min.parquet`
2. `python spatial_geocode.py` (or create `stations_geocoded.csv` manually)
   - If blanks exist, run `python fix_csv_coordinates.py`, edit CSV, then run `python spatial_geocode.py` with `merge` to create `clean_data_geo.parquet` OR run `python merge_final.py` which does a straightforward merge.
3. `python build_spatial_graph.py`
   - Produces `data/processed/spatial/spatial_graph_edges.parquet`
4. `python build_advection_features.py`
   - Produces `data/processed/features/advection_features.parquet`
5. `python prep_tft_dataset.py`
   - Produces `data/processed/tft/tft_ready_data.parquet`
6. `python build_tft_dataset.py`
   - Produces sanitized parquet and dataset params: `tft_ready_data_sanitized.parquet` and `tft_dataset_params.pkl`
7. (Optional) `python prep_hourly_data.py`
   - Uses sanitized parquet to produce `data/processed/tide_ready_1hr.parquet` — your 1-hour dataset
8. `python train_tft.py` (on a GPU-equipped machine)

Run the `verify_*.py` scripts as checkpoints after steps 1, 3, 4, 6 to catch problems early.

---

## Important notes, assumptions & edge-cases

- Station name matching: merges assume the `station` field in the parquet exactly matches the `station` column in `stations_geocoded.csv`. If station names differ (extra suffixes, cases), merges will produce NaNs for lat/lon. Use `fix_csv_coordinates.py` and manual CSV edits to reconcile names.
- Timestamp parsing: `preprocess_cpcb.py` attempts to find any column containing `Timestamp` and normalize to `timestamp`. Downstream scripts expect a `timestamp` column (and later `timestamp` is used to create `time_idx`). Be careful if you changed original CSV column names.
- Column names sanitized: `build_tft_dataset.py` replaces special characters in column names and expects specific pollutant column names (in code it's mapping to target names such as `PM2_5_ug_m3` etc.). If your source naming differs, update `prep_tft_dataset.py` or `build_tft_dataset.py` accordingly.
- Advection physics: `build_advection_features.py` expects `wind_x`, `wind_y` and pollutant columns named the same as in `POLLUTANT_PHYSICS`. Check those keys if you rename pollutants.
- Memory: Some steps read entire parquets into memory (advection computation, TFT dataset construction) — ensure you have enough RAM. If you face memory issues, consider working city-by-city or using chunked processing.

---

## Quick commands (Windows cmd.exe)

Run preprocessing and produce 1-hour dataset (minimal):

```
python preprocess_cpcb.py
python merge_final.py   # or use spatial_geocode.py -> fix_csv_coordinates.py -> spatial_geocode.py (merge)
python build_spatial_graph.py
python build_advection_features.py
python prep_tft_dataset.py
python build_tft_dataset.py
python prep_hourly_data.py
```

If you only want the 1-hour dataset and `data/processed/tft/tft_ready_data_sanitized.parquet` already exists, just run:

```
python prep_hourly_data.py
```

---

## Next steps / suggested improvements

- Add a small `README.md` or Makefile/CLI wrapper to run the pipeline in order (makes reproducing runs easier).
- Add unit/integration checks for column name expectations before merging to make the pipeline safer.
- Allow `prep_hourly_data.py` to accept input/output paths as CLI arguments for flexibility.

---

If you want, I can:

- commit this `PROJECT_FLOW.md` (already added here),
- add a small orchestrator script (`run_pipeline.py`) that runs the steps in the correct order with simple flags (and safe checks), or
- modify `prep_hourly_data.py` to accept CLI args and improve robustness for missing columns.

Tell me which of these you'd like next.
