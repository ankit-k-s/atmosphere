# Atmosphere Project - Complete Codebase Summary

## Project Overview
The Atmosphere project is a comprehensive machine learning pipeline for air quality forecasting using deep neural networks (TiDE and TFT) integrated with physics-based transport modeling.

---

## 📁 Folder Structure & Contents

### 1. **data/** - Data Storage Directory
Contains all processed and raw datasets.

#### Subdirectories:
- **cpcb_downloads/** - Raw CSV files from Central Pollution Control Board (CPCB)
  - Organized by city and station
  - Contains pollutant measurements and meteorological data
  
- **processed/** - Cleaned and processed datasets
  - `clean_data_15min.parquet` - 15-minute interval data after preprocessing
  - `clean_data_geo.parquet` - Geospatially enhanced data with lat/lon coordinates
  - `clean_data_with_wind.parquet` - Data merged with IMD wind information
  
  - **spatial/** - Spatial graph and geographic data
    - `spatial_graph_edges.parquet` - Network edges (50km radius connections between stations)
  
  - **features/** - Feature engineering outputs
    - `advection_features.parquet` - Multi-pollutant advection physics features
  
  - **wind/** - Wind data from external sources
    - `inference_wind_2026.parquet` - Future wind predictions
  
  - **tft/** - Temporal Fusion Transformer datasets
    - `tft_ready_data.parquet` - ML-ready tensor for TFT training
    - `tft_ready_data_sanitized.parquet` - Sanitized version with proper imputation
    - `tide_ready_1hr.parquet` - Hourly data prepared for TiDE model
    - `tft_dataset_params.pkl` - Serialized TFT dataset parameters
    - **evaluation/** - Test data and evaluation results
      - `test_holdout_set.parquet` - Unseen 30-day test vault
      - `tide_96hr_results.parquet` - TiDE model predictions
      - `tide_optimized_96hr_results.parquet` - Optimized TiDE results
  
  - **visualizations/** - Generated outputs for visualization
    - `Delhi_96hr_digital_twin.json` - 96-hour spatiotemporal prediction grid
    - `Delhi_scenario_out.json` - Physics-based scenario simulation results

### 2. **models/** - Trained Model Checkpoints
Stores saved neural network weights and architectures.

#### Subdirectories:
- **tft/** - Temporal Fusion Transformer model
  - `best-tft-model.ckpt` - Best TFT checkpoint
  - TensorBoard logs for training monitoring

- **tide_global/** - Foundation TiDE model (trained on all data)
  - Complete trained weights
  - Ready for 96-hour forecasting

- **tide_optimized_best/** - Hyperparameter-tuned TiDE model
  - Winner of grid search optimization
  - Superior performance on validation set

- **tide_cpu_test/** - Lightweight TiDE variant for CPU smoke testing
  - Verification model for pipeline correctness

- **tide_temp/** - Temporary models during grid search
  - Discarded after optimization complete

### 3. **lightning_logs/** - PyTorch Lightning Training Logs
Stores training progress, metrics, and callbacks.

#### versions 0-21/
- TensorBoard event files
- Checkpoints during training
- Hyperparameter records
- Loss curves and validation metrics

### 4. **mdfiles/** - Documentation & Reports
Markdown documents and analysis reports.

---

## 📜 Python Scripts Summary

### Data Preprocessing & Preparation

#### **preprocess_cpcb.py**
**Purpose:** Clean and standardize raw CPCB CSV data
- Loads CSV files from nested city/station folders
- Cleans column names (removes special characters)
- Fixes timestamp parsing (dayfirst=True for Indian date format)
- Converts all numeric columns
- Aligns data to strict 15-minute grid
- Interpolates missing values (max 3 hours)
- Adds time features (hour, day, weekday, month)
- Adds wind vector components from speed/direction
- **Output:** `clean_data_15min.parquet` (~13.4M rows)

#### **spatial_geocode.py**
**Purpose:** Geocode station names using OpenStreetMap API
- Extracts unique stations from preprocessed data
- Uses geopy Nominatim API to get coordinates
- Rate-limited to 1.2 seconds between requests (ToS compliance)
- Allows manual fixes via CSV editing
- Merges coordinates back with main dataset
- **Output:** `stations_geocoded.csv`, `clean_data_geo.parquet`

#### **fix_csv_coordinates.py**
**Purpose:** Fix missing or incorrect station coordinates
- Contains hardcoded manual overrides for 25+ problematic stations
- Uses 7 decimal place precision for consistency
- Patches missing coordinates automatically
- **Output:** Updated `stations_geocoded.csv`

#### **merge_imd_wind.py**
**Purpose:** Merge Indian Meteorological Department (IMD) wind data
- Loads CPCB pollution data and IMD wind data
- Builds BallTree spatial index for nearest neighbor matching
- Maps each CPCB station to nearest IMD weather station
- Interpolates missing wind measurements (max 3 hours)
- Converts meteorological wind convention to Cartesian vectors
  - Formula: θ = 270° - wind_direction
  - u = speed × cos(θ), v = speed × sin(θ)
- **Output:** `clean_data_with_wind.parquet`

#### **merge_final.py**
**Purpose:** Final merge operation combining all data
- Merges geospatial coordinates with 15-min data
- Produces final spatial dataset
- **Output:** `clean_data_geo.parquet`

---

### Spatial & Physics Features

#### **build_spatial_graph.py**
**Purpose:** Create network graph of stations within 50km radius
- Reads unique station locations
- Calculates haversine distances between all pairs (O(N²))
- Computes bearing angles for wind advection
- Filters to only keep edges ≤ 50km (borderless approach)
- Creates directed edges for advection flow
- **Output:** `spatial_graph_edges.parquet` (~8,000+ edges)

#### **build_advection_features.py**
**Purpose:** Implement multi-pollutant advection physics layer
- Pollutant-specific dispersion coefficients:
  - PM2.5: 10km (moderate travel)
  - PM10: 5km (fast settling)
  - NO2: 8km (reactive)
  - SO2: 8km (reactive)
  - CO: 20km (stable)
- Computes vector projection (wind dot product with bearing)
- Filters upwind flow (ReLU on directed wind speed)
- Gaussian decay function based on distance and pollutant physics
- Aggregates incoming mass flux to target stations
- **Output:** `advection_features.parquet` (new columns: `adv_in_PM2.5`, etc.)

---

### ML Dataset Preparation

#### **prep_tft_dataset.py**
**Purpose:** Prepare Temporal Fusion Transformer dataset
- Enforces strict temporal sorting by station and time
- Creates continuous time index (time_idx) mapped to integers
- Enforces PyTorch-compatible data types:
  - Categoricals: strings
  - Continuous: float32 (RAM optimization)
- Validates tensor compatibility
- **Output:** `tft_ready_data.parquet`

#### **build_tft_dataset.py**
**Purpose:** Sanitize and validate TFT dataset
- Applies safe imputation strategy:
  - Forward fill with limit=3 (45 min max)
  - Preserves NaN for truly offline sensors
- Cleans column names (handles µ, ³, special chars)
- Fills advection columns (0 means no transport)
- **Output:** `tft_ready_data_sanitized.parquet`

#### **prep_hourly_data.py**
**Purpose:** Transform 15-min data to hourly format for TiDE
- Resamples to 1-hour grid via groupby + resample
- Enforces perfect time continuity (no internal gaps)
- Detects and destroys hardware flatlines (>24 hours same value)
- Implements "chunk_id" system for continuous sequences
  - Breaks on NaN, ensures continuity for autoregressive training
- Filters chunks <192 hours (96-in + 96-out requirement)
- Adds temporal covariates (hour, day, weekday, month)
- **Output:** `tide_ready_1hr.parquet`

#### **fetch_inference_wind.py**
**Purpose:** Fetch future wind data for model inference
- Queries Open-Meteo historical/forecast API
- Date range: 2025-12-01 to 2025-12-07 (evaluation period)
- Computes u,v wind components
- Rate-limited with 0.5s delays
- **Output:** `inference_wind_2026.parquet`

---

### Model Training

#### **train_tide.py**
**Purpose:** Train foundation TiDE model (Time-series Dense Encoder)
- **Architecture:**
  - Input: 96-hour lookback
  - Output: 96-hour forecast
  - Hidden size: 512 (wide MLP)
  - Loss: Multi-Quantile (P10, P50, P90)
- **Data Flow:**
  - Melts 5 pollutants into separate target columns
  - Strict temporal split (train up to day 335, test last 30 days)
  - Guillotine filter (removes chunks <192 hours)
  - Filters fragmented sequences
- **Training:**
  - Optimizer: Default (usually Adam)
  - Learning rate: 1e-3
  - Max steps: 3000
  - Batch size: 512
  - Early stopping: 5-step patience
  - Hardware: H100 GPU (bf16-mixed precision)
- **Output:** `models/tide_global/`, `test_holdout_set.parquet`

#### **train_tft.py**
**Purpose:** Train Temporal Fusion Transformer model
- **Architecture:**
  - Multi-target: 5 pollutants
  - Hidden size: 64
  - Attention heads: 4
  - Look-back: 96 hours
  - Look-forward: 96 hours
- **Training:**
  - Learning rate: 0.03
  - Batch size: 1024
  - Max epochs: 50
  - Precision: bf16-mixed
  - Early stopping: 5-epoch patience
  - Persistent workers: 8
- **Output:** `models/tft/best-tft-model.ckpt`

#### **tide_grid_search.py**
**Purpose:** Hyperparameter optimization for TiDE
- **Search Space:**
  - hidden_size: [512, 768]
  - temporal_width: [4, 8]
  - learning_rate: [1e-3, 5e-4]
  - Total: 8 combinations
- **Validation:** 30-day grid validation window
- **Metric:** PM2.5 MAE on validation set
- **Winner:** Saves best model to `tide_optimized_best/`
- **Output:** Optimized model checkpoint

#### **test_run_tide_cpu.py**
**Purpose:** CPU smoke test for TiDE pipeline
- Minimal training (10 steps only)
- Validates all mathematical operations
- Checks data format compatibility
- Tests loss computation
- **Output:** Verification that pipeline is sound

---

### Model Evaluation & Analysis

#### **evaluate_tide.py**
**Purpose:** Evaluate foundation TiDE model on test vault
- Loads trained TiDE from `models/tide_global/`
- Generates 96-hour zero-shot forecast
- Aligns with 30-day test holdout set
- Computes metrics per pollutant:
  - MAE (mean absolute error)
  - RMSE (root mean squared error)
  - Mean baseline vs prediction
- Dynamic column selection for quantile outputs
- **Output:** `tide_96hr_results.parquet`

#### **evaluate_optimized_tide.py**
**Purpose:** Evaluate grid-search optimized TiDE model
- Same evaluation process as evaluate_tide.py
- Uses `models/tide_optimized_best/` instead
- **Output:** `tide_optimized_96hr_results.parquet`

#### **plot_forecast.py**
**Purpose:** Create interactive forecast visualization
- Loads TiDE evaluation results
- Plots ground truth vs prediction
- Interactive Plotly charts
- Handles station name parsing
- **Output:** HTML interactive graphs in `evaluation/plots/`

#### **offline_analytics.py**
**Purpose:** Causal analysis and influence graph
- **Feature 1:** Granger causality test
  - Determines if wind (advection) or local emissions drive pollution
  - Uses lagged correlation
- **Feature 2:** Influence graph
  - Cross-correlation with time lags
  - Identifies station-to-station pollution propagation
  - Weights edges by correlation strength
- **Output:** Mock demonstration (uses synthetic data in example)

---

### Physics Simulation & Digital Twin

#### **generate_96hr_digital_twin.py**
**Purpose:** Create spatial-temporal visualization grid from predictions
- Loads TiDE predictions and wind data
- For each of 96 hours:
  - Filters by timestamp
  - Performs cubic Kriging interpolation (fallback to nearest)
  - Creates smooth concentration maps
  - Generates u,v wind grids
- Renders HTML snapshots (first, middle, last hour)
- Packages as JSON for frontend visualization
- **Output:** `Delhi_96hr_digital_twin.json`

#### **physics_engine.py**
**Purpose:** Fast vectorized CFD engine for advection-diffusion
- **Config:**
  - DT_SUB: 60 seconds (sub-hourly timesteps)
  - DIFFUSION_COEFF: 12.0 (tuned for sharp plumes)
  - BACKGROUND_PM25: 80.0 (boundary inflow)
- **Operations:**
  - Vectorized upwind advection
  - Central difference diffusion
  - Boundary condition handling
  - Scenario application (highway emissions, wind multipliers)
- **Output:** Runs 60 sub-steps per hour, updates concentration field

#### **simulate_transport.py**
**Purpose:** Apply physics-based scenario modeling
- Reads digital twin JSON
- Modifies scenarios:
  - Highway emission reduction: 0-100%
  - Wind multiplier: Scale wind speed
  - Zone-based masking
- Runs vectorized CFD for 96 hours
- **Output:** Modified JSON with scenario effects

#### **source_tracer.py**
**Purpose:** Adjoint tracer algorithm for source attribution
- Injects virtual tracer mass at target location
- Reverses wind vectors and runs backward
- Traces pollution plume origin
- Confidence decreases with time due to diffusion
- Limited to 6 hours backward (diffusion loss)
- **Output:** Estimated source coordinates

---

### Data Verification & Validation

#### **verify_preprocess.py**
**Tests on 15-min data:**
- ✅ Shape, city/station counts
- ✅ 15-minute grid continuity
- ✅ Missing value ratios
- ✅ Physical constraints (bounds, no negatives)
- ✅ Duplicate timestamps
- ✅ Station coverage map

#### **verify_imd_merge.py**
**Tests on merged wind data:**
- ✅ Wind completeness
- ✅ Value ranges (u,v bounds)
- ✅ Zero wind ratio
- ✅ Temporal continuity
- ✅ Wind distribution

#### **verify_spatial_graph.py**
**Tests on spatial edges:**
- ✅ Distance constraints (≤50km)
- ✅ Bearing constraints (0-360°)
- ✅ Self-loop detection
- ✅ Cross-city connections (borderless)
- ✅ Graph connectivity

#### **verify_advection.py**
**Tests on advection features:**
- ✅ Dataset integrity (13.4M rows)
- ✅ Schema validation (all adv_in_* columns)
- ✅ Null value check (all filled)
- ✅ Physical constraints (no negative flow)
- ✅ Signal activation (non-zero advection)

#### **verify_final_dataset.py**
**Tests on geo-enhanced data:**
- ✅ Shape and schema
- ✅ Spatial completeness (100% lat/lon)
- ✅ Geographic bounds (India bounding box)
- ✅ Duplicate records
- ✅ Station coverage

#### **verify_tft_dataset.py**
**Tests on TFT tensor:**
- ✅ Shape and schema
- ✅ PyTorch data types (string categoricals, float32 reals)
- ✅ Time index sorting
- ✅ Null values in critical columns
- ✅ Memory footprint

#### **verify_tft_tensors.py**
**Tests on compiled PyTorch tensors:**
- ✅ TimeSeriesDataSet reconstruction
- ✅ Multi-target configuration (5 pollutants)
- ✅ Validation split generation
- ✅ DataLoader batch compilation
- ✅ NaN detection in tensors
- ✅ Encoder/decoder dimensions

#### **verify_hourly_dataset.py**
**Tests on 1-hour TiDE data:**
- ✅ Required columns
- ✅ Duplicate rows
- ✅ Time continuity
- ✅ Zero ratios per pollutant
- ✅ Negative values check
- ✅ Consecutive flatlines
- ✅ Wind flag consistency

#### **verify_tide_readiness.py**
**Tests on TiDE-ready data (comprehensive):**
- ✅ Absolute NaN check
- ✅ 100% temporal continuity (1H gaps)
- ✅ Minimum sequence length (192H)
- ✅ Static covariate integrity (static lat/lon)
- ✅ Global flatline detection (>24H artifacts)
- ✅ Negative physics check
- ✅ Wind observed flag activation

---

### Baseline & Utility

#### **xgboost_baseline.py**
**Purpose:** XGBoost baseline for comparison
- Uses lagged features: [1, 2, 3, 24, 48, 96] hours
- Auto-generates time features (hour, day, dayofweek, month)
- GPU acceleration (cuda tree_method)
- **Output:** Baseline accuracy metrics for benchmarking

#### **inspect_parquet_schema.py**
**Purpose:** Utility to inspect parquet file contents
- Walks through data/processed directory
- Prints schema, dtypes, sample rows
- Missing value percentages
- Numeric summaries
- Important column checks

---

## 🔄 Pipeline Workflow

```
Raw CPCB CSVs
     ↓
preprocess_cpcb.py → clean_data_15min.parquet
     ↓
spatial_geocode.py → stations_geocoded.csv
     ↓
merge_final.py → clean_data_geo.parquet
     ↓
merge_imd_wind.py → clean_data_with_wind.parquet
     ↓
build_spatial_graph.py → spatial_graph_edges.parquet
     ↓
build_advection_features.py → advection_features.parquet
     ↓
     ├→ prep_tft_dataset.py → tft_ready_data.parquet
     │   ├→ build_tft_dataset.py → tft_ready_data_sanitized.parquet
     │   └→ train_tft.py → models/tft/
     │
     └→ prep_hourly_data.py → tide_ready_1hr.parquet
         ├→ train_tide.py → models/tide_global/
         ├→ tide_grid_search.py → models/tide_optimized_best/
         └→ test_run_tide_cpu.py (validation)

Evaluation Phase:
     ↓
evaluate_tide.py → tide_96hr_results.parquet
evaluate_optimized_tide.py → tide_optimized_96hr_results.parquet
     ↓
generate_96hr_digital_twin.py → Delhi_96hr_digital_twin.json
     ↓
physics_engine.py → simulate_transport.py
     ↓
plot_forecast.py → HTML visualizations
```

---

## 🔧 Key Technologies & Libraries

- **Deep Learning:** PyTorch Lightning, NeuralForecast (TiDE), PyTorch Forecasting (TFT)
- **Data Processing:** Pandas, NumPy, Polars
- **Geospatial:** GeoPy, Shapely, PyProj, GeoDataFrames
- **Physics Simulation:** NumPy vectorized operations
- **Visualization:** Plotly, Folium, Matplotlib
- **Baseline ML:** XGBoost
- **Utilities:** Pandas, Parquet (PyArrow), JSON

---

## 📊 Data Scales

- **Total Records:** ~13.4 million rows
- **Time Period:** ~1.5 years
- **Stations:** 132 stations across 7 major Indian cities
- **Pollutants:** 5 (PM2.5, PM10, NO2, SO2, CO)
- **Spatial Graph Edges:** ~8,000+ (50km radius network)
- **Forecast Horizon:** 96 hours ahead

---

## 🎯 Model Performance Metrics

- **Evaluation Metrics:** MAE, RMSE per pollutant
- **Test Set:** 30-day holdout (unseen during training)
- **Ensemble Approach:** Foundation model + optimized variant
- **Physics Integration:** Advection, diffusion, boundary conditions

