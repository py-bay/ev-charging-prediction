# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Solar Tracker Forecast** compares baseline vs tracker-specific solar PV forecasting models. The project tests the hypothesis that tracker-level forecasting with multiple features (GHI, DNI, DHI, weather) achieves higher accuracy than simple baseline models using only GHI.

**Key Innovation**: 6-hour forecast horizon using historical measured weather data (no forecast API needed) by shifting features back 6 hours while keeping targets at current timestamp.

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Verify installation
uv run python -m src.solar_tracker_forecast --help
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_preprocessing.py -v
```

### Training Models

```bash
# Train Random Forest models (baseline + tracker-specific)
uv run python -m src.solar_tracker_forecast train-all

# Train LSTM models
uv run python -m src.solar_tracker_forecast train-lstm

# Train only baseline models (Linear Regression + Random Forest)
uv run python -m src.solar_tracker_forecast train-baseline

# Train only tracker-specific Random Forest models
uv run python -m src.solar_tracker_forecast train-tracker --corr-threshold 0.05 --top-n 15

# Preprocess data only (without training)
uv run python -m src.solar_tracker_forecast preprocess
```

### Code Quality
```bash
# Format code
uv run black src/

# Lint code
uv run ruff check src/
```

## Architecture

### Two-Pipeline Design

The codebase has **two parallel preprocessing and modeling pipelines**:

1. **Legacy Pipeline** (`src/preprocessing/pipeline.py`, `src/models/random_forest.py`, `src/models/lstm.py`):
   - Originally for EV charging window prediction (classification)
   - Uses `DataPreprocessor` class
   - Generates binary labels for "optimal charging windows"
   - CLI: `src/ev_charging_prediction.py`

2. **Tracker Forecasting Pipeline** (`src/preprocessing/tracker_preprocessing.py`, `src/models/baseline.py`, `src/models/tracker_models.py`, `src/models/lstm_regression.py`):
   - Current focus: Solar PV power regression forecasting
   - Uses `TrackerDataPreprocessor` class
   - Implements 6-hour forecast shift logic
   - Correlation-based feature selection
   - CLI: `src/solar_tracker_forecast.py` ← **Use this**

### Data Flow

```
Raw CSV Data (German column names)
    ↓
TrackerDataPreprocessor.load_pv_data()
    - Maps: "Solarproduktion Tracker 1" → "tracker1_north"
    - Maps: "Solarproduktion Tracker 2" → "tracker2_south"
    - Ignores: "Tracker 3" (always 0)
    ↓
TrackerDataPreprocessor.load_irradiance_data()
    - Extracts: GHI (ghi_cloudy_sky), DNI (dni_cloudy_sky), DHI (dhi_cloudy_sky)
    ↓
TrackerDataPreprocessor.load_weather_data()
    - Hourly data from Open-Meteo
    ↓
TrackerDataPreprocessor.resample_weather()
    - Resamples hourly → 15-minute intervals
    - Continuous vars: time interpolation
    - Categorical vars (weather_code): forward fill
    ↓
TrackerDataPreprocessor.merge_all_data()
    - Aligns on common timestamps
    ↓
TrackerDataPreprocessor.apply_6hour_forecast_shift()
    - Shifts features back 24 steps (6 hours * 4 steps/hour)
    - Keeps targets at current timestamp
    - Creates 6-hour ahead forecast scenario
    ↓
Feature Selection (tracker-specific)
    - Compute correlation with target
    - Select features with |corr| > threshold OR top N by absolute correlation
    - Separate feature sets for Tracker 1 (north) and Tracker 2 (south)
    ↓
Model Training
    - Baseline: GHI only → Total PV
    - Advanced: [GHI, DNI, DHI, Weather] → Tracker 1 + Tracker 2 (summed)
```

### Critical Implementation Details

#### Tracker Orientation
- **Tracker 1**: North-facing (`tracker1_north`)
- **Tracker 2**: South-facing (`tracker2_south`)
- **Tracker 3**: Ignored (always 0)

This mapping is hardcoded in `TrackerDataPreprocessor.load_pv_data()`.

#### German Column Names
PV data uses German names that must be mapped:
- `"Solarproduktion"` → `"pv_total"`
- `"Solarproduktion Tracker 1"` → `"tracker1_north"`
- `"Solarproduktion Tracker 2"` → `"tracker2_south"`

Other columns may include: `"Hausverbrauch"`, `"Ladezustand"`, `"Batterie (Laden)"`, etc.

#### Timestamp Handling
All timestamps are converted to **timezone-aware UTC**:
```python
pd.to_datetime(df[timestamp_col], utc=True)
```

The `_detect_timestamp_column()` method handles various column names: `timestamp`, `time`, `datetime`, `dt_iso`, etc.

#### Feature Selection Fallback
`TrackerDataPreprocessor.select_features_by_correlation()` has critical fallback logic:
- If no features meet correlation threshold → automatically selects top N by absolute correlation
- Prevents "0 features" error when threshold is too strict
- Default threshold: 0.05 (not 0.3, which is too high for some trackers)

#### LSTM Sequence Alignment
LSTM models use lookback windows (default: 12 steps = 3 hours):
- Training creates sequences from lookback steps
- Predictions lose first `lookback` samples
- **Always align y_test**: `y_test_aligned = y_test[lookback:]`
- Example in `train_lstm` command: see lines 599, 634, 663

### Model Architecture

#### Baseline Models
1. **LinearRegressionBaseline** (`src/models/baseline.py`):
   - Single feature: GHI
   - Target: Total PV power
   - Simple sklearn LinearRegression

2. **RandomForestBaseline** (`src/models/baseline.py`):
   - Single feature: GHI
   - Target: Total PV power
   - Uses config from `models.randomforest` in config.yaml

3. **LSTMBaselineModel** (`src/models/lstm_regression.py`):
   - Sequence of GHI values (lookback window)
   - Target: Total PV power
   - PyTorch LSTM with normalization

#### Tracker-Specific Models
1. **TrackerRandomForest** (`src/models/tracker_models.py`):
   - Multiple features: [GHI, DNI, DHI, selected weather features]
   - Separate model per tracker
   - Saves feature names and tracker metadata

2. **LSTMTrackerModel** (`src/models/lstm_regression.py`):
   - Sequences of multi-feature data
   - Separate model per tracker
   - PyTorch LSTM with per-feature normalization

#### Combined Advanced Forecast
Both tracker predictions are summed:
```python
y_pred_total_advanced = y_pred_tracker1 + y_pred_tracker2
```

### Evaluation System

#### Regression Metrics (`src/evaluation/regression_metrics.py`)
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error (skips zero values)
- **nRMSE**: Normalized RMSE (percentage of mean)
- **MBE**: Mean Bias Error (systematic over/under-prediction)

#### Visualizations (`src/evaluation/regression_visualizer.py`)
**Critical**: Uses non-interactive matplotlib backend (`matplotlib.use('Agg')`) to prevent Windows threading errors.

Generates 4 plots per model:
1. Predictions vs Actual scatter plot
2. Residual plot + distribution
3. Time series comparison (first 500 samples)
4. Error distribution by power magnitude

### Configuration System

Uses Pydantic models (`src/config/models.py`) for type-safe config validation:
- `DataPaths`: File locations
- `PreprocessingConfig`: Resampling, train/test split
- `FeaturesConfig`: Feature sets for different model types
- `ModelsConfig`: Hyperparameters for RandomForest and LSTM
- `OutputPaths`: Where to save models, results, plots

Load with: `config = load_config(Path("config/config.yaml"))`

## Data Requirements

### CSV Format Expectations

1. **pv_data.csv**:
   - Semicolon-separated (`;`)
   - 15-minute intervals
   - German column names
   - Must have: timestamp, "Solarproduktion", "Solarproduktion Tracker 1", "Solarproduktion Tracker 2"

2. **solar_irradiance.csv**:
   - Comma-separated
   - 15-minute intervals
   - Must have: dt_iso (timestamp), ghi_cloudy_sky, dni_cloudy_sky, dhi_cloudy_sky
   - May have UTC timezone suffix (stripped during loading)

3. **weather.csv**:
   - Comma-separated
   - Hourly data (resampled to 15-min automatically)
   - Skip first 2 header rows: `pd.read_csv(path, skiprows=2)`
   - Must have: time, temperature_2m, relative_humidity_2m, cloud_cover, wind_speed_10m, etc.

### Date Range Alignment
Preprocessing finds **common date range** across all three datasets. Models only train on overlapping timestamps.

## Common Issues and Solutions

### Issue: "['tracker1_south'] not in index"
**Cause**: Wrong tracker column names in data
**Solution**: Check actual column names. Current mapping expects:
- `"Solarproduktion Tracker 1"` (not `"Tracker 1"`)
- `"Solarproduktion Tracker 2"` (not `"Tracker 2"`)

### Issue: "Found array with 0 feature(s)"
**Cause**: Correlation threshold too high, no features selected
**Solution**: Lower `--corr-threshold` (default now 0.05) or ensure fallback logic in `select_features_by_correlation()` is working

### Issue: "RuntimeError: main thread is not in main loop" (matplotlib)
**Cause**: Interactive matplotlib backend on Windows
**Solution**: Already fixed - `matplotlib.use('Agg')` at top of `regression_visualizer.py`

### Issue: LSTM prediction length mismatch
**Cause**: Forgot to align y_test with lookback offset
**Solution**: Always use `y_test_aligned = y_test[lookback:]` when evaluating LSTM predictions

### Issue: "Cannot compare tz-naive and tz-aware timestamps"
**Cause**: Inconsistent timezone handling
**Solution**: All timestamps converted to UTC-aware via `pd.to_datetime(df[col], utc=True)`

## Output Structure

```
outputs/
├── models/
│   ├── baseline_linear_regression.pkl
│   ├── baseline_random_forest.pkl
│   ├── baseline_lstm.pt
│   ├── tracker1_north_rf.pkl
│   ├── tracker2_south_rf.pkl
│   ├── tracker1_north_lstm.pt
│   └── tracker2_south_lstm.pt
├── results/
│   ├── baseline_*_metrics.json
│   ├── tracker*_metrics.json
│   └── advanced_combined_*_metrics.json
└── plots/
    ├── baseline_*_pred_vs_actual.png
    ├── baseline_*_residuals.png
    ├── baseline_*_time_series.png
    ├── baseline_*_error_by_magnitude.png
    └── [same for tracker-specific and combined models]
```

## Key Files to Understand

### Primary CLI
- `src/solar_tracker_forecast.py`: Main CLI with commands: `train-all`, `train-lstm`, `train-baseline`, `train-tracker`, `preprocess`

### Preprocessing
- `src/preprocessing/tracker_preprocessing.py`: Complete pipeline including 6-hour shift logic and feature selection

### Models
- `src/models/baseline.py`: Linear Regression and Random Forest baselines
- `src/models/tracker_models.py`: Random Forest for individual trackers
- `src/models/lstm_regression.py`: LSTM models for baseline and trackers

### Evaluation
- `src/evaluation/regression_metrics.py`: Metrics calculation (RMSE, MAE, R², MAPE)
- `src/evaluation/regression_visualizer.py`: Plot generation

### Config
- `config/config.yaml`: Hyperparameters, paths, feature sets
- `src/config/models.py`: Pydantic validation models

## Logging

Uses Loguru for structured logging:
- Console output: Colored, formatted with timestamps
- File output: `logs/solar_tracker_forecast_YYYYMMDD_HHMMSS.log`
- Setup: `from src.logging import setup_logging; setup_logging(log_level="INFO")`

## Train/Test Split Philosophy

**Random split (not chronological)** with seed=42 for:
- Fair comparison across all models
- Consistent evaluation
- Focus on pattern learning, not temporal trends

All models use identical splits prepared by `TrackerDataPreprocessor`.
