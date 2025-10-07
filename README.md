# Solar Tracker Forecast

Comparison of baseline vs tracker-specific solar PV forecasting models.

## Overview

This project compares two forecasting approaches for solar PV power prediction:

1. **Baseline Models**: Simple models using only Global Horizontal Irradiance (GHI) to predict total PV power
   - Linear Regression: GHI → Total PV
   - Random Forest: GHI → Total PV

2. **Advanced Tracker-Specific Models**: Separate Random Forest models for each tracker using multiple features
   - Tracker 1 (South-facing): [GHI, DNI, DHI, Weather features] → Tracker 1 PV
   - Tracker 2 (North-facing): [GHI, DNI, DHI, Weather features] → Tracker 2 PV
   - Total forecast: Sum of Tracker 1 + Tracker 2 predictions

**Hypothesis**: Tracker-level forecasting with multiple features should achieve higher accuracy than baseline models.

### Key Features

- **6-Hour Forecast Horizon**: Uses actual measured weather data (no forecast data required)
- **Correlation-Based Feature Selection**: Automatically selects most relevant features per tracker
- **Multiple Irradiance Components**: GHI (Global), DNI (Direct Normal), DHI (Diffuse Horizontal)
- **Weather Features**: Temperature, humidity, cloud cover, wind speed, pressure, etc.
- **Comprehensive Metrics**: RMSE, MAE, R², MAPE, normalized RMSE, bias error
- **Publication-Ready Visualizations**: Predictions vs actual, residuals, time series, error distributions

## Installation

Check out [INSTALLATION.md](./INSTALLATION.md) for installation instructions.

## Usage

### Quick Start - Train All Models

Train and evaluate both baseline and tracker-specific models:

```bash
uv run python -m src.solar_tracker_forecast train-all
```

### Individual Commands

#### 1. Preprocess Data

```bash
uv run python -m src.solar_tracker_forecast preprocess
```

This will:
- Load PV data (with tracker information)
- Load irradiance data (GHI, DNI, DHI)
- Load weather data and resample to 15-minute intervals
- Apply 6-hour forecast shift
- Merge all data sources

#### 2. Train Baseline Models

```bash
uv run python -m src.solar_tracker_forecast train-baseline
```

Trains both:
- Linear Regression (GHI → Total PV)
- Random Forest (GHI → Total PV)

Outputs:
- Model files: `outputs/models/baseline_*.pkl`
- Metrics: `outputs/results/baseline_*_metrics.json`
- Plots: `outputs/plots/baseline_*.png`

#### 3. Train Tracker-Specific Models

```bash
uv run python -m src.solar_tracker_forecast train-tracker \
  --corr-threshold 0.3 \
  --top-n 15
```

**Options:**
- `--corr-threshold, -t`: Minimum correlation for feature selection [default: 0.3]
- `--top-n, -n`: Maximum number of features to select [default: 15]

Trains:
- Tracker 1 (South) Random Forest model
- Tracker 2 (North) Random Forest model
- Evaluates combined forecast (T1 + T2)

Outputs:
- Model files: `outputs/models/tracker{1,2}_*.pkl`
- Metrics: `outputs/results/tracker*_metrics.json`
- Plots: `outputs/plots/tracker*.png`

## Data Requirements

Place CSV files in the `data/` directory:

### 1. pv_data.csv
PV production data (15-minute intervals, semicolon-separated, German column names):
- **Columns**: Timestamp, Solarproduktion (total), Tracker 1, Tracker 2, Tracker 3
- **Note**: Tracker 3 is ignored (always 0)

### 2. solar_irradiance.csv
Solar irradiance data (15-minute intervals):
- **Columns**: dt_iso, ghi_cloudy_sky, dni_cloudy_sky, dhi_cloudy_sky
- **GHI**: Global Horizontal Irradiance
- **DNI**: Direct Normal Irradiance
- **DHI**: Diffuse Horizontal Irradiance

### 3. weather.csv
Hourly weather data from Open-Meteo (automatically resampled to 15-min):
- **Columns**: time, temperature_2m, relative_humidity_2m, precipitation, cloud_cover, wind_speed_10m, wind_gusts_10m, pressure_msl, weather_code, cloud_cover_low, cloud_cover_mid, cloud_cover_high

## Forecasting Methodology

### 6-Hour Forecast Logic

The models predict PV power 6 hours ahead using historical weather and irradiance data:

1. Features (weather, irradiance) are shifted back 6 hours
2. Target (PV power) remains at current timestamp
3. This creates a 6-hour forecast scenario using actual measured data
4. No forecast data required - relies on measurement lag

### Feature Selection

For tracker-specific models:
1. Compute correlation matrix between all features and tracker power
2. Select features with |correlation| > threshold
3. Limit to top N features by absolute correlation
4. Train separate models per tracker with their selected features

### Train/Test Split

- 80% training, 20% testing
- Random shuffle (not chronological)
- Same random seed (42) for reproducibility
- All models use identical splits for fair comparison

## Evaluation Metrics

All models are evaluated using regression metrics:

- **RMSE** (Root Mean Squared Error): Overall prediction error magnitude
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **R²** (Coefficient of Determination): Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Percentage error relative to actual values
- **nRMSE**: Normalized RMSE (as percentage of mean actual value)
- **MBE** (Mean Bias Error): Systematic over/under-prediction
- **Max Error**: Largest prediction error

## Project Structure

```
solar-tracker-forecast/
├── src/
│   ├── models/
│   │   ├── baseline.py              # Linear Regression & RF baseline
│   │   ├── tracker_models.py        # Tracker-specific Random Forest
│   │   ├── random_forest.py         # Legacy RF (classification)
│   │   └── lstm.py                  # Legacy LSTM (classification)
│   ├── preprocessing/
│   │   ├── tracker_preprocessing.py # New tracker-specific preprocessing
│   │   └── pipeline.py              # Legacy preprocessing
│   ├── evaluation/
│   │   ├── regression_metrics.py    # RMSE, MAE, R², MAPE
│   │   ├── regression_visualizer.py # Regression plots
│   │   ├── metrics_calculator.py    # Legacy (classification)
│   │   └── visualizer.py            # Legacy (classification)
│   ├── config/
│   │   └── models.py                # Pydantic configuration models
│   ├── logging/
│   │   └── logging_setup.py         # Structured logging with Loguru
│   └── solar_tracker_forecast.py    # New CLI
├── config/
│   └── config.yaml                  # Configuration file
├── data/
│   ├── pv_data.csv                  # PV production with trackers
│   ├── solar_irradiance.csv         # GHI, DNI, DHI
│   └── weather.csv                  # Hourly weather data
├── outputs/
│   ├── models/                      # Saved model files
│   ├── results/                     # JSON metrics
│   └── plots/                       # Visualizations
├── notebooks/
│   ├── datenanalyse_uebersicht.ipynb   # Data analysis (German)
│   └── wetter_korrelationen.ipynb      # Weather correlations (German)
├── tests/                           # Unit tests
├── logs/                            # Log files
├── pyproject.toml                   # UV project config
└── README.md                        # This file
```

## Testing

Run unit tests with pytest:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_preprocessing.py -v
```

## Logging

Logs are written to:
- **Console**: Formatted with colors and timestamps
- **Files**: `logs/solar_tracker_forecast_YYYYMMDD_HHMMSS.log`

## Comparison: Baseline vs Advanced

### Baseline Approach
- **Pros**: Simple, fast, interpretable
- **Cons**: Single feature (GHI), no tracker specificity, no weather context
- **Use Case**: Quick estimate, low computational requirements

### Advanced Tracker-Specific Approach
- **Pros**: Multiple features, tracker-specific models, weather context, orientation-aware
- **Cons**: More complex, requires feature engineering, longer training time
- **Use Case**: Higher accuracy requirements, production forecasting

## Expected Results

Based on correlation analysis:
- GHI has strong correlation (r ≈ 0.9) with total PV
- DNI, DHI provide additional orientation-specific information
- Weather features (cloud cover, temperature) add context
- Tracker-specific models should capture orientation differences (south vs north)

**Hypothesis**: Advanced approach should achieve 10-20% lower RMSE compared to baseline.

## License

MIT License

## Contributing

Issues and pull requests are welcome on GitHub.
