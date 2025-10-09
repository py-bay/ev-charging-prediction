# PV Tracker Prediction

Deep learning-based solar PV production forecasting using LSTM models with separate tracker-level predictions.

## Overview

This project implements a complete machine learning pipeline for solar PV power prediction using LSTM neural networks. The approach trains separate models for each solar tracker to capture orientation-specific characteristics.

### Model Architecture

**LSTM-based Time Series Forecasting**
- **Architecture**: Multi-layer LSTM (2 layers, 64 hidden units) with dropout regularization (0.2)
- **Input**: Sequential data with 12-timestep lookback window (3 hours at 15-minute intervals)
- **Features**: Weather data, irradiance components (GHI, DNI, DHI), and temporal features (hour, day of year with sin/cos encoding)
- **Output**: Single-step forecast of PV production (Watts)
- **3 Separate Models**: Total production, North tracker, South tracker

### Key Features

- **Complete ML Pipeline**: Preprocessing → Processing → Training → Prediction → Evaluation
- **Independent Pipeline Steps**: Each step can be run separately or as part of complete workflow
- **Multiple Irradiance Components**: GHI (Global), DNI (Direct Normal), DHI (Diffuse Horizontal)
- **Weather Features**: Temperature, humidity, cloud cover, wind speed, pressure, etc.
- **Temporal Features**: Hour, day of year with cyclical encoding (sin/cos)
- **Comprehensive Metrics**: MAE, RMSE, R², MAPE, Max Error
- **Publication-Ready Visualizations**: 5 plot types per dataset in German (300 DPI PNG)

## Installation

Check out [INSTALLATION.md](./INSTALLATION.md) for installation instructions.

## Usage

### Complete Pipeline

Run the entire pipeline from data preprocessing to evaluation:

```bash
python main.py
```

### Individual Pipeline Steps

Each step can be run independently:

#### 1. Preprocessing
Load and merge raw data (PV, irradiance, weather):

```bash
python main.py --step preprocessing
```

**Input**: Raw CSV files in `data/`
**Output**: Preprocessed data in `data/preprocessed/`

#### 2. Processing
Add temporal features, split train/test, and scale:

```bash
python main.py --step processing
```

**Input**: Preprocessed data from step 1
**Output**: Train/test splits in `data/processed/`

#### 3. Training
Train LSTM models for all three datasets:

```bash
python main.py --step training
```

**Input**: Train/test data from step 2
**Output**: Trained models in `outputs/models/`

**Training Configuration** (in `config.yaml`):
- Epochs: 50 (with early stopping, patience=10)
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: MSE

#### 4. Prediction
Generate predictions on test data:

```bash
python main.py --step prediction
```

**Input**: Trained models + test data
**Output**: Predictions CSV in `outputs/predictions/`

#### 5. Evaluation
Calculate metrics and generate visualizations:

```bash
python main.py --step evaluation
```

**Input**: Predictions from step 4
**Output**: Metrics and plots in `outputs/results/`

### Running Multiple Steps

Run from a specific step onwards using `--continue`:

```bash
# Run prediction and evaluation
python main.py --step prediction --continue

# Run training, prediction, and evaluation
python main.py --step training --continue
```

## Data Requirements

Place CSV files in the `data/` directory:

### 1. pv_data.csv
PV production data (15-minute intervals):
- **Columns**: Timestamp, Solarproduktion (total), Solarproduktion Tracker 1 (North), Solarproduktion Tracker 2 (South)
- **Units**: Watts

### 2. irradiance.csv
Solar irradiance data (15-minute intervals):
- **Columns**: dt_iso, ghi_cloudy_sky, dni_cloudy_sky, dhi_cloudy_sky, ghi_clear_sky, dni_clear_sky, dhi_clear_sky
- **GHI**: Global Horizontal Irradiance (W/m²)
- **DNI**: Direct Normal Irradiance (W/m²)
- **DHI**: Diffuse Horizontal Irradiance (W/m²)
- **Cloudy/Clear Sky**: Different atmospheric conditions

### 3. weather.csv
Hourly weather data from Open-Meteo (automatically resampled to 15-min):
- **Columns**: time, temperature_2m, relative_humidity_2m, precipitation, cloud_cover, wind_speed_10m, wind_gusts_10m, pressure_msl, weather_code, cloud_cover_low, cloud_cover_mid, cloud_cover_high
- **Note**: Automatically resampled from hourly to 15-minute intervals

## Methodology

### Training Strategy

**Data Split**:
- 80% training, 20% testing
- Time-based split (no shuffle)
- Scaled using StandardScaler per feature

**Model Training**:
- Optimizer: Adam (learning rate: 0.001)
- Loss function: MSE (Mean Squared Error)
- Batch size: 32
- Max epochs: 50
- Early stopping: Patience of 10 epochs
- Device: Automatic GPU usage if available (CUDA)

**Sequence Creation**:
- Lookback window: 12 timesteps (3 hours)
- Target: Next timestep (single-step forecast)
- All features scaled before sequence creation

### Feature Engineering

**Temporal Features**:
- Hour of day
- Day of year
- Month
- Cyclical encoding: sin/cos transformations for hour and day of year

**Weather & Irradiance Features**:
- All irradiance components (GHI, DNI, DHI for cloudy/clear sky)
- All weather features (temperature, humidity, cloud cover, wind, pressure, etc.)
- No feature selection - model learns feature importance

## Evaluation

### Metrics

All models are evaluated using the following metrics:

- **MAE** (Mean Absolute Error): Average absolute prediction error in Watts
- **RMSE** (Root Mean Squared Error): Overall prediction error magnitude in Watts
- **R²** (Coefficient of Determination): Proportion of variance explained (0-1)
- **MAPE** (Mean Absolute Percentage Error): Percentage error relative to actual values
- **Max Error**: Largest absolute prediction error in Watts

### Visualizations

Five plot types are generated per dataset (total, north, south):

1. **Predicted vs Actual Scatter** (`{dataset}_predicted_vs_actual.png`)
   - Scatter plot with ideal line (y=x) in red
   - Linear regression line in blue with R² score
   - Equal aspect ratio

2. **Time Series Plot** (`{dataset}_timeseries.png`)
   - Actual vs predicted over random week (672 samples)
   - Shows temporal prediction patterns

3. **Residual Plot** (`{dataset}_residuals.png`)
   - Prediction errors over time
   - Zero line to identify systematic errors

4. **Error Distribution** (`{dataset}_error_distribution.png`)
   - Histogram of residuals
   - Tests for normality of errors

5. **Error vs Actual** (`{dataset}_error_vs_actual.png`)
   - Heteroscedasticity check
   - Shows if errors depend on production magnitude

**Plot Specifications**:
- Format: PNG, 300 DPI
- Language: German labels
- No captions (for publication use)
- Saved to `outputs/results/{dataset}/`

## Project Structure

```
pv-tracker-prediction/
├── src/
│   ├── models/
│   │   ├── lstm_model.py            # LSTM architecture
│   │   ├── trainer.py               # Training logic with early stopping
│   │   ├── predictor.py             # Prediction generation
│   │   └── dataset.py               # PyTorch dataset for sequences
│   ├── preprocessing/
│   │   ├── pipeline.py              # Data loading and merging
│   │   ├── data_loader.py           # Individual data loaders
│   │   └── processor.py             # Feature engineering and scaling
│   ├── evaluation/
│   │   └── evaluator.py             # Metrics and visualization
│   ├── config/
│   │   └── models.py                # Pydantic configuration models
│   └── logging/
│       └── logging_setup.py         # Structured logging with Loguru
├── main.py                          # Main pipeline orchestrator
├── config.yaml                      # Configuration file
├── data/
│   ├── pv_data.csv                  # PV production with trackers
│   ├── irradiance.csv               # GHI, DNI, DHI (cloudy/clear)
│   ├── weather.csv                  # Hourly weather data
│   ├── preprocessed/                # After step 1
│   │   ├── total_preprocessed.csv
│   │   ├── north_preprocessed.csv
│   │   └── south_preprocessed.csv
│   └── processed/                   # After step 2
│       ├── total/
│       │   ├── train.csv
│       │   └── test.csv
│       ├── north/
│       └── south/
├── outputs/
│   ├── models/                      # Trained PyTorch models (.pt)
│   ├── predictions/                 # Prediction CSV files
│   └── results/                     # Metrics and plots per dataset
├── logs/                            # Log files
├── pyproject.toml                   # UV project config
└── README.md                        # This file
```

## Configuration

All pipeline parameters are configured in `config.yaml`:

**Model Parameters**:
- `hidden_size`: 64
- `num_layers`: 2
- `dropout`: 0.2
- `lookback_window`: 12
- `batch_size`: 32
- `epochs`: 50
- `learning_rate`: 0.001
- `patience`: 10 (early stopping)

**Data Processing**:
- `train_test_split`: 0.8
- `add_temporal_features`: true
- Features: hour, day_of_year, month, hour_sin, hour_cos, day_of_year_sin, day_of_year_cos

**Paths**: Configurable input/output directories

## Logging

Logs are written to:
- **Console**: Formatted with colors and timestamps (Loguru)
- **Files**: `logs/log_YYYYMMDD_HHMMSS.log`

## Model Characteristics

**Strengths**:
- Captures temporal dependencies with LSTM architecture
- Separate models per tracker capture orientation-specific behavior
- Temporal features enable seasonality learning
- Multiple irradiance components (GHI, DNI, DHI) provide detailed input
- Weather context improves prediction accuracy

**Considerations**:
- Requires sufficient training data (time series)
- GPU recommended for faster training
- Hyperparameter tuning may improve results
- Early stopping prevents overfitting

## License

MIT License

## Contributing

Issues and pull requests are welcome on GitHub.
