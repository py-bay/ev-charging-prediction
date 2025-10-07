# Installation Guide

## Prerequisites

- **Python**: 3.10 or higher
- **uv**: Package and project manager
- **Git**: For version control (optional)

## Step 1: Install UV

### Windows (PowerShell)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Verify Installation
```bash
uv --version
```

## Step 2: Clone/Download Project

```bash
# If using Git
git clone https://github.com/py-bay/ev-charging-prediction.git
cd ev-charging-prediction
```

## Step 3: Install Dependencies

```bash
# This will create a virtual environment and install all dependencies
uv sync
```

## Step 4: Verify Installation

```bash
# Check CLI works
uv run python -m ev_charging_prediction --help

# Run tests
uv run pytest
```

Expected output:
```
 Usage: ev_charging_prediction.py [OPTIONS] COMMAND [ARGS]...                                              
                                                                                                           
 EV Charge Prediction - ML-based optimal charging window prediction

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────╮
│ preprocess   Preprocess data: load, clean, resample, engineer features, and generate labels.            │
│ train        Train a model on preprocessed data.                                                        │
│ evaluate     Evaluate a trained model and generate metrics and visualizations.                          │
│ train-all    Train and evaluate all models (RandomForest + LSTM).                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Step 5: Prepare Data

Create CSV files in the `data/` directory:

1. **pv_data.csv** - PV production data (15-minute intervals)

2. **solar_irradiance.csv** - Solar irradiance (15-minute intervals)

3. **weather.csv** - Hourly weather data from Open-Meteo

## Getting Help

- **Issues**: Check error messages in `logs/` directory
- **Support**: Open an issue on GitHub repository
