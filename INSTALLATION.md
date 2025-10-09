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
git clone https://github.com/py-bay/pv-tracker-prediction.git
cd pv-tracker-prediction.git
```

## Step 3: Install Dependencies

```bash
# This will create a virtual environment and install all dependencies
uv sync
```
## Step 4: Prepare Data

Create CSV files in the `data/` directory:

1. **pv_data.csv** - PV production data (15-minute intervals)

2. **irradiance.csv** - Solar irradiance (15-minute intervals)

3. **weather.csv** - Hourly weather data from Open-Meteo
