# Installation Guide

## Prerequisites

- **Python**: 3.10 or higher
- **UV**: Package and project manager
- **Git**: For version control (optional)
- **Docker**: For containerized execution (optional)

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
git clone <repository-url>
cd ev-charging-prediction

# Or download and extract ZIP, then navigate to directory
```

## Step 3: Install Dependencies

```bash
# This will create a virtual environment and install all dependencies
uv sync
```

**Note**: First installation may take 5-10 minutes as it downloads:
- PyTorch (~230 MB)
- SciPy (~37 MB)
- scikit-learn (~8 MB)
- pandas (~11 MB)
- numpy (~12 MB)
- matplotlib (~8 MB)
- And other dependencies

## Step 4: Verify Installation

```bash
# Check CLI works
uv run python -m ev_charge_forecasting.cli --help

# Run tests
uv run pytest
```

Expected output:
```
Usage: python -m ev_charge_forecasting.cli [OPTIONS] COMMAND [ARGS]...

  EV Charge Forecasting - ML-based optimal charging window prediction

Commands:
  evaluate     Evaluate a trained model and generate metrics and visualizations.
  preprocess   Preprocess data: load, clean, resample, engineer features, and generate labels.
  train        Train a model on preprocessed data.
  train-all    Train and evaluate all models (RandomForest + LSTM).
```

## Step 5: Prepare Data

Create CSV files in the `data/` directory:

1. **pv_data.csv** - PV production data (15-minute intervals)
   ```csv
   timestamp,pv_power
   2024-01-01 00:00:00,0.0
   2024-01-01 00:15:00,0.0
   ...
   ```

2. **solar_irradiance.csv** - Solar irradiance (15-minute intervals)
   ```csv
   timestamp,irradiance
   2024-01-01 00:00:00,0.0
   2024-01-01 00:15:00,0.0
   ...
   ```

3. **weather.csv** - Hourly weather data from Open-Meteo
   ```csv
   timestamp,temperature_2m,relative_humidity_2m,dewpoint_2m,precipitation,cloud_cover,wind_speed_10m,wind_gusts_10m,pressure_msl,weather_code
   2024-01-01 00:00:00,15.2,65.0,8.5,0.0,45.0,3.2,5.1,1013.2,1
   ...
   ```

## Step 6: Run First Model

```bash
uv run python -m ev_charge_forecasting.cli train-all
```

This will:
1. Load and preprocess data
2. Train RandomForest model
3. Train LSTM model
4. Generate evaluation metrics
5. Create visualization plots

## Troubleshooting

### Issue: UV not found
**Solution**: Make sure UV is installed and in your PATH. Restart terminal after installation.

### Issue: Python version mismatch
**Solution**: UV will automatically use Python 3.10+. If not available, install Python 3.10 or higher first.

### Issue: Virtual environment activation fails
**Solution**: Use `uv run` prefix for all commands - UV manages the virtual environment automatically.

### Issue: Import errors
**Solution**: Always run commands with `uv run python -m ev_charge_forecasting.cli` from project root.

### Issue: Out of memory (LSTM training)
**Solution**:
1. Reduce batch size in `config/config.yaml`:
   ```yaml
   models:
     lstm:
       batch_size: 16  # Reduce from 32
   ```
2. Reduce sequence length:
   ```yaml
   models:
     lstm:
       sequence_length: 8  # Reduce from 12
   ```

### Issue: CUDA/GPU errors
**Solution**: PyTorch will automatically fall back to CPU if GPU is unavailable. This is normal and expected.

### Issue: Data file not found
**Solution**: Ensure CSV files are in `data/` directory with exact names:
- `pv_data.csv`
- `solar_irradiance.csv`
- `weather.csv`

## Docker Installation (Alternative)

If you prefer containerized execution:

### Build Image
```bash
docker-compose build
```

### Run Commands
```bash
# Train all models
docker-compose run ev-forecasting train-all

# Train specific model
docker-compose run ev-forecasting train --model randomforest --features weather
```

### Mount Custom Data
Edit `docker-compose.yml` to point to your data directory:
```yaml
volumes:
  - /path/to/your/data:/app/data:ro
```

## Development Setup

For development with hot-reload and testing:

```bash
# Install dev dependencies
uv sync --all-extras

# Install pre-commit hooks (optional)
uv run pre-commit install

# Run tests with coverage
uv run pytest --cov=ev_charge_forecasting --cov-report=html

# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/
```

## IDE Configuration

### VS Code
1. Install Python extension
2. Select interpreter: `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Unix)
3. Configure settings:
   ```json
   {
     "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
     "python.testing.pytestEnabled": true,
     "python.testing.unittestEnabled": false
   }
   ```

### PyCharm
1. Open project directory
2. File → Settings → Project → Python Interpreter
3. Add Interpreter → Existing environment
4. Select `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Unix)

## System Requirements

### Minimum
- CPU: 2 cores
- RAM: 4 GB
- Disk: 2 GB free space
- OS: Windows 10+, macOS 10.15+, Ubuntu 20.04+

### Recommended
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 5+ GB free space
- GPU: NVIDIA GPU with CUDA support (for LSTM acceleration)

## Uninstallation

```bash
# Remove virtual environment
rm -rf .venv  # Unix
rmdir /s .venv  # Windows

# Remove UV cache (optional)
uv cache clean
```

## Getting Help

- **Documentation**: See [README.md](README.md)
- **Quick Start**: See [QUICK_START.md](QUICK_START.md)
- **Issues**: Check error messages in `logs/` directory
- **Support**: Open an issue on GitHub repository

## Next Steps

1. ✅ Installation complete
2. 📁 Add your data files
3. 🚀 Run `train-all` command
4. 📊 Check results in `outputs/`
5. 🔬 Experiment with different configurations

Happy forecasting! 🚗⚡☀️
