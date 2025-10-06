# Quick Start Guide

## 1. Setup (5 minutes)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Verify installation
uv run python -m ev_charge_forecasting.cli --help
```

## 2. Prepare Data

Place your CSV files in the `data/` directory:
- `pv_data.csv` - PV power production (15-minute intervals)
- `solar_irradiance.csv` - Solar irradiance (15-minute intervals)
- `weather.csv` - Weather data (hourly from Open-Meteo)

## 3. Train Models (Quick)

```bash
# Train all models with default settings
uv run python -m ev_charge_forecasting.cli train-all

# This will:
# - Preprocess data
# - Train RandomForest and LSTM
# - Generate evaluation metrics
# - Create plots (ROC, confusion matrix)
```

## 4. View Results

Check the `outputs/` directory:
- `outputs/models/` - Trained model files
- `outputs/results/` - Evaluation metrics (JSON)
- `outputs/plots/` - ROC curves, confusion matrices

## 5. Experiment with Different Settings

### Try Different Feature Sets

```bash
# Weather data only
uv run python -m ev_charge_forecasting.cli train \
  --model randomforest \
  --features weather

# Irradiance only
uv run python -m ev_charge_forecasting.cli train \
  --model lstm \
  --features irradiance

# Combined (default)
uv run python -m ev_charge_forecasting.cli train \
  --model randomforest \
  --features combined
```

### Try Different Thresholds

```bash
# Low threshold (more optimal windows)
uv run python -m ev_charge_forecasting.cli train \
  --model randomforest \
  --threshold 0.5

# High threshold (fewer, better windows)
uv run python -m ev_charge_forecasting.cli train \
  --model randomforest \
  --threshold 1.5
```

### Use Relative Labeling

```bash
# Label optimal windows as top 20% of PV production
uv run python -m ev_charge_forecasting.cli train \
  --model randomforest \
  --threshold 0.8 \
  --method relative
```

## 6. Evaluate Saved Model

```bash
uv run python -m ev_charge_forecasting.cli evaluate \
  --model randomforest \
  --model-path outputs/models/randomforest_combined_t1.0.pkl \
  --features combined \
  --threshold 1.0
```

## 7. Run Tests

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_preprocessing.py -v

# With coverage
uv run pytest --cov=ev_charge_forecasting
```

## 8. Docker Usage (Optional)

```bash
# Build image
docker-compose build

# Train all models
docker-compose run ev-forecasting train-all

# Train specific model
docker-compose run ev-forecasting train --model randomforest --features weather
```

## Common Issues

### "FileNotFoundError: Config file not found"
**Solution**: Make sure you're running commands from the project root directory.

### "No module named 'ev_charge_forecasting'"
**Solution**: Use `uv run python -m ev_charge_forecasting.cli` instead of running Python directly.

### "PV data file not found"
**Solution**: Check that CSV files are in `data/` directory with correct names.

## Tips

1. **Start simple**: Use `train-all` command first to see baseline results
2. **Check logs**: Review `logs/` directory for detailed execution logs
3. **Iterate**: Try different feature sets and thresholds to optimize performance
4. **Use config file**: Modify `config/config.yaml` for persistent settings changes
5. **GPU acceleration**: LSTM will automatically use CUDA if available

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Experiment with hyperparameters in `config/config.yaml`
- Add custom features in `src/ev_charge_forecasting/preprocessing/pipeline.py`
- Explore evaluation plots in `outputs/plots/`
