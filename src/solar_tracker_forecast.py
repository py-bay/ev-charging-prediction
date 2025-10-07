"""Command-line interface for solar tracker forecasting."""

import json
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from src.config import load_config
from src.evaluation.regression_metrics import RegressionMetricsCalculator
from src.evaluation.regression_visualizer import RegressionVisualizer
from src.models.baseline import LinearRegressionBaseline, RandomForestBaseline
from src.models.tracker_models import TrackerRandomForest, AdvancedTrackerForecaster
from src.models.lstm_regression import LSTMBaselineModel, LSTMTrackerModel
from src.preprocessing.tracker_preprocessing import TrackerDataPreprocessor
from src.logging import setup_logging

app = typer.Typer(
    help="Solar Tracker Forecast - Compare baseline vs tracker-specific PV forecasting",
    add_completion=False,
)


@app.command()
def preprocess(
    config_path: str = typer.Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Preprocess data: load, merge, resample, apply 6-hour forecast shift.
    """
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("SOLAR TRACKER FORECAST - PREPROCESSING")
    logger.info("=" * 80)

    try:
        config = load_config(Path(config_path))
        preprocessor = TrackerDataPreprocessor(config)
        df_combined = preprocessor.run_full_pipeline()

        logger.info("Preprocessing completed successfully!")
        logger.info(f"Final data shape: {df_combined.shape}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def train_baseline(
    config_path: str = typer.Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Train baseline models: Linear Regression and Random Forest (GHI -> Total PV).
    """
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("TRAINING BASELINE MODELS")
    logger.info("=" * 80)

    try:
        config = load_config(Path(config_path))

        # Preprocess data
        preprocessor = TrackerDataPreprocessor(config)
        df_combined = preprocessor.run_full_pipeline()

        # Prepare baseline data
        X_train, X_test, y_train, y_test = preprocessor.prepare_baseline_data(df_combined)

        # Train Linear Regression
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING LINEAR REGRESSION BASELINE")
        logger.info("=" * 80)
        lr_model = LinearRegressionBaseline(config)
        lr_info = lr_model.train(X_train, y_train)
        logger.info(f"Training info: {lr_info}")

        # Save model
        lr_model.save(config.output_paths.models, "baseline_linear_regression")

        # Evaluate
        y_pred_lr = lr_model.predict(X_test)
        metrics_lr = RegressionMetricsCalculator.compute_all_metrics(y_test, y_pred_lr)
        logger.info(f"Linear Regression Test Metrics:")
        logger.info(f"  RMSE: {metrics_lr['rmse']:.4f}")
        logger.info(f"  MAE: {metrics_lr['mae']:.4f}")
        logger.info(f"  R²: {metrics_lr['r2']:.4f}")
        logger.info(f"  MAPE: {metrics_lr['mape']:.2f}%")

        # Save metrics
        metrics_path = config.output_paths.results / "baseline_linear_regression_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_lr, f, indent=2)

        # Visualize
        RegressionVisualizer.plot_all_regression_visualizations(
            y_test, y_pred_lr, config.output_paths.plots, "baseline_linear_regression"
        )

        # Train Random Forest
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING RANDOM FOREST BASELINE")
        logger.info("=" * 80)
        rf_model = RandomForestBaseline(config)
        rf_info = rf_model.train(X_train, y_train)
        logger.info(f"Training info: {rf_info}")

        # Save model
        rf_model.save(config.output_paths.models, "baseline_random_forest")

        # Evaluate
        y_pred_rf = rf_model.predict(X_test)
        metrics_rf = RegressionMetricsCalculator.compute_all_metrics(y_test, y_pred_rf)
        logger.info(f"Random Forest Test Metrics:")
        logger.info(f"  RMSE: {metrics_rf['rmse']:.4f}")
        logger.info(f"  MAE: {metrics_rf['mae']:.4f}")
        logger.info(f"  R²: {metrics_rf['r2']:.4f}")
        logger.info(f"  MAPE: {metrics_rf['mape']:.2f}%")

        # Save metrics
        metrics_path = config.output_paths.results / "baseline_random_forest_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_rf, f, indent=2)

        # Visualize
        RegressionVisualizer.plot_all_regression_visualizations(
            y_test, y_pred_rf, config.output_paths.plots, "baseline_random_forest"
        )

        logger.info("\n" + "=" * 80)
        logger.info("BASELINE TRAINING COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Baseline training failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def train_tracker(
    config_path: str = typer.Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    correlation_threshold: float = typer.Option(
        0.05,
        "--corr-threshold",
        "-t",
        help="Minimum correlation for feature selection",
    ),
    top_n_features: int = typer.Option(
        15,
        "--top-n",
        "-n",
        help="Maximum number of features to select",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Train tracker-specific models: Random Forest per tracker with multiple features.
    """
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("TRAINING TRACKER-SPECIFIC MODELS")
    logger.info("=" * 80)

    try:
        config = load_config(Path(config_path))

        # Preprocess data
        preprocessor = TrackerDataPreprocessor(config)
        df_combined = preprocessor.run_full_pipeline()

        # Select features for Tracker 1 (south)
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE SELECTION FOR TRACKER 1 (SOUTH)")
        logger.info("=" * 80)
        tracker1_features = preprocessor.select_features_by_correlation(
            df_combined, "tracker1_south", threshold=correlation_threshold, top_n=top_n_features
        )

        # Prepare data for Tracker 1
        X_train_t1, X_test_t1, y_train_t1, y_test_t1 = preprocessor.prepare_tracker_data(
            df_combined, "tracker1_south", tracker1_features
        )

        # Train Tracker 1 model
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING TRACKER 1 MODEL")
        logger.info("=" * 80)
        tracker1_model = TrackerRandomForest(config, "Tracker1_south", tracker1_features)
        tracker1_info = tracker1_model.train(X_train_t1, y_train_t1)

        # Save model
        tracker1_model.save(config.output_paths.models, "tracker1_south_rf")

        # Evaluate Tracker 1
        y_pred_t1 = tracker1_model.predict(X_test_t1)
        metrics_t1 = RegressionMetricsCalculator.compute_all_metrics(y_test_t1, y_pred_t1)
        logger.info(f"Tracker 1 Test Metrics:")
        logger.info(f"  RMSE: {metrics_t1['rmse']:.4f}")
        logger.info(f"  MAE: {metrics_t1['mae']:.4f}")
        logger.info(f"  R²: {metrics_t1['r2']:.4f}")
        logger.info(f"  MAPE: {metrics_t1['mape']:.2f}%")

        # Save metrics
        metrics_path = config.output_paths.results / "tracker1_south_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({**metrics_t1, **tracker1_info}, f, indent=2)

        # Visualize
        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_t1, y_pred_t1, config.output_paths.plots, "tracker1_south"
        )

        # Select features for Tracker 2 (north)
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE SELECTION FOR TRACKER 2 (NORTH)")
        logger.info("=" * 80)
        tracker2_features = preprocessor.select_features_by_correlation(
            df_combined, "tracker2_north", threshold=correlation_threshold, top_n=top_n_features
        )

        # Prepare data for Tracker 2
        X_train_t2, X_test_t2, y_train_t2, y_test_t2 = preprocessor.prepare_tracker_data(
            df_combined, "tracker2_north", tracker2_features
        )

        # Train Tracker 2 model
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING TRACKER 2 MODEL")
        logger.info("=" * 80)
        tracker2_model = TrackerRandomForest(config, "Tracker2_north", tracker2_features)
        tracker2_info = tracker2_model.train(X_train_t2, y_train_t2)

        # Save model
        tracker2_model.save(config.output_paths.models, "tracker2_north_rf")

        # Evaluate Tracker 2
        y_pred_t2 = tracker2_model.predict(X_test_t2)
        metrics_t2 = RegressionMetricsCalculator.compute_all_metrics(y_test_t2, y_pred_t2)
        logger.info(f"Tracker 2 Test Metrics:")
        logger.info(f"  RMSE: {metrics_t2['rmse']:.4f}")
        logger.info(f"  MAE: {metrics_t2['mae']:.4f}")
        logger.info(f"  R²: {metrics_t2['r2']:.4f}")
        logger.info(f"  MAPE: {metrics_t2['mape']:.2f}%")

        # Save metrics
        metrics_path = config.output_paths.results / "tracker2_north_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({**metrics_t2, **tracker2_info}, f, indent=2)

        # Visualize
        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_t2, y_pred_t2, config.output_paths.plots, "tracker2_north"
        )

        # Create combined forecaster and evaluate total
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING COMBINED TRACKER FORECAST")
        logger.info("=" * 80)

        # Need to get test data with both feature sets - use intersection of test indices
        # For simplicity, re-split with same random state and get y_test_total
        X_test_combined = df_combined[tracker1_features + tracker2_features].values
        y_test_total = df_combined["pv_total"].values

        # Use same random split
        from sklearn.model_selection import train_test_split
        _, X_test_idx, _, y_test_total_split = train_test_split(
            X_test_combined, y_test_total, test_size=0.2, random_state=42, shuffle=True
        )

        # Get predictions for both trackers on their respective features
        # This is simplified - in production you'd align indices properly
        y_pred_total_advanced = y_pred_t1 + y_pred_t2
        y_test_total_subset = y_test_t1 + y_test_t2  # Approximate total for evaluation

        metrics_advanced = RegressionMetricsCalculator.compute_all_metrics(
            y_test_total_subset, y_pred_total_advanced
        )
        logger.info(f"Advanced (Combined Trackers) Test Metrics:")
        logger.info(f"  RMSE: {metrics_advanced['rmse']:.4f}")
        logger.info(f"  MAE: {metrics_advanced['mae']:.4f}")
        logger.info(f"  R²: {metrics_advanced['r2']:.4f}")
        logger.info(f"  MAPE: {metrics_advanced['mape']:.2f}%")

        # Save combined metrics
        metrics_path = config.output_paths.results / "advanced_combined_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_advanced, f, indent=2)

        # Visualize combined
        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_total_subset, y_pred_total_advanced,
            config.output_paths.plots, "advanced_combined_trackers"
        )

        logger.info("\n" + "=" * 80)
        logger.info("TRACKER-SPECIFIC TRAINING COMPLETED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Tracker training failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def train_all(
    config_path: str = typer.Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    correlation_threshold: float = typer.Option(
        0.05,
        "--corr-threshold",
        "-t",
        help="Minimum correlation for feature selection",
    ),
    top_n_features: int = typer.Option(
        15,
        "--top-n",
        "-n",
        help="Maximum number of features to select",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Train all models: baseline + tracker-specific, and compare results.
    """
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("TRAINING ALL MODELS")
    logger.info("=" * 80)

    try:
        config = load_config(Path(config_path))

        # Preprocess data once
        logger.info("Running preprocessing pipeline...")
        preprocessor = TrackerDataPreprocessor(config)
        df_combined = preprocessor.run_full_pipeline()

        # ========== BASELINE MODELS ==========
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING BASELINE MODELS")
        logger.info("=" * 80)

        # Prepare baseline data
        X_train, X_test, y_train, y_test = preprocessor.prepare_baseline_data(df_combined)

        # Linear Regression
        logger.info("\nTraining Linear Regression Baseline...")
        lr_model = LinearRegressionBaseline(config)
        lr_info = lr_model.train(X_train, y_train)
        lr_model.save(config.output_paths.models, "baseline_linear_regression")

        y_pred_lr = lr_model.predict(X_test)
        metrics_lr = RegressionMetricsCalculator.compute_all_metrics(y_test, y_pred_lr)
        logger.info(f"Linear Regression: RMSE={metrics_lr['rmse']:.4f}, MAE={metrics_lr['mae']:.4f}, R²={metrics_lr['r2']:.4f}")

        metrics_path = config.output_paths.results / "baseline_linear_regression_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_lr, f, indent=2)

        RegressionVisualizer.plot_all_regression_visualizations(
            y_test, y_pred_lr, config.output_paths.plots, "baseline_linear_regression"
        )

        # Random Forest
        logger.info("\nTraining Random Forest Baseline...")
        rf_model = RandomForestBaseline(config)
        rf_info = rf_model.train(X_train, y_train)
        rf_model.save(config.output_paths.models, "baseline_random_forest")

        y_pred_rf = rf_model.predict(X_test)
        metrics_rf = RegressionMetricsCalculator.compute_all_metrics(y_test, y_pred_rf)
        logger.info(f"Random Forest: RMSE={metrics_rf['rmse']:.4f}, MAE={metrics_rf['mae']:.4f}, R²={metrics_rf['r2']:.4f}")

        metrics_path = config.output_paths.results / "baseline_random_forest_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_rf, f, indent=2)

        RegressionVisualizer.plot_all_regression_visualizations(
            y_test, y_pred_rf, config.output_paths.plots, "baseline_random_forest"
        )

        # ========== TRACKER-SPECIFIC MODELS ==========
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING TRACKER-SPECIFIC MODELS")
        logger.info("=" * 80)

        # Tracker 1 (North)
        logger.info("\nFeature selection for Tracker 1 (North)...")
        tracker1_features = preprocessor.select_features_by_correlation(
            df_combined, "tracker1_north", threshold=correlation_threshold, top_n=top_n_features
        )

        X_train_t1, X_test_t1, y_train_t1, y_test_t1 = preprocessor.prepare_tracker_data(
            df_combined, "tracker1_north", tracker1_features
        )

        logger.info("Training Tracker 1 model...")
        tracker1_model = TrackerRandomForest(config, "Tracker1_north", tracker1_features)
        tracker1_info = tracker1_model.train(X_train_t1, y_train_t1)
        tracker1_model.save(config.output_paths.models, "tracker1_north_rf")

        y_pred_t1 = tracker1_model.predict(X_test_t1)
        metrics_t1 = RegressionMetricsCalculator.compute_all_metrics(y_test_t1, y_pred_t1)
        logger.info(f"Tracker 1 (North): RMSE={metrics_t1['rmse']:.4f}, MAE={metrics_t1['mae']:.4f}, R²={metrics_t1['r2']:.4f}")

        metrics_path = config.output_paths.results / "tracker1_north_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({**metrics_t1, **tracker1_info}, f, indent=2)

        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_t1, y_pred_t1, config.output_paths.plots, "tracker1_north"
        )

        # Tracker 2 (South)
        logger.info("\nFeature selection for Tracker 2 (South)...")
        tracker2_features = preprocessor.select_features_by_correlation(
            df_combined, "tracker2_south", threshold=correlation_threshold, top_n=top_n_features
        )

        X_train_t2, X_test_t2, y_train_t2, y_test_t2 = preprocessor.prepare_tracker_data(
            df_combined, "tracker2_south", tracker2_features
        )

        logger.info("Training Tracker 2 model...")
        tracker2_model = TrackerRandomForest(config, "Tracker2_south", tracker2_features)
        tracker2_info = tracker2_model.train(X_train_t2, y_train_t2)
        tracker2_model.save(config.output_paths.models, "tracker2_south_rf")

        y_pred_t2 = tracker2_model.predict(X_test_t2)
        metrics_t2 = RegressionMetricsCalculator.compute_all_metrics(y_test_t2, y_pred_t2)
        logger.info(f"Tracker 2 (South): RMSE={metrics_t2['rmse']:.4f}, MAE={metrics_t2['mae']:.4f}, R²={metrics_t2['r2']:.4f}")

        metrics_path = config.output_paths.results / "tracker2_south_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({**metrics_t2, **tracker2_info}, f, indent=2)

        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_t2, y_pred_t2, config.output_paths.plots, "tracker2_south"
        )

        # Combined
        logger.info("\nEvaluating combined tracker forecast...")
        y_pred_total_advanced = y_pred_t1 + y_pred_t2
        y_test_total_subset = y_test_t1 + y_test_t2

        metrics_advanced = RegressionMetricsCalculator.compute_all_metrics(
            y_test_total_subset, y_pred_total_advanced
        )
        logger.info(f"Advanced (Combined): RMSE={metrics_advanced['rmse']:.4f}, MAE={metrics_advanced['mae']:.4f}, R²={metrics_advanced['r2']:.4f}")

        metrics_path = config.output_paths.results / "advanced_combined_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_advanced, f, indent=2)

        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_total_subset, y_pred_total_advanced,
            config.output_paths.plots, "advanced_combined_trackers"
        )

        # ========== SUMMARY ==========
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE - RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nBaseline Models (GHI only):")
        logger.info(f"  Linear Regression: RMSE={metrics_lr['rmse']:.4f}, R²={metrics_lr['r2']:.4f}")
        logger.info(f"  Random Forest:     RMSE={metrics_rf['rmse']:.4f}, R²={metrics_rf['r2']:.4f}")
        logger.info(f"\nAdvanced Model (Multi-feature trackers):")
        logger.info(f"  Combined Trackers: RMSE={metrics_advanced['rmse']:.4f}, R²={metrics_advanced['r2']:.4f}")

        improvement = ((metrics_rf['rmse'] - metrics_advanced['rmse']) / metrics_rf['rmse']) * 100
        logger.info(f"\nImprovement over RF baseline: {improvement:+.2f}% RMSE reduction")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def train_lstm(
    config_path: str = typer.Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    correlation_threshold: float = typer.Option(
        0.05,
        "--corr-threshold",
        "-t",
        help="Minimum correlation for feature selection",
    ),
    top_n_features: int = typer.Option(
        15,
        "--top-n",
        "-n",
        help="Maximum number of features to select",
    ),
    lookback: int = typer.Option(
        12,
        "--lookback",
        "-lb",
        help="Number of time steps to look back (default: 12 = 3 hours)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Train LSTM models: baseline + tracker-specific.
    """
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("TRAINING LSTM MODELS")
    logger.info("=" * 80)

    try:
        config = load_config(Path(config_path))

        # Preprocess data once
        logger.info("Running preprocessing pipeline...")
        preprocessor = TrackerDataPreprocessor(config)
        df_combined = preprocessor.run_full_pipeline()

        # ========== LSTM BASELINE ==========
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING LSTM BASELINE MODEL")
        logger.info("=" * 80)

        # Prepare baseline data
        X_train, X_test, y_train, y_test = preprocessor.prepare_baseline_data(df_combined)

        # Train LSTM Baseline
        logger.info("\nTraining LSTM Baseline...")
        lstm_baseline = LSTMBaselineModel(config, lookback=lookback)
        lstm_info = lstm_baseline.train(X_train, y_train)
        lstm_baseline.save(config.output_paths.models, "baseline_lstm")

        # Predict (note: LSTM returns predictions only for samples with enough history)
        y_pred_lstm = lstm_baseline.predict(X_test)
        # Align y_test with predictions (remove first lookback samples)
        y_test_aligned = y_test[lookback:]

        metrics_lstm = RegressionMetricsCalculator.compute_all_metrics(y_test_aligned, y_pred_lstm)
        logger.info(f"LSTM Baseline: RMSE={metrics_lstm['rmse']:.4f}, MAE={metrics_lstm['mae']:.4f}, R²={metrics_lstm['r2']:.4f}")

        metrics_path = config.output_paths.results / "baseline_lstm_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({**metrics_lstm, **lstm_info}, f, indent=2)

        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_aligned, y_pred_lstm, config.output_paths.plots, "baseline_lstm"
        )

        # ========== LSTM TRACKER-SPECIFIC MODELS ==========
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING LSTM TRACKER-SPECIFIC MODELS")
        logger.info("=" * 80)

        # Tracker 1 (North)
        logger.info("\nFeature selection for Tracker 1 (North)...")
        tracker1_features = preprocessor.select_features_by_correlation(
            df_combined, "tracker1_north", threshold=correlation_threshold, top_n=top_n_features
        )

        X_train_t1, X_test_t1, y_train_t1, y_test_t1 = preprocessor.prepare_tracker_data(
            df_combined, "tracker1_north", tracker1_features
        )

        logger.info("Training LSTM Tracker 1 model...")
        lstm_tracker1 = LSTMTrackerModel(config, "Tracker1_north", tracker1_features, lookback=lookback)
        tracker1_info = lstm_tracker1.train(X_train_t1, y_train_t1)
        lstm_tracker1.save(config.output_paths.models, "tracker1_north_lstm")

        y_pred_t1 = lstm_tracker1.predict(X_test_t1)
        y_test_t1_aligned = y_test_t1[lookback:]

        metrics_t1 = RegressionMetricsCalculator.compute_all_metrics(y_test_t1_aligned, y_pred_t1)
        logger.info(f"LSTM Tracker 1 (North): RMSE={metrics_t1['rmse']:.4f}, MAE={metrics_t1['mae']:.4f}, R²={metrics_t1['r2']:.4f}")

        metrics_path = config.output_paths.results / "tracker1_north_lstm_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({**metrics_t1, **tracker1_info}, f, indent=2)

        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_t1_aligned, y_pred_t1, config.output_paths.plots, "tracker1_north_lstm"
        )

        # Tracker 2 (South)
        logger.info("\nFeature selection for Tracker 2 (South)...")
        tracker2_features = preprocessor.select_features_by_correlation(
            df_combined, "tracker2_south", threshold=correlation_threshold, top_n=top_n_features
        )

        X_train_t2, X_test_t2, y_train_t2, y_test_t2 = preprocessor.prepare_tracker_data(
            df_combined, "tracker2_south", tracker2_features
        )

        logger.info("Training LSTM Tracker 2 model...")
        lstm_tracker2 = LSTMTrackerModel(config, "Tracker2_south", tracker2_features, lookback=lookback)
        tracker2_info = lstm_tracker2.train(X_train_t2, y_train_t2)
        lstm_tracker2.save(config.output_paths.models, "tracker2_south_lstm")

        y_pred_t2 = lstm_tracker2.predict(X_test_t2)
        y_test_t2_aligned = y_test_t2[lookback:]

        metrics_t2 = RegressionMetricsCalculator.compute_all_metrics(y_test_t2_aligned, y_pred_t2)
        logger.info(f"LSTM Tracker 2 (South): RMSE={metrics_t2['rmse']:.4f}, MAE={metrics_t2['mae']:.4f}, R²={metrics_t2['r2']:.4f}")

        metrics_path = config.output_paths.results / "tracker2_south_lstm_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({**metrics_t2, **tracker2_info}, f, indent=2)

        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_t2_aligned, y_pred_t2, config.output_paths.plots, "tracker2_south_lstm"
        )

        # Combined
        logger.info("\nEvaluating combined LSTM tracker forecast...")
        y_pred_total_lstm = y_pred_t1 + y_pred_t2
        y_test_total_subset = y_test_t1_aligned + y_test_t2_aligned

        metrics_lstm_advanced = RegressionMetricsCalculator.compute_all_metrics(
            y_test_total_subset, y_pred_total_lstm
        )
        logger.info(f"LSTM Advanced (Combined): RMSE={metrics_lstm_advanced['rmse']:.4f}, MAE={metrics_lstm_advanced['mae']:.4f}, R²={metrics_lstm_advanced['r2']:.4f}")

        metrics_path = config.output_paths.results / "advanced_combined_lstm_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_lstm_advanced, f, indent=2)

        RegressionVisualizer.plot_all_regression_visualizations(
            y_test_total_subset, y_pred_total_lstm,
            config.output_paths.plots, "advanced_combined_trackers_lstm"
        )

        # ========== SUMMARY ==========
        logger.info("=" * 80)
        logger.info("LSTM TRAINING COMPLETE - RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nLSTM Baseline (GHI sequence):")
        logger.info(f"  LSTM Baseline: RMSE={metrics_lstm['rmse']:.4f}, R²={metrics_lstm['r2']:.4f}")
        logger.info(f"\nLSTM Advanced (Multi-feature trackers):")
        logger.info(f"  Combined Trackers: RMSE={metrics_lstm_advanced['rmse']:.4f}, R²={metrics_lstm_advanced['r2']:.4f}")

        improvement = ((metrics_lstm['rmse'] - metrics_lstm_advanced['rmse']) / metrics_lstm['rmse']) * 100
        logger.info(f"\nImprovement over LSTM baseline: {improvement:+.2f}% RMSE reduction")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
