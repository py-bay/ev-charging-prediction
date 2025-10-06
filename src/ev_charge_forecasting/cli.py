"""Command-line interface for EV charge forecasting."""

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from .config import Config, load_config
from .evaluation import ModelEvaluator
from .models import LSTMModel, RandomForestModel
from .preprocessing import DataPreprocessor
from .utils import setup_logging

app = typer.Typer(
    help="EV Charge Forecasting - ML-based optimal charging window prediction",
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
    feature_set: str = typer.Option(
        "combined",
        "--features",
        "-f",
        help="Feature set to use: weather, irradiance, or combined",
    ),
    label_threshold: float = typer.Option(
        1.0,
        "--threshold",
        "-t",
        help="Threshold for label generation",
    ),
    label_method: str = typer.Option(
        "absolute",
        "--method",
        "-m",
        help="Label generation method: absolute or relative",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Preprocess data: load, clean, resample, engineer features, and generate labels.
    """
    # Setup logging
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("EV CHARGE FORECASTING - PREPROCESSING")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)

        # Run preprocessing pipeline
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(
            feature_set=feature_set,
            label_threshold=label_threshold,
            label_method=label_method,
        )

        # Save processed data
        preprocessor.save_processed_data()

        logger.info("Preprocessing completed successfully!")
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def train(
    config_path: str = typer.Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    model: str = typer.Option(
        "randomforest",
        "--model",
        "-m",
        help="Model to train: randomforest or lstm",
    ),
    feature_set: str = typer.Option(
        "combined",
        "--features",
        "-f",
        help="Feature set to use: weather, irradiance, or combined",
    ),
    label_threshold: float = typer.Option(
        1.0,
        "--threshold",
        "-t",
        help="Threshold for label generation",
    ),
    label_method: str = typer.Option(
        "absolute",
        "--method",
        help="Label generation method: absolute or relative",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Train a model on preprocessed data.
    """
    # Setup logging
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info(f"EV CHARGE FORECASTING - TRAINING {model.upper()}")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Validate model name
        if model not in ["randomforest", "lstm"]:
            logger.error(f"Unknown model: {model}. Choose 'randomforest' or 'lstm'")
            raise typer.Exit(code=1)

        # Preprocess data
        logger.info("Running preprocessing pipeline...")
        preprocessor = DataPreprocessor(config)
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(
            feature_set=feature_set,
            label_threshold=label_threshold,
            label_method=label_method,
        )

        # Initialize model
        if model == "randomforest":
            model_instance = RandomForestModel(config)
        else:  # lstm
            model_instance = LSTMModel(config)

        # Train model
        train_info = model_instance.train(X_train, y_train)
        logger.info(f"Training info: {train_info}")

        # Save model
        model_name = f"{model}_{feature_set}_t{label_threshold}"
        model_instance.save(config.output_paths.models, model_name)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    config_path: str = typer.Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    model: str = typer.Option(
        "randomforest",
        "--model",
        "-m",
        help="Model to evaluate: randomforest or lstm",
    ),
    feature_set: str = typer.Option(
        "combined",
        "--features",
        "-f",
        help="Feature set to use: weather, irradiance, or combined",
    ),
    label_threshold: float = typer.Option(
        1.0,
        "--threshold",
        "-t",
        help="Threshold for label generation",
    ),
    label_method: str = typer.Option(
        "absolute",
        "--method",
        help="Label generation method: absolute or relative",
    ),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-p",
        help="Path to saved model (if not provided, will train new model)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Evaluate a trained model and generate metrics and visualizations.
    """
    # Setup logging
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info(f"EV CHARGE FORECASTING - EVALUATING {model.upper()}")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Validate model name
        if model not in ["randomforest", "lstm"]:
            logger.error(f"Unknown model: {model}. Choose 'randomforest' or 'lstm'")
            raise typer.Exit(code=1)

        # Preprocess data
        logger.info("Running preprocessing pipeline...")
        preprocessor = DataPreprocessor(config)
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(
            feature_set=feature_set,
            label_threshold=label_threshold,
            label_method=label_method,
        )

        # Initialize model
        if model == "randomforest":
            model_instance = RandomForestModel(config)
        else:  # lstm
            model_instance = LSTMModel(config)

        # Load or train model
        if model_path is not None:
            logger.info(f"Loading model from {model_path}")
            model_instance.load(Path(model_path))
        else:
            logger.info("No model path provided. Training new model...")
            model_instance.train(X_train, y_train)

        # Evaluate model
        evaluator = ModelEvaluator(config)
        model_name = f"{model}_{feature_set}_t{label_threshold}"
        results = evaluator.generate_evaluation_report(
            model_instance, X_test, y_test, model_name
        )

        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to {config.output_paths.results}")
        logger.info(f"Plots saved to {config.output_paths.plots}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def train_all(
    config_path: str = typer.Option(
        "config/config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    feature_set: str = typer.Option(
        "combined",
        "--features",
        "-f",
        help="Feature set to use: weather, irradiance, or combined",
    ),
    label_threshold: float = typer.Option(
        1.0,
        "--threshold",
        "-t",
        help="Threshold for label generation",
    ),
    label_method: str = typer.Option(
        "absolute",
        "--method",
        help="Label generation method: absolute or relative",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
):
    """
    Train and evaluate all models (RandomForest + LSTM).
    """
    # Setup logging
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("EV CHARGE FORECASTING - TRAINING AND EVALUATING ALL MODELS")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Preprocess data once
        logger.info("Running preprocessing pipeline...")
        preprocessor = DataPreprocessor(config)
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(
            feature_set=feature_set,
            label_threshold=label_threshold,
            label_method=label_method,
        )

        # Train and evaluate each model
        for model_type in config.models.enabled:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"Processing model: {model_type.upper()}")
            logger.info("=" * 80)

            # Initialize model
            if model_type == "randomforest":
                model_instance = RandomForestModel(config)
            elif model_type == "lstm":
                model_instance = LSTMModel(config)
            else:
                logger.warning(f"Unknown model type: {model_type}. Skipping.")
                continue

            # Train model
            logger.info(f"Training {model_type}...")
            train_info = model_instance.train(X_train, y_train)

            # Save model
            model_name = f"{model_type}_{feature_set}_t{label_threshold}"
            model_instance.save(config.output_paths.models, model_name)

            # Evaluate model
            logger.info(f"Evaluating {model_type}...")
            evaluator = ModelEvaluator(config)
            results = evaluator.generate_evaluation_report(
                model_instance, X_test, y_test, model_name
            )

        logger.info("")
        logger.info("=" * 80)
        logger.info("All models trained and evaluated successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training/evaluation failed: {e}")
        raise typer.Exit(code=1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
