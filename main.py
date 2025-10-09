"""Main entry point for the PV solar tracker prediction pipeline.

This script orchestrates the complete pipeline:
1. Preprocessing: Load and merge raw data
2. Processing: Add features, split train/test, and scale
3. Training: Train LSTM models for all trackers
4. Prediction: Make predictions on test data using trained models
5. Evaluation: Calculate metrics and generate visualizations

Each step can be run independently via command-line arguments.
"""

import argparse
import logging
from pathlib import Path

from src.config.models import load_config
from src.evaluation.evaluator import ModelEvaluator
from src.logging.logging_setup import setup_logging
from src.models.predictor import LSTMPredictor
from src.models.trainer import LSTMTrainer
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.processor import DataProcessor

logger = logging.getLogger(__name__)


def run_preprocessing(config):
    """Step 1: Preprocess raw data (merge PV, irradiance, weather data).

    Input: Raw CSV files from data_paths in config
    Output: Preprocessed data in preprocessing.output_dir
    """
    logger.info("="*60)
    logger.info("STEP 1: PREPROCESSING")
    logger.info("="*60)

    pipeline = PreprocessingPipeline(config)
    pipeline.run()

    logger.info("✓ Preprocessing completed")


def run_processing(config):
    """Step 2: Process data (add features, split, scale).

    Input: Preprocessed data from preprocessing.output_dir
    Output: Train/test splits in processing.output_dir
    """
    logger.info("="*60)
    logger.info("STEP 2: PROCESSING")
    logger.info("="*60)

    processor = DataProcessor(config)
    processor.run()

    logger.info("✓ Processing completed")


def run_training(config):
    """Step 3: Train LSTM models for all trackers.

    Input: Train/test splits from processing.output_dir
    Output: Trained models in output_paths.models
    """
    logger.info("="*60)
    logger.info("STEP 3: TRAINING")
    logger.info("="*60)

    # Define datasets to train
    datasets = [
        {
            "name": "total",
            "train_path": config.processing.output_dir / "total" / "train.csv",
            "test_path": config.processing.output_dir / "total" / "test.csv",
            "target_col": "Solarproduktion",
        },
        {
            "name": "north",
            "train_path": config.processing.output_dir / "north" / "train.csv",
            "test_path": config.processing.output_dir / "north" / "test.csv",
            "target_col": "pv_production_north",
        },
        {
            "name": "south",
            "train_path": config.processing.output_dir / "south" / "train.csv",
            "test_path": config.processing.output_dir / "south" / "test.csv",
            "target_col": "pv_production_south",
        },
    ]

    # Train model for each dataset
    for dataset in datasets:
        logger.info(f"\nTraining LSTM model for: {dataset['name'].upper()}")

        # Create trainer
        trainer = LSTMTrainer(config)

        # Load data
        train_loader, test_loader, input_size = trainer.load_data(
            train_path=dataset["train_path"],
            test_path=dataset["test_path"],
            target_col=dataset["target_col"],
        )

        # Initialize model
        trainer.initialize_model(input_size)

        # Train
        history = trainer.train(train_loader, test_loader)

        # Save model
        trainer.save_model(
            output_path=config.output_paths.models,
            dataset_name=dataset["name"],
        )

        logger.info(f"✓ Completed training for {dataset['name']}")
        logger.info(f"  Final train loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"  Final val loss: {history['val_loss'][-1]:.6f}")

    logger.info("✓ All models trained successfully")


def run_prediction(config):
    """Step 4: Make predictions on test data.

    Input: Trained models from output_paths.models + test data from processing.output_dir
    Output: Predictions in output_paths.predictions
    """
    logger.info("="*60)
    logger.info("STEP 4: PREDICTION")
    logger.info("="*60)

    # Define datasets to predict
    datasets = [
        {
            "name": "total",
            "test_path": config.processing.output_dir / "total" / "test.csv",
            "target_col": "Solarproduktion",
        },
        {
            "name": "north",
            "test_path": config.processing.output_dir / "north" / "test.csv",
            "target_col": "pv_production_north",
        },
        {
            "name": "south",
            "test_path": config.processing.output_dir / "south" / "test.csv",
            "target_col": "pv_production_south",
        },
    ]

    # Make predictions for each dataset
    for dataset in datasets:
        logger.info(f"\nMaking predictions for: {dataset['name'].upper()}")

        # Create predictor
        predictor = LSTMPredictor(config)

        # Run prediction pipeline
        predictions_df = predictor.run_prediction(
            model_path=config.output_paths.models,
            test_path=dataset["test_path"],
            output_path=config.output_paths.predictions,
            dataset_name=dataset["name"],
            target_col=dataset["target_col"],
        )

        logger.info(f"✓ Completed predictions for {dataset['name']}")
        logger.info(f"  Total predictions: {len(predictions_df)}")

    logger.info("✓ All predictions completed successfully")


def run_evaluation(config):
    """Step 5: Evaluate model predictions.

    Input: Predictions from output_paths.predictions
    Output: Metrics and plots in output_paths.results
    """
    logger.info("="*60)
    logger.info("STEP 5: EVALUATION")
    logger.info("="*60)

    # Define datasets to evaluate
    datasets = ["total", "north", "south"]

    # Create evaluator
    evaluator = ModelEvaluator(config)

    # Evaluate each dataset
    all_metrics = {}
    for dataset_name in datasets:
        logger.info(f"\nEvaluating dataset: {dataset_name.upper()}")

        predictions_file = config.output_paths.predictions / f"predictions_{dataset_name}.csv"

        if not predictions_file.exists():
            logger.warning(f"Predictions file not found: {predictions_file}")
            continue

        # Run evaluation
        metrics = evaluator.evaluate_dataset(
            predictions_path=predictions_file,
            output_path=config.output_paths.results,
            dataset_name=dataset_name,
        )

        all_metrics[dataset_name] = metrics

    logger.info("✓ All evaluations completed successfully")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    for dataset_name, metrics in all_metrics.items():
        logger.info(f"\n{dataset_name.upper()}:")
        logger.info(f"  MAE:  {metrics['mae']:.2f} W")
        logger.info(f"  RMSE: {metrics['rmse']:.2f} W")
        logger.info(f"  R²:   {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f} %")


def main():
    """Run the complete pipeline or individual steps."""
    parser = argparse.ArgumentParser(
        description="PV Solar Tracker Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py

  # Run only preprocessing
  python main.py --step preprocessing

  # Run only processing (requires preprocessed data)
  python main.py --step processing

  # Run only training (requires processed data)
  python main.py --step training

  # Run only prediction (requires trained models)
  python main.py --step prediction

  # Run only evaluation (requires predictions)
  python main.py --step evaluation

  # Run from prediction onwards (prediction + evaluation)
  python main.py --step prediction --continue
        """
    )

    parser.add_argument(
        "--step",
        choices=["preprocessing", "processing", "training", "prediction", "evaluation"],
        help="Run a specific pipeline step (default: run all steps)",
    )

    parser.add_argument(
        "--continue",
        dest="continue_pipeline",
        action="store_true",
        help="Continue with subsequent steps after the specified step",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file (default: config.yaml)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(log_dir=config.output_paths.logs, log_level="INFO")

    logger.info("Starting PV Solar Tracker Prediction Pipeline")
    logger.info(f"Configuration loaded from: {args.config}")

    # Define pipeline steps
    steps = {
        "preprocessing": run_preprocessing,
        "processing": run_processing,
        "training": run_training,
        "prediction": run_prediction,
        "evaluation": run_evaluation,
    }

    # Determine which steps to run
    if args.step:
        # Run specific step
        step_order = ["preprocessing", "processing", "training", "prediction", "evaluation"]
        start_idx = step_order.index(args.step)

        if args.continue_pipeline:
            # Run from this step onwards
            steps_to_run = step_order[start_idx:]
        else:
            # Run only this step
            steps_to_run = [args.step]
    else:
        # Run all steps
        steps_to_run = ["preprocessing", "processing", "training", "prediction", "evaluation"]

    # Execute steps
    for step_name in steps_to_run:
        steps[step_name](config)

    logger.info("="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)


if __name__ == "__main__":
    main()
