"""Logging configuration and setup."""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path = Path("logs"), log_level: str = "INFO"):
    """
    Configure logging with loguru.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove default handler
    logger.remove()

    # Add console handler with formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"log_{timestamp}.log"

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Log level: {log_level}")
