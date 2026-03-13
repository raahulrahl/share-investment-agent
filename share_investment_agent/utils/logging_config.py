"""Logging configuration for share investment agent."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    """Set up logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional - create logs directory if needed)
    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        log_file = logs_dir / f"share_investment_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # If we can't create file handler, continue with console only
        logger.warning(f"Failed to create file handler: {e}")
        pass

    return logger


# Success and error icons for console output
SUCCESS_ICON = "✅"
ERROR_ICON = "❌"
WARNING_ICON = "⚠️"
WAIT_ICON = "⏳"
INFO_ICON = "i"
