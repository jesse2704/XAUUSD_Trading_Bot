"""Logging configuration for the XAUUSD Trading Bot."""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from config.settings import settings


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and return a configured logger instance.

    Args:
        name: Logger name (typically the module's __name__).
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to settings.
        log_file: Path to log file. Defaults to settings.

    Returns:
        Configured logging.Logger instance.
    """
    log_level_str = level or settings.logging.level
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_file_path = log_file or settings.logging.log_file

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotating)
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=settings.logging.max_bytes,
        backupCount=settings.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module name.

    Convenience wrapper around setup_logger that uses the global settings.

    Args:
        name: Logger name (pass __name__ from calling module).

    Returns:
        Configured logging.Logger instance.
    """
    return setup_logger(name)
