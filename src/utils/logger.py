"""
Logging utility module for AI Media Trends Insight Analyzer.
Provides centralized logging configuration and helper functions.
"""

import logging
import logging.handlers
import os
from typing import Optional

from src.config import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOG_CONSOLE_FORMAT,
    LOG_DIR,
    LOG_FILE,
)

DEFAULT_LOGGER_NAME = 'ai_media_trends'

def setup_logger(
    logger_name: str = DEFAULT_LOGGER_NAME,
    log_file: Optional[str] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        logger_name: Name of the logger
        log_file: Path to log file. If None, uses default from config
        level: Logging level. If None, uses default from config
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    
    # Set log level
    log_level = getattr(logging, (level or LOG_LEVEL).upper())
    logger.setLevel(log_level)
    
    # Prevent adding handlers if they already exist
    if logger.handlers:
        return logger
        
    # File handler with rotation
    if not log_file:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, LOG_FILE)
        
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(
        logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(fmt=LOG_CONSOLE_FORMAT)
    )
    logger.addHandler(console_handler)
    
    return logger

# Helper functions
def log_error(message: str, logger_name: str = DEFAULT_LOGGER_NAME) -> None:
    """Log an error message."""
    logging.getLogger(logger_name).error(message)

def log_warning(message: str, logger_name: str = DEFAULT_LOGGER_NAME) -> None:
    """Log a warning message."""
    logging.getLogger(logger_name).warning(message)

def log_info(message: str, logger_name: str = DEFAULT_LOGGER_NAME) -> None:
    """Log an info message."""
    logging.getLogger(logger_name).info(message)

def log_debug(message: str, logger_name: str = DEFAULT_LOGGER_NAME) -> None:
    """Log a debug message."""
    logging.getLogger(logger_name).debug(message)

# Initialize default logger
default_logger = setup_logger() 