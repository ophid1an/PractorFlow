import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from typing import Optional

# Ensure logs directory always exists (only used if file logging is enabled)
os.makedirs("logs", exist_ok=True)

# Cache for reusable loggers
_logger_cache = {}

# ANSI color codes for stdout logging
_LEVEL_COLORS = {
    logging.DEBUG: "\033[90m",      # Gray
    logging.INFO: "\033[32m",       # Green
    logging.WARNING: "\033[33m",    # Yellow
    logging.ERROR: "\033[31m",      # Red
    logging.CRITICAL: "\033[1;31m", # Bold Red
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{_RESET}" if color else message


def _normalize_level(level: str) -> int:
    """
    Normalize string log level to logging module level.
    Defaults to NOTSET if invalid.
    """
    if not isinstance(level, str):
        return logging.NOTSET
    return logging._nameToLevel.get(level.upper(), logging.INFO)


def get_logger(
    logger_name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = 2 * 1024 * 1024,
    backup_count: int = 2,
    stdout: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a cached logger.

    Stdout logging is enabled by default (cloud-friendly).
    File logging is optional and disabled unless log_file is provided.

    Args:
        logger_name (str): Unique name for the logger (e.g., "mylogger").
        log_file (Optional[str]): Path to the log file or None to disable file logging.
        level (str): Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        max_bytes (int): Max log file size before rotation (default 2 MB).
        backup_count (int): Number of rotated logs to keep (default 2).
        stdout (bool): Enable logging to stdout (default True).

    Returns:
        logging.Logger: Configured and reusable logger instance.
    """
    if logger_name in _logger_cache:
        return _logger_cache[logger_name]

    log_level = _normalize_level(level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    if not logger.handlers:
        base_format = "%(asctime)s - %(levelname)s - %(message)s"

        # Stdout handler (default)
        if stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(ColorFormatter(base_format))
            logger.addHandler(stream_handler)

        # Optional file handler
        if log_file:
            file_handler = RotatingFileHandler(
                log_file,
                mode="w",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(base_format))
            logger.addHandler(file_handler)

    _logger_cache[logger_name] = logger
    return logger
