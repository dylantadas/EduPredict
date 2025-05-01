import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works with tqdm progress bars"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def setup_logger(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Sets up the main logger for the EduPredict system.
    
    Args:
        log_dir: Directory where log files will be stored
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger instance configured with file and console handlers
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logger
    logger = logging.getLogger('edupredict')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create and configure file handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / 'edupredict.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    
    # Create and configure console handler with tqdm compatibility
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_memory_usage(logger: logging.Logger, operation: str, memory_mb: float):
    """
    Logs memory usage for a given operation.
    
    Args:
        logger: Logger instance
        operation: Description of the operation being logged
        memory_mb: Memory usage in megabytes
    """
    logger.info(f"Memory usage after {operation}: {memory_mb:.2f} MB")

def log_progress(logger: logging.Logger, stage: str, current: int, total: int):
    """
    Logs progress of a processing stage.
    
    Args:
        logger: Logger instance
        stage: Name of the processing stage
        current: Current progress count
        total: Total items to process
    """
    percentage = (current / total) * 100
    logger.info(f"{stage} progress: {current}/{total} ({percentage:.1f}%)")