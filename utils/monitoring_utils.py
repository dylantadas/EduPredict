import psutil
import logging
import time
import functools
from typing import Optional, Callable, Any
from contextlib import contextmanager
from tqdm import tqdm

logger = logging.getLogger('edupredict')

# Memory tracking state
_last_warning_time = 0
_warning_interval = 300  # 5 minutes between warnings
_memory_threshold = 1500  # MB

def track_execution_time(func: Optional[Callable] = None, context: Optional[str] = None) -> Callable:
    """
    Decorator to track execution time of functions.
    
    Args:
        func: The function to be decorated
        context: Optional string describing the context of execution
        
    Returns:
        Wrapped function that logs execution time
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = f(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            message = f"Execution time: {execution_time:.2f} seconds"
            if context:
                message = f"{context} - {message}"
            
            logger.info(message)
            return result
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

def monitor_memory_usage(context: Optional[str] = None) -> float:
    """
    Monitor memory usage during feature generation and processing.
    
    Args:
        context: Optional string describing the current operation context
        
    Returns:
        Current memory usage in MB
    """
    try:
        global _last_warning_time
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        
        if context:
            logger.debug(f"{context} - Current memory usage: {memory_usage_mb:.2f} MB")
        
        current_time = time.time()
        if memory_usage_mb > _memory_threshold and (current_time - _last_warning_time) > _warning_interval:
            logger.warning(f"High memory usage detected: {memory_usage_mb:.2f} MB")
            _last_warning_time = current_time
        
        return memory_usage_mb
        
    except Exception as e:
        logger.error(f"Error monitoring memory usage: {str(e)}")
        return -1

def track_progress(iterable, desc: str = None, total: Optional[int] = None) -> Any:
    """
    Wrapper around tqdm to provide progress tracking with logging integration.
    
    Args:
        iterable: Iterable to track progress for
        desc: Description of the progress bar
        total: Total number of items (optional)
        
    Returns:
        tqdm wrapped iterable
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        mininterval=1.0,
        maxinterval=5.0,
        miniters=1,
        unit='chunks'
    )