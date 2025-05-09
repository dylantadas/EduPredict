import psutil
import logging
import time
import functools
from typing import Optional, Callable, Any
from contextlib import contextmanager
from tqdm import tqdm

# Memory tracking state
_last_warning_time = 0
_warning_interval = 300  # 5 minutes between warnings
_memory_threshold = 1000  # MB - Reduced from 1500MB to 1000MB for earlier warnings

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
            # Get the first logger argument if one is passed, otherwise use the central logger
            logger = None
            for arg in args:
                if isinstance(arg, logging.Logger):
                    logger = arg
                    break
            
            if logger is None:
                for _, arg_val in kwargs.items():
                    if isinstance(arg_val, logging.Logger):
                        logger = arg_val
                        break
            
            # If no logger was passed, use the central one
            if logger is None:
                logger = logging.getLogger('edupredict')
            
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

def monitor_memory_usage(func_or_context=None, force_gc: bool = False, logger: Optional[logging.Logger] = None) -> Any:
    """
    Monitor memory usage. Can be used as a decorator or called directly.
    
    When used as a decorator:
        @monitor_memory_usage
        def my_function():
            ...
    
    When used as a function call:
        monitor_memory_usage("Some context")
    
    Args:
        func_or_context: Either a function (when used as decorator) or a context string
        force_gc: Whether to force garbage collection if memory usage is high
        logger: Optional logger instance to use
        
    Returns:
        Decorator: The decorated function when used as a decorator
        float: Current memory usage in MB when called directly
    """
    # Check if this is being used as a decorator (func_or_context is a callable)
    if callable(func_or_context):
        # This is being used as a decorator
        @functools.wraps(func_or_context)
        def wrapper(*args, **kwargs):
            # Get the first logger argument if one is passed, otherwise use the central logger
            log = logger
            if log is None:
                for arg in args:
                    if isinstance(arg, logging.Logger):
                        log = arg
                        break
                
                if log is None:
                    for _, arg_val in kwargs.items():
                        if isinstance(arg_val, logging.Logger):
                            log = arg_val
                            break
            
            # If no logger was passed, use the central one
            if log is None:
                log = logging.getLogger('edupredict')
            
            # Get memory usage before function execution
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Execute the function
            result = func_or_context(*args, **kwargs)
            
            # Get memory usage after function execution
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_diff = memory_after - memory_before
            
            # Log memory usage
            func_name = getattr(func_or_context, '__name__', 'function')
            log.debug(f"Memory usage for {func_name}: {memory_after:.2f} MB (Change: {memory_diff:+.2f} MB)")
            
            # Handle high memory usage
            global _last_warning_time
            current_time = time.time()
            if memory_after > _memory_threshold and (current_time - _last_warning_time) > _warning_interval:
                log.warning(f"High memory usage detected in {func_name}: {memory_after:.2f} MB")
                _last_warning_time = current_time
                
                if force_gc or memory_after > _memory_threshold * 1.5:
                    import gc
                    before_gc = memory_after
                    gc.collect()
                    # Get updated memory info after collection
                    after_gc = process.memory_info().rss / 1024 / 1024
                    if (before_gc - after_gc) > 100:  # Only log if significant memory was freed
                        log.info(f"{func_name} - Memory reduced by {before_gc - after_gc:.2f} MB after garbage collection")
            
            return result
        
        return wrapper
    
    else:
        # This is being called directly, not as a decorator
        try:
            # Use provided logger or get the central one
            if logger is None:
                logger = logging.getLogger('edupredict')
            
            context = func_or_context  # In this case, func_or_context is the context string
            
            global _last_warning_time
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            
            if context:
                logger.debug(f"{context} - Current memory usage: {memory_usage_mb:.2f} MB")
            
            current_time = time.time()
            if memory_usage_mb > _memory_threshold:
                if (current_time - _last_warning_time) > _warning_interval:
                    logger.warning(f"High memory usage detected: {memory_usage_mb:.2f} MB")
                    _last_warning_time = current_time
                
                # Force garbage collection when memory usage is high
                if force_gc or memory_usage_mb > _memory_threshold * 1.5:
                    import gc
                    before_gc = memory_usage_mb
                    gc.collect()
                    # Get updated memory info after collection
                    memory_info = process.memory_info()
                    after_gc = memory_info.rss / 1024 / 1024
                    if context and (before_gc - after_gc) > 100:  # Only log if significant memory was freed
                        logger.info(f"{context} - Memory reduced by {before_gc - after_gc:.2f} MB after garbage collection")
            
            return memory_usage_mb
            
        except Exception as e:
            # Use provided logger or get the central one
            if logger is None:
                logger = logging.getLogger('edupredict')
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