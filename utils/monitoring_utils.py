import psutil
import logging
from typing import Optional

logger = logging.getLogger('edupredict')

def monitor_memory_usage(context: Optional[str] = None) -> float:
    """
    Monitor memory usage during feature generation and processing.
    
    Args:
        context: Optional string describing the current operation context
        
    Returns:
        Current memory usage in MB
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        
        message = f"Current memory usage: {memory_usage_mb:.2f} MB"
        if context:
            message = f"{context} - {message}"
            
        logger.info(message)
        
        if memory_usage_mb > 1000:  # Warning if usage exceeds 1GB
            logger.warning(f"High memory usage detected: {memory_usage_mb:.2f} MB")
        
        return memory_usage_mb
        
    except Exception as e:
        logger.error(f"Error monitoring memory usage: {str(e)}")
        return -1