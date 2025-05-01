from typing import Dict, List, Optional, Generator
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from config import FEATURE_ENGINEERING, DATA_PROCESSING, DIRS

logger = logging.getLogger('edupredict')

def create_sequential_features(
    activity_data: pd.DataFrame,
    chunk_size: int = DATA_PROCESSING['chunk_size']
) -> Generator[pd.DataFrame, None, None]:
    """
    Creates sequential features using chunked processing for memory efficiency.
    
    Args:
        activity_data: DataFrame containing student activity data
        chunk_size: Size of chunks for processing, defaults to value in DATA_PROCESSING config
        
    Returns:
        Generator yielding DataFrames containing sequential features for each chunk
    """
    try:
        # Sort data by student and timestamp
        sorted_data = activity_data.sort_values(['id_student', 'date'])
        
        # Process data in chunks
        for chunk_start in range(0, len(sorted_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(sorted_data))
            chunk = sorted_data.iloc[chunk_start:chunk_end]
            
            # Log memory usage
            _log_memory_usage("Processing chunk", chunk)
            
            # Process chunk
            chunk_features = _process_sequence_chunk(chunk)
            
            yield chunk_features
            
            # Clear memory
            del chunk
            
    except Exception as e:
        logger.error(f"Error creating sequential features: {str(e)}")
        raise

def _process_sequence_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a chunk of sequential data.
    
    Args:
        chunk: DataFrame containing a subset of student activity data
        
    Returns:
        DataFrame containing processed sequential features for the chunk
    """
    try:
        # Group by student
        grouped = chunk.groupby('id_student')
        
        # Calculate sequential metrics
        sequence_metrics = []
        for student_id, student_data in grouped:
            metrics = _calculate_student_metrics(student_data)
            metrics['id_student'] = student_id
            sequence_metrics.append(metrics)
            
        return pd.DataFrame(sequence_metrics)
        
    except Exception as e:
        logger.error(f"Error processing sequence chunk: {str(e)}")
        raise

def _calculate_student_metrics(student_data: pd.DataFrame) -> Dict:
    """
    Calculates sequential metrics for a student.
    
    Args:
        student_data: DataFrame containing activity data for a single student
        
    Returns:
        Dictionary containing calculated sequential metrics including:
        - avg_time_between_activities
        - max_time_between_activities
        - activity_regularity
        - total_activities
        - unique_activity_types
        - num_sessions
    """
    try:
        metrics = {}
        
        # Time-based metrics
        time_diffs = student_data['date'].diff()
        metrics['avg_time_between_activities'] = time_diffs.mean()
        metrics['max_time_between_activities'] = time_diffs.max()
        metrics['activity_regularity'] = time_diffs.std()
        
        # Activity pattern metrics
        metrics['total_activities'] = len(student_data)
        metrics['unique_activity_types'] = student_data['activity_type'].nunique()
        
        # Session metrics
        SESSION_GAP = pd.Timedelta(hours=1)
        session_breaks = time_diffs > SESSION_GAP
        metrics['num_sessions'] = session_breaks.sum() + 1
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating student metrics: {str(e)}")
        raise

def _log_memory_usage(message: str, df: pd.DataFrame) -> None:
    """Logs memory usage statistics."""
    try:
        usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
        logger.info(f"{message} - Memory usage: {usage:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error logging memory usage: {str(e)}")
        raise

def combine_sequential_features(
    feature_chunks: Generator[pd.DataFrame, None, None]
) -> pd.DataFrame:
    """
    Combines chunked sequential features into a single DataFrame.
    
    Args:
        feature_chunks: Generator of DataFrames containing sequential features
        
    Returns:
        DataFrame containing combined sequential features for all students
    """
    try:
        combined_features = []
        total_memory = 0
        
        for chunk in feature_chunks:
            _log_memory_usage("Processing feature chunk", chunk)
            combined_features.append(chunk)
            
            # Track total memory usage
            total_memory += chunk.memory_usage(deep=True).sum()
            
        combined_df = pd.concat(combined_features, ignore_index=True)
        
        # Log final memory usage
        _log_memory_usage("Final combined features", combined_df)
        
        # Export metadata
        _export_sequential_metadata(combined_df, total_memory)
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error combining sequential features: {str(e)}")
        raise

def _export_sequential_metadata(features: pd.DataFrame, total_memory: float) -> None:
    """Exports metadata about sequential features."""
    try:
        metadata_path = Path(DIRS['feature_metadata']) / 'sequential_features.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'feature_count': len(features.columns),
            'row_count': len(features),
            'memory_usage_mb': total_memory / 1024**2,
            'feature_names': features.columns.tolist(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Exported sequential feature metadata to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error exporting sequential metadata: {str(e)}")
        raise

def process_in_chunks(df: pd.DataFrame, 
                     func: callable,
                     chunk_size: int = 10000,
                     **kwargs) -> pd.DataFrame:
    """
    Process large dataframes in chunks to manage memory usage.
    
    Args:
        df: Input dataframe to be processed
        func: Function to apply to each chunk
        chunk_size: Number of rows per chunk, defaults to 10000
        **kwargs: Additional arguments for the processing function
        
    Returns:
        DataFrame containing the processed results from all chunks
    """
    chunks = []
    for start_idx in range(0, len(df), chunk_size):
        end_idx = min(start_idx + chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]
        
        # Process chunk
        processed_chunk = func(chunk, **kwargs)
        chunks.append(processed_chunk)
        
        # Log progress
        progress = (end_idx / len(df)) * 100
        logger.info(f"Processed {progress:.1f}% of data")
        
    return pd.concat(chunks, axis=0)

def generate_sequential_features(
    df: pd.DataFrame,
    target_col: str,
    groupby_cols: List[str],
    window_sizes: List[int] = FEATURE_ENGINEERING['window_sizes'],
    chunk_size: int = DATA_PROCESSING['chunk_size']
) -> pd.DataFrame:
    """
    Generate sequential features with memory-efficient chunked processing.
    
    Args:
        df: Input DataFrame containing data for feature generation
        target_col: Column name for the target variable
        groupby_cols: List of column names to group data by
        window_sizes: List of window sizes for rolling statistics, defaults to config
        chunk_size: Number of rows per chunk, defaults to config value
        
    Returns:
        DataFrame containing generated sequential features
    """
    monitor_memory_usage()  # Monitor initial memory state
    
    def process_chunk(chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features = []
        for window in kwargs['window_sizes']:
            # Rolling statistics
            rolled = chunk.groupby(kwargs['groupby_cols'])[kwargs['target_col']].rolling(
                window=window, min_periods=1
            )
            
            features.extend([
                rolled.mean().reset_index(drop=True).rename(f'{kwargs["target_col"]}_mean_{window}'),
                rolled.std().reset_index(drop=True).rename(f'{kwargs["target_col"]}_std_{window}'),
                rolled.max().reset_index(drop=True).rename(f'{kwargs["target_col"]}_max_{window}'),
                rolled.min().reset_index(drop=True).rename(f'{kwargs["target_col"]}_min_{window}')
            ])
            
            # Log progress
            logger.debug(f"Processed window size {window} for chunk")
        
        monitor_memory_usage()  # Monitor memory during processing
        return pd.concat(features, axis=1)
    
    # Process in chunks
    sequential_features = process_in_chunks(
        df,
        chunk_size=chunk_size,
        func=process_chunk,
        window_sizes=window_sizes,
        target_col=target_col,
        groupby_cols=groupby_cols
    )
    
    # Final memory check
    monitor_memory_usage()
    
    return sequential_features

def monitor_memory_usage():
    """Monitor memory usage during feature generation."""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    logger.info(f"Current memory usage: {memory_usage_mb:.2f} MB")
    
    if memory_usage_mb > 1000:  # Warning if usage exceeds 1GB
        logger.warning(f"High memory usage detected: {memory_usage_mb:.2f} MB")
    
    return memory_usage_mb