from typing import Dict, List, Optional, Generator
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from config import FEATURE_ENGINEERING, DATA_PROCESSING, DIRS
from utils.monitoring_utils import monitor_memory_usage, track_progress
from feature_engineering.feature_selector import NumpyJSONEncoder

logger = logging.getLogger('edupredict')

def create_sequential_features(
    activity_data: pd.DataFrame,
    chunk_size: int = DATA_PROCESSING['chunk_size']
) -> Generator[pd.DataFrame, None, None]:
    """Creates sequential features using chunked processing for memory efficiency."""
    try:
        # Sort data by student and timestamp
        sorted_data = activity_data.sort_values(['id_student', 'date'])
        total_chunks = (len(sorted_data) + chunk_size - 1) // chunk_size
        
        # Process data in chunks with progress tracking
        chunks = range(0, len(sorted_data), chunk_size)
        for chunk_start in track_progress(chunks, desc="Processing sequential features", total=total_chunks):
            chunk_end = min(chunk_start + chunk_size, len(sorted_data))
            chunk = sorted_data.iloc[chunk_start:chunk_end]
            
            # Only monitor memory usage every 10 chunks to reduce log spam
            if (chunk_start // chunk_size) % 10 == 0:
                monitor_memory_usage(f"Processing chunk {chunk_start}-{chunk_end}")
            
            # Process chunk
            chunk_features = _process_sequence_chunk(chunk)
            
            yield chunk_features
            
            # Clear memory
            del chunk
            
    except Exception as e:
        logger.error(f"Error creating sequential features: {str(e)}")
        raise

def _process_sequence_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Processes a chunk of sequential data."""
    try:
        # Monitor memory before processing
        monitor_memory_usage("Before chunk processing")
        
        # Group by student
        grouped = chunk.groupby('id_student')
        
        # Calculate sequential metrics
        sequence_metrics = []
        for student_id, student_data in grouped:
            metrics = _calculate_student_metrics(student_data)
            metrics['id_student'] = student_id
            sequence_metrics.append(metrics)
        
        # Monitor memory after processing
        monitor_memory_usage("After chunk processing")
        
        return pd.DataFrame(sequence_metrics)
        
    except Exception as e:
        logger.error(f"Error processing sequence chunk: {str(e)}")
        raise

def _calculate_student_metrics(student_data: pd.DataFrame) -> Dict:
    """
    Calculates sequential metrics for a student with enhanced temporal context.
    Properly handles the timeline relative to module start date.
    """
    try:
        metrics = {}
        
        # Ensure date is properly handled relative to module start
        student_data['absolute_day'] = student_data['date'].abs()
        student_data['is_before_start'] = student_data['date'] < 0
        
        # Time-based metrics with module context
        time_diffs = student_data['date'].diff()
        
        # General engagement timing
        metrics['avg_time_between_activities'] = time_diffs.mean()
        metrics['max_time_between_activities'] = time_diffs.max() if not pd.isna(time_diffs.max()) else 0
        metrics['activity_regularity'] = time_diffs.std() if not pd.isna(time_diffs.std()) else 0
        
        # Pre/Post module start metrics
        pre_module = student_data[student_data['is_before_start']]
        post_module = student_data[~student_data['is_before_start']]
        
        metrics['pre_module_activity_count'] = len(pre_module)
        metrics['post_module_activity_count'] = len(post_module)
        metrics['pre_post_ratio'] = (len(pre_module) / len(post_module)) if len(post_module) > 0 else 0
        
        # Activity pattern metrics with temporal context
        metrics['total_activities'] = len(student_data)
        metrics['unique_activity_types'] = student_data['activity_type'].nunique()
        metrics['activity_type_entropy'] = calculate_entropy(student_data['activity_type'])
        
        # Early vs late engagement patterns
        if len(student_data) >= 2:
            midpoint = student_data['date'].median()
            early_phase = student_data[student_data['date'] <= midpoint]
            late_phase = student_data[student_data['date'] > midpoint]
            
            metrics['early_phase_intensity'] = early_phase['sum_click'].mean() if len(early_phase) > 0 else 0
            metrics['late_phase_intensity'] = late_phase['sum_click'].mean() if len(late_phase) > 0 else 0
            metrics['engagement_phase_ratio'] = (
                metrics['early_phase_intensity'] / metrics['late_phase_intensity']
                if metrics['late_phase_intensity'] > 0 else 0
            )
        
        # Session analysis
        SESSION_GAP = pd.Timedelta(hours=1)
        session_breaks = time_diffs > SESSION_GAP
        metrics['num_sessions'] = session_breaks.sum() + 1
        
        if metrics['num_sessions'] > 0:
            metrics['avg_activities_per_session'] = len(student_data) / metrics['num_sessions']
        else:
            metrics['avg_activities_per_session'] = 0
        
        # Activity transitions
        if len(student_data) > 1:
            transitions = student_data['activity_type'].shift() + '_to_' + student_data['activity_type']
            metrics['unique_transitions'] = transitions.nunique()
            metrics['transition_entropy'] = calculate_entropy(transitions)
        else:
            metrics['unique_transitions'] = 0
            metrics['transition_entropy'] = 0
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating student metrics: {str(e)}")
        raise

def calculate_entropy(series: pd.Series) -> float:
    """Calculate Shannon entropy of a series."""
    value_counts = series.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts))

def combine_sequential_features(
    feature_chunks: Generator[pd.DataFrame, None, None]
) -> pd.DataFrame:
    """Combines chunked sequential features."""
    try:
        combined_features = []
        total_memory = 0
        
        # Track progress while combining chunks
        for i, chunk in enumerate(track_progress(feature_chunks, desc="Combining feature chunks")):
            combined_features.append(chunk)
            
            # Monitor memory periodically
            if i % 10 == 0:
                total_memory = monitor_memory_usage(f"After processing {i} chunks")
            
        combined_df = pd.concat(combined_features, ignore_index=True)
        
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
            'memory_usage_mb': float(total_memory / 1024**2),
            'feature_names': features.columns.tolist(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
            
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

def generate_sequential_features(df: pd.DataFrame,
                              window_sizes: List[int],
                              target_col: str,
                              groupby_cols: List[str],
                              chunk_size: int = 10000) -> pd.DataFrame:
    """
    Generate sequential features with memory-efficient chunked processing.
    """
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
    
    return sequential_features