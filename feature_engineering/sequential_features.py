from typing import Dict, List, Optional, Generator, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from config import FEATURE_ENGINEERING, DATA_PROCESSING, DIRS, TEMPORAL_CONFIG
from utils.monitoring_utils import monitor_memory_usage, track_progress, track_execution_time
from feature_engineering.feature_selector import NumpyJSONEncoder
from feature_engineering.demographic_features import load_features, save_features

logger = logging.getLogger('edupredict')

@track_execution_time
@monitor_memory_usage
def create_sequential_features(
    vle_data: pd.DataFrame,
    chunk_size: int = 10000,
    max_sequence_length: int = 100,
    min_activity_threshold: int = 5
) -> pd.DataFrame:
    """
    Creates sequential features from VLE data with enhanced memory management.
    
    Args:
        vle_data: DataFrame containing VLE interaction data
        chunk_size: Number of rows to process at once
        max_sequence_length: Maximum sequence length to consider
        min_activity_threshold: Minimum number of activities required
        
    Returns:
        DataFrame containing sequential features
    """
    try:
        # Convert datatypes to more memory-efficient ones
        optimize_dtypes = {
            'id_student': 'int32',
            'code_module': 'category',
            'code_presentation': 'category',
            'date': 'float32',  # Days from module start
            'sum_click': 'float32',
            'id_site': 'int32',
            'activity_type': 'category'
        }
        
        # Only convert columns that exist and match expected types
        for col, dtype in optimize_dtypes.items():
            if col in vle_data.columns:
                try:
                    vle_data[col] = vle_data[col].astype(dtype)
                except:
                    logger.warning(f"Could not convert {col} to {dtype}")
        
        # Sort data by student ID and date for sequential processing
        logger.info("Sorting data for sequential feature extraction...")
        sorted_data = vle_data.sort_values(['id_student', 'date']).reset_index(drop=True)
        
        # Get the initial memory usage
        initial_memory = sorted_data.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Process in smaller chunks to manage memory
        feature_chunks = []
        logger.info(f"Processing {len(sorted_data)} rows in chunks of {chunk_size}")
        
        # Calculate total number of chunks for better progress tracking
        total_chunks = (len(sorted_data) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(0, total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(sorted_data))
            
            # Extract chunk with only necessary columns to reduce memory
            chunk = sorted_data.iloc[chunk_start:chunk_end][
                ['id_student', 'code_module', 'code_presentation', 'date', 
                 'sum_click', 'activity_type', 'id_site']
            ].copy()
            
            # Only monitor memory usage every few chunks
            if chunk_idx % 5 == 0:
                monitor_memory_usage(f"Processing sequential chunk {chunk_idx+1}/{total_chunks}")
                logger.info(f"Processing chunk {chunk_idx+1}/{total_chunks} ({chunk_start}-{chunk_end})")
            
            # Process chunk
            chunk_features = _process_sequence_chunk(chunk, max_sequence_length)
            
            if chunk_features is not None and not chunk_features.empty:
                feature_chunks.append(chunk_features)
            
            # Clear memory
            del chunk
            
            # Force garbage collection periodically
            if chunk_idx % 5 == 0:
                import gc
                gc.collect()
        
        # Combine all chunks
        logger.info(f"Combining {len(feature_chunks)} feature chunks")
        
        # Process the combine operation in batches to reduce peak memory
        if len(feature_chunks) > 10:
            logger.info("Large number of chunks detected, combining in stages")
            combined_batches = []
            batch_size = 10  # Combine 10 chunks at a time
            
            for i in range(0, len(feature_chunks), batch_size):
                batch_chunks = feature_chunks[i:i+batch_size]
                batch_combined = pd.concat(batch_chunks, ignore_index=True)
                combined_batches.append(batch_combined)
                
                # Clear memory from the processed batch chunks
                for chunk in batch_chunks:
                    del chunk
                import gc
                gc.collect()
            
            combined_features = pd.concat(combined_batches, ignore_index=True)
            del combined_batches
        else:
            combined_features = pd.concat(feature_chunks, ignore_index=True)
        
        # Final memory cleanup
        del feature_chunks
        import gc
        gc.collect()
        
        # Monitor final memory usage
        final_memory = combined_features.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Final sequential features shape: {combined_features.shape}, Memory: {final_memory:.2f} MB")
        monitor_memory_usage("Completed sequential feature creation")
        
        # Export metadata for monitoring
        _export_sequential_metadata(combined_features, final_memory)
        
        return combined_features
            
    except Exception as e:
        logger.error(f"Error creating sequential features: {str(e)}")
        raise

def _process_sequence_chunk(chunk: pd.DataFrame, max_sequence_length: int = 100) -> pd.DataFrame:
    """
    Processes a chunk of sequential data with memory optimization.
    
    Args:
        chunk: DataFrame containing a chunk of VLE data
        max_sequence_length: Maximum sequence length to consider
        
    Returns:
        DataFrame containing sequential features for students in this chunk
    """
    try:
        # Monitor memory before processing
        monitor_memory_usage("Before chunk processing")
        
        # Group by student
        grouped = chunk.groupby('id_student')
        
        # Calculate sequential metrics
        sequence_metrics = []
        for student_id, student_data in grouped:
            # Limit sequence length to reduce memory usage
            if len(student_data) > max_sequence_length:
                # Take first and last portions to capture both early and late behaviors
                half_length = max_sequence_length // 2
                student_data = pd.concat([
                    student_data.iloc[:half_length],
                    student_data.iloc[-half_length:]
                ])
                student_data = student_data.sort_values('date')
            
            # Calculate metrics
            metrics = _calculate_student_metrics(student_data)
            metrics['id_student'] = student_id
            
            # Add module and presentation info
            if 'code_module' in student_data.columns and 'code_presentation' in student_data.columns:
                # Get most common values
                metrics['code_module'] = student_data['code_module'].iloc[0]
                metrics['code_presentation'] = student_data['code_presentation'].iloc[0]
            
            sequence_metrics.append(metrics)
        
        # Create DataFrame from metrics with optimized memory usage
        result_df = pd.DataFrame(sequence_metrics)
        
        # Convert appropriate columns to more efficient dtypes
        for col in result_df.columns:
            if 'count' in col or 'num' in col:
                # Count columns to integer
                result_df[col] = result_df[col].astype(np.int32)
            elif 'ratio' in col or 'avg' in col or 'regularity' in col or 'entropy' in col:
                # Ratio/float columns to float32
                result_df[col] = result_df[col].astype(np.float32)
        
        # Monitor memory after processing
        monitor_memory_usage("After chunk processing")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing sequence chunk: {str(e)}")
        raise

def _calculate_student_metrics(student_data: pd.DataFrame) -> Dict:
    """
    Calculates sequential metrics for a student with enhanced temporal context.
    Uses TEMPORAL_CONFIG settings for consistent time-based analysis.
    """
    try:
        metrics = {}
        
        # Time differences between activities (already in days)
        time_diffs = student_data['date'].diff()
        
        # General engagement timing metrics
        metrics['avg_time_between_activities'] = time_diffs.mean() if not pd.isna(time_diffs.mean()) else 0
        metrics['max_time_between_activities'] = time_diffs.max() if not pd.isna(time_diffs.max()) else 0
        metrics['activity_regularity'] = time_diffs.std() if not pd.isna(time_diffs.std()) else 0
        
        # Pre/Post module activity (using configured windows)
        pre_module = student_data[
            (student_data['date'] >= TEMPORAL_CONFIG['activity_windows']['pre_module'][0]) &
            (student_data['date'] < TEMPORAL_CONFIG['activity_windows']['pre_module'][1])
        ]
        
        # Activity phases based on configuration
        early_phase = student_data[
            (student_data['date'] >= TEMPORAL_CONFIG['activity_windows']['early_phase'][0]) &
            (student_data['date'] < TEMPORAL_CONFIG['activity_windows']['early_phase'][1])
        ]
        mid_phase = student_data[
            (student_data['date'] >= TEMPORAL_CONFIG['activity_windows']['mid_phase'][0]) &
            (student_data['date'] < TEMPORAL_CONFIG['activity_windows']['mid_phase'][1])
        ]
        late_phase = student_data[
            (student_data['date'] >= TEMPORAL_CONFIG['activity_windows']['late_phase'][0]) &
            (student_data['date'] <= TEMPORAL_CONFIG['activity_windows']['late_phase'][1])
        ]
        
        # Calculate phase metrics
        metrics.update({
            'pre_module_activity_count': len(pre_module),
            'early_phase_activity_count': len(early_phase),
            'mid_phase_activity_count': len(mid_phase),
            'late_phase_activity_count': len(late_phase),
            'early_phase_intensity': early_phase['sum_click'].mean() if len(early_phase) > 0 else 0,
            'mid_phase_intensity': mid_phase['sum_click'].mean() if len(mid_phase) > 0 else 0,
            'late_phase_intensity': late_phase['sum_click'].mean() if len(late_phase) > 0 else 0
        })
        
        # Calculate phase ratios
        total_activities = len(student_data)
        if total_activities > 0:
            metrics.update({
                'early_phase_ratio': len(early_phase) / total_activities,
                'mid_phase_ratio': len(mid_phase) / total_activities,
                'late_phase_ratio': len(late_phase) / total_activities
            })
        
        # Session analysis using configured gap threshold
        session_breaks = time_diffs > TEMPORAL_CONFIG['session_gap_days']
        metrics['num_sessions'] = session_breaks.sum() + 1
        
        if metrics['num_sessions'] > 0:
            metrics['avg_activities_per_session'] = total_activities / metrics['num_sessions']
        else:
            metrics['avg_activities_per_session'] = 0
        
        # Activity type transitions and entropy
        if len(student_data) > 1:
            # Convert categorical to string before concatenation
            activity_type_str = student_data['activity_type'].astype(str)
            transitions = activity_type_str.shift() + '_to_' + activity_type_str
            
            # Remove NaN transitions from the first row
            transitions = transitions.dropna()
            
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

@track_execution_time
def combine_sequential_features(
    sequential_features: Dict[str, pd.DataFrame],
    feature_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Combines multiple sequential feature sets with optional weighting.
    
    Args:
        sequential_features: Dictionary of sequential feature DataFrames
        feature_weights: Optional dictionary of feature weights
        
    Returns:
        Combined DataFrame of sequential features
    """
    try:
        combined_features = []
        total_memory = 0
        
        # Track progress while combining chunks
        for i, chunk in enumerate(track_progress(sequential_features.values(), desc="Combining feature chunks")):
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

def load_sequential_features(feature_path: str, format: str = 'parquet') -> pd.DataFrame:
    """
    Load sequential features from disk in either parquet or csv format.
    Wrapper around load_features for semantic clarity.
    
    Args:
        feature_path: Path to the feature file
        format: File format ('parquet' or 'csv')
        
    Returns:
        DataFrame containing the sequential features
    """
    return load_features(feature_path, format)

def save_sequential_features(features: pd.DataFrame, output_path: str, format: str = 'parquet') -> str:
    """
    Save sequential features to disk in either parquet or csv format.
    Wrapper around save_features for semantic clarity.
    
    Args:
        features: DataFrame containing the features
        output_path: Path to save the features
        format: File format ('parquet' or 'csv')
        
    Returns:
        Path to the saved file
    """
    return save_features(features, output_path, format)