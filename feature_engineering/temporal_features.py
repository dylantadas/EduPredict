import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime
from config import FEATURE_ENGINEERING, FAIRNESS, DIRS
from utils.monitoring_utils import monitor_memory_usage, track_progress

logger = logging.getLogger('edupredict')

def create_temporal_features(
    activity_data: pd.DataFrame,
    demographic_data: pd.DataFrame,
    window_sizes: List[int] = FEATURE_ENGINEERING['window_sizes']
) -> Dict[str, pd.DataFrame]:
    """
    Creates time-based engagement features with demographic fairness monitoring.
    
    Args:
        activity_data: DataFrame containing student activity data
        demographic_data: DataFrame containing demographic data
        window_sizes: List of time window sizes for feature creation

    Returns:
        Dictionary of DataFrames containing temporal features for each window size
    """
    try:
        temporal_features = {}
        
        # Monitor initial memory state
        monitor_memory_usage("Starting temporal feature creation")
        
        # Log the number of windows being processed
        logger.info(f"Creating temporal features for {len(window_sizes)} window sizes: {window_sizes}")
        
        # Log initial data shape
        logger.info(f"Activity data shape: {activity_data.shape}, Demographic data shape: {demographic_data.shape}")
        
        # Use only necessary columns from demographic data to reduce memory
        demographic_subset = demographic_data[['id_student'] + FAIRNESS['protected_attributes']].copy()
        
        # Optimize dtypes before merging
        for col in activity_data.select_dtypes(include=['int64']).columns:
            activity_data[col] = activity_data[col].astype(np.int32)
            
        for col in activity_data.select_dtypes(include=['float64']).columns:
            activity_data[col] = activity_data[col].astype(np.float32)
        
        # Merge demographic data for fairness monitoring - use smaller subset
        merged_data = activity_data.merge(
            demographic_subset,
            on='id_student',
            how='left'
        )
        logger.info(f"Merged data for temporal features: {merged_data.shape}")
        
        # Process each window size individually
        for window_size in track_progress(window_sizes, desc="Processing time windows", total=len(window_sizes)):
            monitor_memory_usage(f"Processing window size {window_size}")
            
            # Create a copy of necessary columns only to reduce memory footprint
            window_data = merged_data[['id_student', 'code_module', 'code_presentation', 
                                     'date', 'sum_click', 'id_site', 'activity_type'] + 
                                    FAIRNESS['protected_attributes']].copy()
            
            # Create window-based features with reduced memory footprint
            logger.info(f"Creating features for window size {window_size}")
            window_features = _create_window_features(window_data, window_size)
            logger.info(f"Created features for window size {window_size}: {window_features.shape}")
            
            # Monitor demographic distribution
            _monitor_demographic_distribution(
                window_features,
                window_size,
                FAIRNESS['protected_attributes']
            )
            
            # Store to results dictionary
            temporal_features[f'window_{window_size}'] = window_features
            
            # Force garbage collection after each window
            import gc
            gc.collect()
        
        monitor_memory_usage("Completed temporal feature creation")
        return temporal_features
        
    except Exception as e:
        logger.error(f"Error creating temporal features: {str(e)}")
        raise

def _create_window_features(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Creates features for a specific time window.
    The date values are already in days relative to module start.
    """
    try:
        # Create window boundaries using floor division of days
        data['window'] = np.floor(data['date'] / window_size).astype(np.int32)
        
        # Get unique student-module-window combinations for groupby
        unique_combinations = data[['id_student', 'code_module', 'code_presentation', 'window']].drop_duplicates()
        total_groups = len(unique_combinations)
        logger.debug(f"Processing {total_groups} student-module-window combinations")
        
        # Use more efficient groupby with observed=True where possible
        # Use standard pandas aggregation functions without dtype parameter
        metrics = data.groupby(['id_student', 'code_module', 'code_presentation', 'window'], observed=True).agg({
            'sum_click': ['sum', 'mean', 'std'],
            'id_site': ['nunique'],
            'activity_type': ['nunique'],
            'date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        metrics.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] else col[0]
            for col in metrics.columns
        ]
        
        # Calculate derived metrics with controlled dtypes - conversion happens after calculation
        metrics['window_span'] = (metrics['date_max'] - metrics['date_min'])
        metrics['engagement_density'] = (metrics['sum_click_sum'] / metrics['window_span'].clip(1))
        metrics['pre_module_activities'] = (metrics['date_min'] < 0).astype(np.int8)  # Use int8 for boolean flags
        
        # Rename for clarity and drop intermediate columns
        metrics = metrics.rename(columns={
            'sum_click_sum': 'total_clicks',
            'sum_click_mean': 'avg_clicks',
            'sum_click_std': 'click_std',
            'id_site_nunique': 'unique_resources',
            'activity_type_nunique': 'unique_activities'
        })
        
        # Convert columns to efficient datatypes after aggregation
        for col in metrics.select_dtypes(include=['float64']).columns:
            metrics[col] = metrics[col].astype(np.float32)
            
        for col in metrics.select_dtypes(include=['int64']).columns:
            if col != 'id_student':  # Preserve id_student as original type
                metrics[col] = metrics[col].astype(np.int32)
        
        # Delete intermediate columns to save memory
        del metrics['date_min']
        del metrics['date_max']
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error creating window features: {str(e)}")
        raise

def _monitor_demographic_distribution(
    features: pd.DataFrame,
    window_size: int,
    protected_attributes: List[str]
) -> None:
    """Monitors feature distributions across demographic groups."""
    try:
        for attr in protected_attributes:
            if attr not in features.columns:
                continue
                
            # Calculate engagement metrics by group
            group_metrics = features.groupby(attr).agg({
                'total_clicks': ['mean', 'std'],
                'avg_clicks': ['mean', 'std'],
                'unique_resources': ['mean', 'std']
            })
            
            # Calculate disparity metrics
            for metric in ['total_clicks', 'avg_clicks', 'unique_resources']:
                max_mean = group_metrics[(metric, 'mean')].max()
                min_mean = group_metrics[(metric, 'mean')].min()
                disparity = (max_mean - min_mean) / max_mean if max_mean != 0 else 0
                
                if disparity > FAIRNESS['threshold']:
                    logger.warning(
                        f"High demographic disparity detected in window {window_size} "
                        f"for {attr} in {metric}: {disparity:.2f}"
                    )
                    
                    # Log group-specific statistics
                    logger.info(
                        f"Group statistics for {metric} in window {window_size}:\n"
                        f"{group_metrics[metric]}"
                    )
                    
    except Exception as e:
        logger.error(f"Error monitoring demographic distribution: {str(e)}")
        raise

def validate_temporal_parameters(params: Dict) -> bool:
    """
    Validates temporal feature parameters.
    
    Args:
        params: Dictionary containing parameters for temporal feature generation
        
    Returns:
        bool: True if parameters are valid, False otherwise
    """
    try:
        window_sizes = params.get('window_sizes', FEATURE_ENGINEERING['window_sizes'])
        
        # Check window sizes are positive integers
        if not all(isinstance(w, int) and w > 0 for w in window_sizes):
            logger.error("Window sizes must be positive integers")
            return False
            
        # Check window sizes are in ascending order
        if sorted(window_sizes) != window_sizes:
            logger.warning("Window sizes should be in ascending order")
            
        # Check for reasonable window size ranges
        min_window = min(window_sizes)
        max_window = max(window_sizes)
        if min_window < 1 or max_window > 365:
            logger.warning(
                f"Window sizes ({min_window}, {max_window}) outside "
                "recommended range (1-365 days)"
            )
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating temporal parameters: {str(e)}")
        return False