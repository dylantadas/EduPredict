import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime
from config import FEATURE_ENGINEERING, FAIRNESS, DIRS
from utils.monitoring_utils import monitor_memory_usage

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
        
        # Merge demographic data for fairness monitoring
        merged_data = activity_data.merge(
            demographic_data[['id_student'] + FAIRNESS['protected_attributes']],
            on='id_student',
            how='left'
        )
        
        for window_size in window_sizes:
            monitor_memory_usage(f"Processing window size {window_size}")
            
            # Create window-based features
            window_features = _create_window_features(merged_data, window_size)
            
            # Monitor demographic distribution
            _monitor_demographic_distribution(
                window_features,
                window_size,
                FAIRNESS['protected_attributes']
            )
            
            temporal_features[f'window_{window_size}'] = window_features
            
        monitor_memory_usage("Completed temporal feature creation")
        return temporal_features
        
    except Exception as e:
        logger.error(f"Error creating temporal features: {str(e)}")
        raise

def _create_window_features(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Creates features for a specific time window.
    Handles both negative and positive dates relative to module start,
    ensuring consistent interpretation across the timeline.
    """
    try:
        # Create normalized timeline for better interpretation
        # Negative dates are before module start, 0 is module start
        data['absolute_day'] = data['date'].abs()
        data['is_before_start'] = data['date'] < 0
        
        # Group by student and time window using floor division
        # This ensures consistent window boundaries for both negative and positive dates
        data['window'] = np.floor(data['date'] / window_size).astype(int)
        
        # Calculate engagement metrics with timeline context
        metrics = data.groupby(['id_student', 'code_module', 'code_presentation', 'window']).agg({
            'sum_click': ['sum', 'mean', 'std'],
            'id_site': 'nunique',
            'activity_type': 'nunique',
            'is_before_start': 'sum',  # Count activities before module start
            'absolute_day': ['min', 'max']  # Track timeline span
        }).reset_index()
        
        # Flatten column names
        metrics.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in metrics.columns
        ]
        
        # Calculate additional temporal metrics
        metrics['window_span'] = metrics['absolute_day_max'] - metrics['absolute_day_min']
        metrics['engagement_density'] = metrics['sum_click_sum'] / metrics['window_span'].clip(1)
        
        # Rename for clarity and drop intermediate columns
        metrics = metrics.rename(columns={
            'sum_click_sum': 'total_clicks',
            'sum_click_mean': 'avg_clicks',
            'sum_click_std': 'click_std',
            'id_site_nunique': 'unique_resources',
            'activity_type_nunique': 'unique_activities',
            'is_before_start_sum': 'pre_module_activities'
        }).drop(columns=['absolute_day_min', 'absolute_day_max'])
        
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