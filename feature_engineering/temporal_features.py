import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime
from config import FEATURE_ENGINEERING, FAIRNESS, DIRS

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
        
        # Merge demographic data for fairness monitoring
        merged_data = activity_data.merge(
            demographic_data[['id_student'] + FAIRNESS['protected_attributes']],
            on='id_student',
            how='left'
        )
        
        for window_size in window_sizes:
            # Create window-based features
            window_features = _create_window_features(merged_data, window_size)
            
            # Monitor demographic distribution
            _monitor_demographic_distribution(
                window_features,
                window_size,
                FAIRNESS['protected_attributes']
            )
            
            temporal_features[f'window_{window_size}'] = window_features
            
        return temporal_features
        
    except Exception as e:
        logger.error(f"Error creating temporal features: {str(e)}")
        raise

def _create_window_features(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Creates features for a specific time window."""
    try:
        # Group by student and time window
        data['window'] = data['date'] // window_size
        
        # Calculate engagement metrics
        metrics = data.groupby(['id_student', 'code_module', 'code_presentation', 'window']).agg({
            'click_count': ['sum', 'mean', 'std'],
            'duration': ['sum', 'mean', 'std'],
            'unique_materials': 'nunique'
        }).reset_index()
        
        # Flatten column names
        metrics.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in metrics.columns
        ]
        
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
                'click_count_sum': ['mean', 'std'],
                'duration_sum': ['mean', 'std']
            })
            
            # Calculate disparity metrics
            for metric in ['click_count_sum', 'duration_sum']:
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