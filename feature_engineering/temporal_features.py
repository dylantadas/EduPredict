import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from config import FEATURE_ENGINEERING, DIRS, TEMPORAL_CONFIG
from utils.monitoring_utils import monitor_memory_usage, track_progress
from feature_engineering.feature_selector import NumpyJSONEncoder

logger = logging.getLogger('edupredict')

class TemporalFeatureProcessor:
    """
    Processes and generates temporal features from time-series data.
    """
    
    def __init__(
        self,
        time_windows: List[int],
        aggregation_functions: List[str] = ['mean', 'std', 'max', 'min'],
        enable_lag_features: bool = True,
        max_lag: int = 3
    ):
        """
        Initialize temporal feature processor.
        
        Args:
            time_windows: List of time windows (in days) for rolling features
            aggregation_functions: List of aggregation functions to apply
            enable_lag_features: Whether to create lagged features
            max_lag: Maximum number of lags to create
        """
        self.time_windows = time_windows
        self.aggregation_functions = aggregation_functions
        self.enable_lag_features = enable_lag_features
        self.max_lag = max_lag
        self.activity_windows = TEMPORAL_CONFIG['activity_windows']
        
        self.feature_names_: List[str] = []
        self.temporal_statistics_: Dict = {}
        
    @monitor_memory_usage
    def fit_transform(
        self,
        data: pd.DataFrame,
        time_column: str,
        target_columns: List[str],
        group_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate temporal features from the data.
        
        Args:
            data: Input DataFrame
            time_column: Name of the timestamp column
            target_columns: Columns to generate features for
            group_columns: Columns to group by (e.g., student_id)
            
        Returns:
            DataFrame with temporal features
        """
        try:
            # Sort data by time
            data = data.sort_values(time_column)
            
            # Initialize results DataFrame
            result_features = pd.DataFrame(index=data.index)
            
            # Group data if group columns specified
            if group_columns:
                groups = data.groupby(group_columns)
            else:
                groups = [(None, data)]
            
            logger.info(f"Generating temporal features for {len(target_columns)} columns")
            
            for target_col in track_progress(target_columns):
                # Create rolling window features
                for window in self.time_windows:
                    for group_key, group_data in groups:
                        # Calculate rolling statistics
                        rolling = group_data[target_col].rolling(
                            window=window,
                            min_periods=1
                        )
                        
                        # Apply aggregation functions
                        for func in self.aggregation_functions:
                            feature_name = f"{target_col}_{func}_{window}d"
                            if group_key is not None:
                                feature_name = f"{feature_name}_{group_key}"
                            
                            if func == 'mean':
                                result_features[feature_name] = rolling.mean()
                            elif func == 'std':
                                result_features[feature_name] = rolling.std()
                            elif func == 'max':
                                result_features[feature_name] = rolling.max()
                            elif func == 'min':
                                result_features[feature_name] = rolling.min()
                            
                            self.feature_names_.append(feature_name)
                
                # Create lag features if enabled
                if self.enable_lag_features:
                    for lag in range(1, self.max_lag + 1):
                        for group_key, group_data in groups:
                            feature_name = f"{target_col}_lag_{lag}"
                            if group_key is not None:
                                feature_name = f"{feature_name}_{group_key}"
                                
                            result_features[feature_name] = group_data[target_col].shift(lag)
                            self.feature_names_.append(feature_name)
                            
            # Add activity window features
            for window_name, (start, end) in self.activity_windows.items():
                mask = (data[time_column] >= start) & (data[time_column] < end)
                for target_col in target_columns:
                    feature_name = f"{target_col}_{window_name}"
                    result_features[feature_name] = data[mask][target_col].agg(self.aggregation_functions)
                    self.feature_names_.append(feature_name)
                    
            # Calculate temporal statistics
            self._calculate_temporal_statistics(result_features)
            
            return result_features
            
        except Exception as e:
            logger.error(f"Error generating temporal features: {str(e)}")
            raise
            
    def _calculate_temporal_statistics(self, features: pd.DataFrame) -> None:
        """
        Calculate and store statistics about temporal features.
        """
        try:
            stats = {}
            
            for col in features.columns:
                col_stats = features[col].describe()
                stats[col] = {
                    'mean': col_stats['mean'],
                    'std': col_stats['std'],
                    'min': col_stats['min'],
                    'max': col_stats['max'],
                    'missing_pct': features[col].isnull().mean() * 100
                }
                
            self.temporal_statistics_ = stats
            
        except Exception as e:
            logger.error(f"Error calculating temporal statistics: {str(e)}")
            raise
            
    def export_feature_metadata(self, output_dir: Union[str, Path]) -> None:
        """
        Export metadata about temporal features.
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                'time_windows': self.time_windows,
                'aggregation_functions': self.aggregation_functions,
                'feature_names': self.feature_names_,
                'temporal_statistics': self.temporal_statistics_,
                'timestamp': datetime.now().isoformat()
            }
            
            output_path = output_dir / 'temporal_features_metadata.json'
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
                
            logger.info(f"Exported temporal feature metadata to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting temporal metadata: {str(e)}")
            raise

def calculate_time_based_statistics(
    data: pd.DataFrame,
    time_column: str,
    value_column: str,
    group_column: Optional[str] = None,
    resample_freq: str = 'D'
) -> pd.DataFrame:
    """
    Calculate time-based statistics for a given column.
    
    Args:
        data: Input DataFrame
        time_column: Name of timestamp column
        value_column: Name of column to calculate statistics for
        group_column: Optional column to group by
        resample_freq: Frequency for resampling ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
        DataFrame with time-based statistics
    """
    try:
        # Set timestamp as index
        data = data.set_index(time_column)
        
        if group_column:
            groups = data.groupby(group_column)
        else:
            groups = [(None, data)]
            
        stats_dfs = []
        
        for group_key, group_data in groups:
            # Resample and calculate statistics
            resampled = group_data[value_column].resample(resample_freq)
            
            stats = pd.DataFrame({
                'count': resampled.count(),
                'mean': resampled.mean(),
                'std': resampled.std(),
                'min': resampled.min(),
                'max': resampled.max()
            })
            
            if group_key is not None:
                stats['group'] = group_key
                
            stats_dfs.append(stats)
            
        # Combine all statistics
        combined_stats = pd.concat(stats_dfs)
        
        return combined_stats
        
    except Exception as e:
        logger.error(f"Error calculating time-based statistics: {str(e)}")
        raise

def create_temporal_features(
    data: pd.DataFrame,
    student_ids: Dict[str, List[str]],
    window_sizes: Optional[List[int]] = None,
    module_info: Optional[pd.DataFrame] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create temporal features for each data split.
    
    Args:
        data: DataFrame with VLE interaction data
        student_ids: Dictionary mapping split names to lists of student IDs
        window_sizes: List of window sizes for temporal features
        module_info: Optional DataFrame with module registration info
        logger: Logger instance
        
    Returns:
        Dictionary containing temporal features for each split
    """
    try:
        if logger is None:
            logger = logging.getLogger('edupredict')
            
        if window_sizes is None:
            window_sizes = FEATURE_ENGINEERING['window_sizes']
            
        # Initialize temporal processor
        processor = TemporalFeatureProcessor(
            time_windows=window_sizes,
            enable_lag_features=TEMPORAL_CONFIG.get('enable_lag_features', True),
            max_lag=TEMPORAL_CONFIG.get('max_lag', 3)
        )
        
        # Process each split
        results = {}
        for split_name, split_ids in student_ids.items():
            logger.info(f"Creating temporal features for {split_name} split...")
            
            # Filter data for this split
            split_data = data[data['id_student'].isin(split_ids)].copy()
            
            # Add module context if available
            if module_info is not None:
                split_data = pd.merge(
                    split_data,
                    module_info[['id_student', 'date_registration']],
                    on='id_student',
                    how='left'
                )
                split_data['time_since_registration'] = split_data['date'] - split_data['date_registration']
            
            # Generate temporal features
            temporal_features = processor.fit_transform(
                data=split_data,
                time_column='date',
                target_columns=['sum_click', 'time_since_registration'] if module_info is not None else ['sum_click'],
                group_columns=['id_student', 'code_module']
            )
            
            results[split_name] = temporal_features
            logger.info(f"Created {len(temporal_features.columns)} temporal features for {split_name}")
            
        return results
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating temporal features: {str(e)}")
        raise