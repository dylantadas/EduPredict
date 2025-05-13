import pandas as pd
import numpy as np
import gc  # Add gc import
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
        Generate temporal features from the data with optimized memory usage.
        Properly handles pre-module dates (negative values) and post-module dates.
        """
        try:
            # Sort data by time while preserving negative dates
            data = data.sort_values([time_column])
            
            # Initialize results DataFrame with index only
            result_features = pd.DataFrame(index=data.index)
            
            # Group data if group columns specified
            if group_columns:
                groups = data.groupby(group_columns, observed=True)  # Add observed=True for categorical columns
                total_groups = len(groups)
            else:
                groups = [(None, data)]
                total_groups = 1
                
            total_steps = len(target_columns) * len(self.time_windows) * total_groups
            logger.info(f"Generating temporal features: {total_steps} total operations")
            
            # Process features in chunks for each target column
            for target_col in track_progress(target_columns, desc="Processing columns"):
                window_features = {}
                
                # Create rolling window features in chunks
                chunk_size = min(100000, len(data))  # Limit chunk size
                n_chunks = (len(data) + chunk_size - 1) // chunk_size
                
                for chunk_idx in track_progress(range(n_chunks), desc=f"Processing chunks for {target_col}"):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, len(data))
                    chunk_data = data.iloc[start_idx:end_idx].copy()
                    
                    # Calculate features for this chunk
                    for window in self.time_windows:
                        for group_key, group_data in groups:
                            group_mask = None if group_key is None else chunk_data.index.isin(group_data.index)
                            chunk_group_data = chunk_data if group_key is None else chunk_data[group_mask]
                            
                            if len(chunk_group_data) == 0:
                                continue
                            
                            # Handle pre-module activity separately
                            pre_module_mask = chunk_group_data[time_column] < 0
                            if pre_module_mask.any():
                                pre_module_data = chunk_group_data[pre_module_mask]
                                feature_prefix = f"{target_col}_premodule"
                                if group_key is not None:
                                    feature_prefix = f"{feature_prefix}_{group_key}"
                                
                                window_features.update({
                                    f"{feature_prefix}_total": pre_module_data[target_col].sum(),
                                    f"{feature_prefix}_mean": pre_module_data[target_col].mean(),
                                    f"{feature_prefix}_count": len(pre_module_data)
                                })
                                
                                self.feature_names_.extend([
                                    f"{feature_prefix}_{stat}" 
                                    for stat in ['total', 'mean', 'count']
                                ])
                            
                            # Regular rolling window features for all data
                            rolling = chunk_group_data[target_col].rolling(
                                window=window,
                                min_periods=1
                            )
                            
                            feature_prefix = f"{target_col}_{window}d"
                            if group_key is not None:
                                feature_prefix = f"{feature_prefix}_{group_key}"
                            
                            # Calculate all aggregations at once
                            aggs = rolling.agg(['mean', 'std', 'max', 'min'])
                            for func in ['mean', 'std', 'max', 'min']:
                                window_features[f"{feature_prefix}_{func}"] = aggs[func]
                            
                            self.feature_names_.extend([
                                f"{feature_prefix}_{func}" 
                                for func in ['mean', 'std', 'max', 'min']
                            ])
                    
                    # Batch add window features for this chunk
                    if window_features:
                        chunk_df = pd.DataFrame(window_features)
                        chunk_df = chunk_df.ffill().fillna(0)
                        result_features.loc[chunk_df.index] = chunk_df
                        
                        # Clear chunk data
                        del chunk_df
                        window_features.clear()
                        gc.collect()
                
                # Create lag features if enabled
                if self.enable_lag_features:
                    lag_features = {}
                    
                    for lag in track_progress(range(1, self.max_lag + 1), desc="Creating lags"):
                        for chunk_idx in range(n_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = min((chunk_idx + 1) * chunk_size, len(data))
                            chunk_data = data.iloc[start_idx:end_idx].copy()
                            
                            for group_key, group_data in groups:
                                group_mask = None if group_key is None else chunk_data.index.isin(group_data.index)
                                chunk_group_data = chunk_data if group_key is None else chunk_data[group_mask]
                                
                                if len(chunk_group_data) == 0:
                                    continue
                                    
                                feature_name = f"{target_col}_lag_{lag}"
                                if group_key is not None:
                                    feature_name = f"{feature_name}_{group_key}"
                                
                                # Handle lags across module boundary (negative to positive dates)
                                series = chunk_group_data[target_col].copy()
                                lag_features[feature_name] = series.shift(lag)
                                self.feature_names_.append(feature_name)
                        
                        # Batch add lag features
                        if lag_features:
                            chunk_df = pd.DataFrame(lag_features)
                            chunk_df = chunk_df.fillna(0)
                            result_features.loc[chunk_df.index] = chunk_df
                            del chunk_df
                            lag_features.clear()
                            gc.collect()
            
            # Create defragmented copy before returning
            result_features = result_features.copy()
            
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
            groups = data.groupby(group_column, observed=False)
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

def calculate_temporal_features(df, window_sizes=[7, 14, 30], features_to_track=None):
    """Calculate temporal features with pre-allocated DataFrames and reduced memory fragmentation"""
    if features_to_track is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        features_to_track = [col for col in numeric_cols if col not in ['student_id', 'timestamp']]
    
    # Pre-allocate the final DataFrame
    result_features = pd.DataFrame(index=df.index)
    
    for window in window_sizes:
        # Process features in chunks to reduce memory usage
        chunk_size = 5  # Process 5 features at a time
        feature_chunks = [features_to_track[i:i + chunk_size] for i in range(0, len(features_to_track), chunk_size)]
        
        for feature_chunk in feature_chunks:
            chunk_features = {}
            for feature in feature_chunk:
                try:
                    # Create a copy of the series to prevent fragmentation
                    series = df[feature].copy()
                    rolling = series.rolling(window=window, min_periods=1)
                    
                    # Calculate all statistics at once
                    chunk_features.update({
                        f"{feature}_mean_{window}d": rolling.mean(),
                        f"{feature}_std_{window}d": rolling.std(),
                        f"{feature}_max_{window}d": rolling.max(),
                        f"{feature}_min_{window}d": rolling.min()
                    })
                except Exception as e:
                    logger.warning(f"Error calculating features for {feature}: {str(e)}")
                    continue
            
            # Add chunk features to result
            if chunk_features:
                result_features = pd.concat([result_features, pd.DataFrame(chunk_features)], axis=1)
                
            # Force garbage collection after each chunk
            gc.collect()
    
    return result_features

def create_temporal_features(
    vle_data: pd.DataFrame,
    student_ids: Dict[str, List[str]],
    window_sizes: Optional[List[int]] = None,
    module_info: Optional[pd.DataFrame] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create temporal features for each data split.
    """
    try:
        logger = logger or logging.getLogger('edupredict')
        window_sizes = window_sizes or FEATURE_ENGINEERING['window_sizes']
        
        # Initialize temporal processor
        processor = TemporalFeatureProcessor(
            time_windows=window_sizes,
            enable_lag_features=TEMPORAL_CONFIG.get('enable_lag_features', True),
            max_lag=TEMPORAL_CONFIG.get('max_lag', 3)
        )
        
        # Process each split with progress tracking
        results = {}
        
        for split_name, split_ids in student_ids.items():
            logger.info(f"Creating temporal features for {split_name} split...")
            
            # Filter data for this split
            split_data = vle_data[vle_data['id_student'].isin(split_ids)].copy()
            if len(split_data) == 0:
                logger.warning(f"No VLE data found for {split_name} split!")
                continue
                
            logger.info(f"Processing {len(split_data):,} interactions for {split_name}")
            
            # Add module context if available
            if module_info is not None:
                split_data = pd.merge(
                    split_data,
                    module_info[['id_student', 'date_registration']],
                    on='id_student',
                    how='left'
                )
                split_data['time_since_registration'] = split_data['date'] - split_data['date_registration']
            
            # Process features in chunks
            chunk_size = min(100000, len(split_data))
            chunks = [split_data[i:i + chunk_size] for i in range(0, len(split_data), chunk_size)]
            
            split_features = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{len(chunks)} for {split_name}")
                
                # Generate temporal features for this chunk
                chunk_features = processor.fit_transform(
                    data=chunk,
                    time_column='date',
                    target_columns=['sum_click'],
                    group_columns=['id_student', 'code_module']
                )
                
                split_features.append(chunk_features)
                gc.collect()  # Force garbage collection after each chunk
            
            # Combine chunks
            results[split_name] = pd.concat(split_features, axis=0)
            logger.info(f"Created {len(results[split_name].columns)} features for {split_name}")
            
            # Clear memory
            del split_features
            gc.collect()
        
        return results
        
    except Exception as e:
        logger.error(f"Error creating temporal features: {str(e)}")
        raise