import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Set
import logging
from pathlib import Path
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from config import FEATURE_ENGINEERING, FAIRNESS, DIRS
from utils.monitoring_utils import monitor_memory_usage
from json import JSONEncoder
from scipy import stats

logger = logging.getLogger('edupredict')


class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class DemographicFeatureProcessor:
    """
    Processes demographic features with fairness considerations.
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        standardize: bool = True,
        handle_missing: str = 'mean'
    ):
        self.protected_attributes = protected_attributes
        self.standardize = standardize
        self.handle_missing = handle_missing
        
        self.scalers_: Dict[str, StandardScaler] = {}
        self.feature_statistics_: Dict = {}
        self.imputation_values_: Dict = {}
        self.categorical_mappings_: Dict = {}

    def _standardize_numeric_features(
        self,
        data: pd.DataFrame,
        categorical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Standardize numeric features while preserving target and specified columns.
        """
        try:
            result = data.copy()
            
            # Define columns to exclude from standardization
            excluded_cols = set(categorical_columns)
            excluded_cols.update(self.protected_attributes)
            
            # Add target column to excluded columns if present
            target_col = FEATURE_ENGINEERING['target_encoding']['column']
            if target_col in result.columns:
                excluded_cols.add(target_col)
            
            # Only standardize numeric columns that aren't excluded
            numeric_columns = result.select_dtypes(include=['int64', 'float64']).columns
            numeric_columns = [col for col in numeric_columns if col not in excluded_cols]
            
            # Create and apply standardization for each numeric column
            for col in numeric_columns:
                scaler = StandardScaler()
                result[col] = scaler.fit_transform(result[[col]])
                self.scalers_[col] = scaler
                
            return result
            
        except Exception as e:
            logger.error(f"Error standardizing numeric features: {str(e)}")
            raise

    def _encode_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode target variable using configured mapping.
        """
        try:
            result = data.copy()
            target_config = FEATURE_ENGINEERING['target_encoding']
            target_col = target_config['column']
            
            if target_col in result.columns:
                encoding_map = target_config['encoding']
                result[target_col] = result[target_col].str.lower().map(encoding_map)
                logger.info(f"Encoded target variable {target_col} using mapping: {encoding_map}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error encoding target variable: {str(e)}")
            raise

    @monitor_memory_usage
    def fit_transform(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Process demographic features with fairness considerations.
        """
        try:
            result = data.copy()
            
            if categorical_columns is None:
                categorical_columns = []

            # First encode target to prevent it from being processed
            result = self._encode_target(result)
            
            # Handle missing values before any transformations
            self._handle_missing_values(result, categorical_columns)
            
            # Standardize numeric features before categorical processing
            if self.standardize:
                result = self._standardize_numeric_features(result, categorical_columns)
            
            # Process categorical features last to preserve encodings
            result = self._process_categorical_features(result, categorical_columns)
            
            # Calculate statistics after all transformations
            self._calculate_feature_statistics(result)
            
            # Check fairness metrics on final features
            self._check_fairness_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing demographic features: {str(e)}")
            raise
            
    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        categorical_columns: List[str]
    ) -> None:
        """
        Handle missing values in the data.
        """
        try:
            for col in data.columns:
                if data[col].isnull().any():
                    if col in categorical_columns:
                        # For categorical columns, use mode
                        mode_value = data[col].mode()
                        if not mode_value.empty:
                            fill_value = mode_value.iloc[0]
                        else:
                            fill_value = 'unknown'
                    else:
                        # For numeric columns, use specified strategy
                        if self.handle_missing == 'mean':
                            fill_value = data[col].mean()
                            if pd.isna(fill_value):  # If mean is NaN
                                fill_value = 0
                        elif self.handle_missing == 'median':
                            fill_value = data[col].median()
                            if pd.isna(fill_value):  # If median is NaN
                                fill_value = 0
                        else:
                            mode_value = data[col].mode()
                            if not mode_value.empty:
                                fill_value = mode_value.iloc[0]
                            else:
                                fill_value = 0
                            
                    data[col].fillna(fill_value, inplace=True)
                    self.imputation_values_[col] = fill_value
                    
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise RuntimeError(f"Error handling missing values: {str(e)}")
            
    def _process_categorical_features(
        self,
        data: pd.DataFrame,
        categorical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Process categorical features with fairness considerations.
        """
        try:
            result = data.copy()
            
            for col in categorical_columns:
                if col in self.protected_attributes:
                    # For protected attributes, use binary encoding if possible
                    unique_values = result[col].unique()
                    if len(unique_values) == 2:
                        mapping = {val: idx for idx, val in enumerate(unique_values)}
                        result[col] = result[col].map(mapping)
                        self.categorical_mappings_[col] = mapping
                else:
                    # For other categorical features, use one-hot encoding
                    dummies = pd.get_dummies(result[col], prefix=col)
                    result = pd.concat([result, dummies], axis=1)
                    result.drop(col, axis=1, inplace=True)
                    
            return result
            
        except Exception as e:
            logger.error(f"Error processing categorical features: {str(e)}")
            raise
            
    def _calculate_feature_statistics(self, data: pd.DataFrame) -> None:
        """
        Calculate and store feature statistics.
        """
        try:
            stats = {}
            
            for col in data.columns:
                col_stats = {}
                series = data[col]
                
                # Calculate missing value percentage for all columns
                missing_pct = series.isnull().mean() * 100
                col_stats['missing_pct'] = missing_pct
                
                # For numeric columns, calculate numeric statistics
                if pd.api.types.is_numeric_dtype(series) and not series.empty:
                    try:
                        desc = series.describe()
                        col_stats.update({
                            'mean': float(desc['mean']) if not pd.isna(desc['mean']) else 0.0,
                            'std': float(desc['std']) if not pd.isna(desc['std']) else 0.0,
                            'min': float(desc['min']) if not pd.isna(desc['min']) else 0.0,
                            'max': float(desc['max']) if not pd.isna(desc['max']) else 0.0
                        })
                    except Exception as e:
                        logger.warning(f"Could not calculate numeric stats for {col}: {str(e)}")
                        col_stats.update({
                            'mean': 0.0,
                            'std': 0.0,
                            'min': 0.0,
                            'max': 0.0
                        })
                # For categorical columns, calculate distribution statistics
                else:
                    value_counts = series.value_counts(normalize=True)
                    col_stats.update({
                        'categories': list(value_counts.index),
                        'frequencies': value_counts.to_dict(),
                        'n_categories': len(value_counts)
                    })
                
                stats[col] = col_stats
                
            self.feature_statistics_ = stats
            
        except Exception as e:
            logger.error(f"Error calculating feature statistics: {str(e)}")
            raise
            
    def _check_fairness_metrics(self, data: pd.DataFrame) -> None:
        """
        Check fairness metrics for protected attributes.
        """
        try:
            for attr in self.protected_attributes:
                if attr in data.columns:
                    # For numeric/encoded columns, use non-zero threshold
                    if pd.api.types.is_numeric_dtype(data[attr]):
                        value_counts = (data[attr] > 0).value_counts(normalize=True)
                    else:
                        value_counts = data[attr].value_counts(normalize=True)

                    if len(value_counts) >= 2:  # Need at least 2 groups
                        max_ratio = value_counts.max()
                        if max_ratio > FAIRNESS.get('max_group_ratio', 0.75):
                            logger.warning(
                                f"Potential bias detected in {attr}: "
                                f"Dominant group represents {max_ratio*100:.1f}% "
                                "of the data"
                            )
                        
        except Exception as e:
            logger.error(f"Error checking fairness metrics: {str(e)}")
            raise
            
    def export_feature_metadata(self, output_dir: Union[str, Path]) -> None:
        """
        Export metadata about demographic features.
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                'protected_attributes': self.protected_attributes,
                'feature_statistics': self.feature_statistics_,
                'imputation_values': self.imputation_values_,
                'categorical_mappings': self.categorical_mappings_,
                'timestamp': datetime.now().isoformat()
            }
            
            output_path = output_dir / 'demographic_features_metadata.json'
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
                
            logger.info(f"Exported demographic feature metadata to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting demographic metadata: {str(e)}")
            raise

def analyze_demographic_distributions(
    data: pd.DataFrame,
    demographic_columns: List[str]
) -> Dict[str, Dict]:
    """
    Analyze distributions of demographic features.
    
    Args:
        data: Input DataFrame
        demographic_columns: List of demographic columns to analyze
        
    Returns:
        Dictionary with distribution statistics for each demographic feature
    """
    try:
        distributions = {}
        
        for col in demographic_columns:
            if col in data.columns:
                value_counts = data[col].value_counts(normalize=True)
                
                distributions[col] = {
                    'distribution': value_counts.to_dict(),
                    'entropy': stats.entropy(value_counts),
                    'unique_values': len(value_counts),
                    'most_common': value_counts.index[0],
                    'most_common_pct': value_counts.iloc[0] * 100
                }
                
        return distributions
        
    except Exception as e:
        logger.error(f"Error analyzing demographic distributions: {str(e)}")
        raise

def create_demographic_features(
    data: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    one_hot: bool = True,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Create demographic features from input data.
    """
    try:
        if logger is None:
            logger = logging.getLogger('edupredict')
            
        # Initialize feature processor
        processor = DemographicFeatureProcessor(
            protected_attributes=list(FAIRNESS['protected_attributes']),
            standardize=FEATURE_ENGINEERING['standardize_numeric'],
            handle_missing=FEATURE_ENGINEERING['missing_value_strategy']
        )
        
        # Set default categorical columns if not provided
        if categorical_cols is None:
            categorical_cols = FEATURE_ENGINEERING['categorical_cols']
        
        # Process features
        processed_features = processor.fit_transform(
            data=data,
            categorical_columns=categorical_cols
        )
        
        # Ensure we preserve essential columns
        essential_cols = ['id_student', 'final_result', 'split']
        for col in essential_cols:
            if col in data.columns and col not in processed_features.columns:
                processed_features[col] = data[col]
        
        # Split features by split column if it exists
        if 'split' in processed_features.columns:
            split_features = {}
            for split_name in ['train', 'validation', 'test']:
                split_mask = processed_features['split'] == split_name
                split_features[split_name] = processed_features[split_mask].copy()
                logger.info(f"{split_name} set: {len(split_features[split_name])} samples with {len(split_features[split_name].columns)} features")
            processed_features = split_features
            
        # Export metadata if output directory is configured
        if DIRS.get('feature_data'):
            processor.export_feature_metadata(DIRS['feature_data'])
            
        return processed_features
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating demographic features: {str(e)}")
        raise

def save_features(features: pd.DataFrame, output_path: Union[str, Path], logger: Optional[logging.Logger] = None) -> None:
    """
    Save features to a parquet file.
    
    Args:
        features: DataFrame of features to save
        output_path: Path to save features to
        logger: Optional logger instance
    """
    try:
        if logger is None:
            logger = logging.getLogger('edupredict')
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        features.to_parquet(output_path)
        logger.info(f"Saved features to {output_path}")
        
    except Exception as e:
        if logger:
            logger.error(f"Error saving features: {str(e)}")
        raise

def load_features(input_path: Union[str, Path], logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load features from a parquet file.
    
    Args:
        input_path: Path to load features from
        logger: Optional logger instance
        
    Returns:
        DataFrame containing loaded features
    """
    try:
        if logger is None:
            logger = logging.getLogger('edupredict')
            
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Feature file not found: {input_path}")
            
        features = pd.read_parquet(input_path)
        logger.info(f"Loaded features from {input_path}")
        
        return features
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading features: {str(e)}")
        raise