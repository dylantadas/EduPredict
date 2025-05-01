import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json
from pathlib import Path
from config import FEATURE_ENGINEERING, FAIRNESS, PROTECTED_ATTRIBUTES, DIRS, DEMOGRAPHIC_STANDARDIZATION
from utils.monitoring_utils import monitor_memory_usage
from feature_engineering.feature_selector import NumpyJSONEncoder

logger = logging.getLogger('edupredict')

def standardize_demographic_values(
    data: pd.DataFrame,
    standardization_rules: Dict = DEMOGRAPHIC_STANDARDIZATION
) -> pd.DataFrame:
    """
    Standardizes demographic values according to defined rules.
    
    Args:
        data: DataFrame containing demographic data
        standardization_rules: Dictionary of standardization rules
        
    Returns:
        DataFrame with standardized demographic values
    """
    standardized = data.copy()
    
    try:
        for col, rule in standardization_rules.items():
            if col not in standardized.columns:
                continue
                
            if isinstance(rule, dict):
                # Apply mapping dictionary
                standardized[col] = standardized[col].map(rule).fillna(standardized[col])
            elif rule == 'lower':
                # Convert to lowercase
                standardized[col] = standardized[col].str.lower()
            
        return standardized
        
    except Exception as e:
        logger.error(f"Error standardizing demographic values: {str(e)}")
        raise

def validate_demographic_parameters(params: Dict) -> bool:
    """
    Validates demographic feature engineering parameters.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        bool: True if parameters are valid
    """
    try:
        # Check protected attributes
        if not all(attr in PROTECTED_ATTRIBUTES for attr in params.get('protected_attributes', [])):
            logger.error("Invalid protected attributes specified")
            return False
            
        # Check group size thresholds
        if params.get('min_group_size', 0) < 1:
            logger.error("Minimum group size must be positive")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating demographic parameters: {str(e)}")
        return False

@monitor_memory_usage
def create_demographic_features(
    data: pd.DataFrame,
    params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Creates demographic features with fairness considerations.
    
    Args:
        data: DataFrame containing demographic information
        params: Optional parameters for feature creation
        
    Returns:
        DataFrame with processed demographic features
    """
    try:
        logger.info("Starting demographic feature creation")
        demographic_features = data.copy()

        for col in demographic_features.columns:
            try:
                # Create protected attribute features
                if col in FAIRNESS['protected_attributes']:
                    # Create dummy variables
                    attr_dummies = pd.get_dummies(
                        demographic_features[col], 
                        prefix=col,
                        prefix_sep='_'
                    )
                    demographic_features = pd.concat([demographic_features, attr_dummies], axis=1)
                    
                    # Check group sizes
                    group_sizes = demographic_features[col].value_counts()
                    small_groups = group_sizes[group_sizes < FAIRNESS['min_group_size']]
                    if not small_groups.empty:
                        logger.warning(
                            f"Small group sizes detected in {col}: "
                            f"{small_groups.to_dict()}"
                        )
                
                # Handle education level
                if col == 'highest_education':
                    education_order = [
                        'no_formal',
                        'below_a_level',
                        'a_level',
                        'he_qualification',
                        'post_graduate'
                    ]
                    demographic_features['education_level'] = pd.Categorical(
                        demographic_features['highest_education'],
                        categories=education_order,
                        ordered=True
                    ).codes
                    
                    edu_dummies = pd.get_dummies(
                        demographic_features['highest_education'],
                        prefix='education'
                    )
                    demographic_features = pd.concat([demographic_features, edu_dummies], axis=1)
                
                # Handle numeric demographic features
                numeric_columns = ['studied_credits', 'num_of_prev_attempts']
                if col in numeric_columns:
                    # Convert to numeric safely
                    numeric_values = pd.to_numeric(demographic_features[col], errors='coerce')
                    demographic_features[col] = numeric_values
                    
                    # Only create bins if we have valid numeric values
                    if not numeric_values.isna().all():
                        try:
                            # Create bins using qcut
                            bins = pd.qcut(
                                numeric_values,
                                q=5,
                                labels=[f'{col}_q{i+1}' for i in range(5)],
                                duplicates='drop'
                            )
                            demographic_features[f'{col}_binned'] = bins
                        except ValueError as e:
                            logger.warning(f"Could not create bins for {col}: {str(e)}")
                    
                    # Calculate statistics for scaling using distinct variable names
                    valid_numeric = numeric_values.dropna()
                    if not valid_numeric.empty:
                        # Calculate statistics with unique names to avoid conflicts
                        col_mean = float(valid_numeric.mean())
                        col_std = float(valid_numeric.std())
                        
                        if col_std > 0:
                            demographic_features[f'{col}_scaled'] = (
                                (numeric_values - col_mean) / col_std
                            )
                        else:
                            logger.warning(f"Could not scale {col} due to zero standard deviation")
                            demographic_features[f'{col}_scaled'] = numeric_values

            except Exception as column_error:
                logger.error("Error processing column %s: %s", col, str(column_error))
                raise

        # Create interaction terms
        _create_interaction_terms(demographic_features)
        
        # Monitor feature distributions
        _monitor_feature_distributions(demographic_features)
        
        # Export feature metadata
        _export_demographic_metadata(demographic_features, params)
        
        return demographic_features

    except Exception as e:
        logger.error("Error in create_demographic_features: %s", str(e))
        raise

def _monitor_original_distributions(data: pd.DataFrame) -> None:
    """Monitors original demographic distributions."""
    try:
        # Log distribution of protected attributes
        for attr in FAIRNESS['protected_attributes']:
            if attr in data.columns:
                dist = data[attr].value_counts(normalize=True)
                logger.info(f"\n{attr} distribution:\n{dist}")
                
                # Check for minimum representation
                if attr in PROTECTED_ATTRIBUTES:
                    threshold = PROTECTED_ATTRIBUTES[attr]['balanced_threshold']
                    below_threshold = dist[dist < threshold]
                    if not below_threshold.empty:
                        logger.warning(
                            f"{attr} has groups below minimum representation threshold: "
                            f"{below_threshold.to_dict()}"
                        )
    except Exception as e:
        logger.error(f"Error monitoring distributions: {str(e)}")

def _create_interaction_terms(features: pd.DataFrame) -> None:
    """Creates interaction terms between relevant features."""
    try:
        # Create education-region interaction
        if all(col in features.columns for col in ['highest_education', 'region']):
            features['education_by_region'] = features.apply(
                lambda x: f"{x['highest_education']}_{x['region']}",
                axis=1
            )
        
        # Create credit load by age band
        if all(col in features.columns for col in ['studied_credits', 'age_band']):
            features['credits_by_age'] = features.apply(
                lambda x: f"credits_{x['studied_credits']}_{x['age_band']}",
                axis=1
            )
            
        # Create disability-education interaction
        if all(col in features.columns for col in ['disability', 'highest_education']):
            features['disability_by_education'] = features.apply(
                lambda x: f"{x['disability']}_{x['highest_education']}",
                axis=1
            )
            
    except Exception as e:
        logger.error(f"Error creating interaction terms: {str(e)}")

def _monitor_feature_distributions(features: pd.DataFrame) -> None:
    """Monitors distributions of created features."""
    try:
        # Log correlation between numeric features
        numeric_features = features.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 1:
            corr_matrix = features[numeric_features].corr()
            high_corr = np.where(np.abs(corr_matrix) > FEATURE_ENGINEERING['correlation_threshold'])
            high_corr_pairs = [
                (numeric_features[i], numeric_features[j], corr_matrix.iloc[i, j])
                for i, j in zip(*high_corr)
                if i < j  # Only get unique pairs
            ]
            
            if high_corr_pairs:
                logger.warning("High correlations detected between features:")
                for feat1, feat2, corr in high_corr_pairs:
                    logger.warning(f"{feat1} - {feat2}: {corr:.3f}")
                    
        # Monitor categorical feature cardinality
        cat_features = features.select_dtypes(include=['category', 'object']).columns
        for col in cat_features:
            n_unique = features[col].nunique()
            if n_unique > 50:  # High cardinality warning
                logger.warning(
                    f"High cardinality detected in {col}: {n_unique} unique values"
                )
                    
    except Exception as e:
        logger.error(f"Error monitoring feature distributions: {str(e)}")

def _export_demographic_metadata(features: pd.DataFrame, params: Dict) -> None:
    """Exports metadata about demographic features."""
    try:
        metadata = {
            'feature_counts': {
                'total': len(features.columns),
                'protected': len(FAIRNESS['protected_attributes']),
                'numeric': len(features.select_dtypes(include=[np.number]).columns),
                'categorical': len(features.select_dtypes(include=['category', 'object']).columns)
            },
            'group_sizes': {
                attr: features[attr].value_counts().to_dict()
                for attr in FAIRNESS['protected_attributes']
                if attr in features.columns
            },
            'class_balance': {
                attr: features[attr].value_counts(normalize=True).to_dict()
                for attr in FAIRNESS['protected_attributes']
                if attr in features.columns
            },
            'parameters': params,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save metadata
        output_path = DIRS['feature_metadata'] / 'demographic_features.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
            
        logger.info(f"Exported demographic feature metadata to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting feature metadata: {str(e)}")
        raise

def load_features(feature_path: str, format: str = 'parquet') -> pd.DataFrame:
    """
    Load features from disk in either parquet or csv format.
    
    Args:
        feature_path: Path to the feature file
        format: File format ('parquet' or 'csv')
        
    Returns:
        DataFrame containing the features
    """
    try:
        if format.lower() == 'parquet':
            try:
                return pd.read_parquet(feature_path)
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to load parquet file: {e}. Trying CSV format.")
                return pd.read_csv(feature_path)
        else:
            return pd.read_csv(feature_path)
    except Exception as e:
        logger.error(f"Error loading features from {feature_path}: {str(e)}")
        raise

def save_features(features: pd.DataFrame, output_path: str, format: str = 'parquet') -> str:
    """
    Save features to disk in either parquet or csv format.
    
    Args:
        features: DataFrame containing the features
        output_path: Path to save the features
        format: File format ('parquet' or 'csv')
        
    Returns:
        Path to the saved file
    """
    try:
        if format.lower() == 'parquet':
            try:
                features.to_parquet(output_path)
                return output_path
            except ImportError as e:
                logger.warning(f"Failed to save as parquet: {e}. Falling back to CSV format.")
                csv_path = output_path.replace('.parquet', '.csv')
                features.to_csv(csv_path, index=False)
                return csv_path
        else:
            features.to_csv(output_path, index=False)
            return output_path
    except Exception as e:
        logger.error(f"Error saving features to {output_path}: {str(e)}")
        raise