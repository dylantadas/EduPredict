import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path
from config import FEATURE_ENGINEERING, FAIRNESS, PROTECTED_ATTRIBUTES, DIRS

logger = logging.getLogger('edupredict')

def validate_demographic_parameters(params: Dict) -> bool:
    """
    Validates demographic feature engineering parameters.
    
    Args:
        params: Dictionary containing demographic feature parameters
        
    Returns:
        bool: True if parameters are valid, False otherwise
    """
    try:
        # Validate encoding parameters
        encoding_method = params.get('encoding_method', 'both')
        if encoding_method not in ['label', 'onehot', 'both']:
            logger.error(f"Invalid encoding method: {encoding_method}")
            return False
            
        # Validate minimum group size
        min_group_size = params.get('min_group_size', FAIRNESS['min_group_size'])
        if min_group_size < 1:
            logger.error("Minimum group size must be positive")
            return False
            
        # Validate feature creation flags
        if not isinstance(params.get('create_interaction_terms', True), bool):
            logger.error("create_interaction_terms must be boolean")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating demographic parameters: {str(e)}")
        return False

def create_demographic_features(
    demographic_data: pd.DataFrame,
    params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Creates demographic features with fairness monitoring.
    
    Args:
        demographic_data: DataFrame containing demographic data
        params: Dictionary containing demographic feature parameters
    
    Returns:
        pd.DataFrame: DataFrame containing the created demographic features
    """
    try:
        params = params or {}
        if not validate_demographic_parameters(params):
            raise ValueError("Invalid demographic parameters")
            
        features = demographic_data.copy()
        
        # Track original distributions for fairness monitoring
        _monitor_original_distributions(features)
        
        # Create encoded features
        categorical_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band']
        
        encoding_method = params.get('encoding_method', 'both')
        if encoding_method in ['label', 'both']:
            # Label encoding
            for col in categorical_cols:
                if col in features.columns:
                    features[f"{col}_encoded"] = pd.Categorical(features[col]).codes
                    
        if encoding_method in ['onehot', 'both']:
            # One-hot encoding
            for col in categorical_cols:
                if col in features.columns:
                    one_hot = pd.get_dummies(
                        features[col],
                        prefix=col,
                        drop_first=True
                    )
                    features = pd.concat([features, one_hot], axis=1)
                    
        # Create educational background features
        if 'num_of_prev_attempts' in features.columns:
            features['is_first_attempt'] = (features['num_of_prev_attempts'] == 0)
            features['credit_density'] = (
                features['studied_credits'] / 
                features['num_of_prev_attempts'].clip(1)
            )
            
        # Create interaction terms if specified
        if params.get('create_interaction_terms', True):
            _create_interaction_terms(features)
            
        # Monitor feature distributions
        _monitor_feature_distributions(features)
        
        # Export feature metadata
        _export_demographic_metadata(features, params)
        
        return features
        
    except Exception as e:
        logger.error(f"Error creating demographic features: {str(e)}")
        raise

def _monitor_original_distributions(data: pd.DataFrame) -> None:
    """Monitors original demographic distributions."""
    try:
        for attr in FAIRNESS['protected_attributes']:
            if attr not in data.columns:
                continue
                
            # Calculate group sizes
            group_sizes = data[attr].value_counts()
            min_size = group_sizes.min()
            
            # Check minimum group size
            if min_size < FAIRNESS['min_group_size']:
                logger.warning(
                    f"Small group size detected for {attr}. "
                    f"Minimum size: {min_size}"
                )
                
            # Calculate group ratios
            max_ratio = group_sizes.max() / min_size
            if max_ratio > FAIRNESS.get('max_ratio', 3.0):
                logger.warning(
                    f"High imbalance detected for {attr}. "
                    f"Max/min ratio: {max_ratio:.2f}"
                )
                
            # Log distribution
            logger.info(f"\nDistribution for {attr}:")
            logger.info(group_sizes.to_string())
            
    except Exception as e:
        logger.error(f"Error monitoring original distributions: {str(e)}")
        raise

def _create_interaction_terms(features: pd.DataFrame) -> None:
    """Creates interaction terms between relevant features."""
    try:
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        
        # Create interactions between educational features
        if 'studied_credits' in numeric_cols and 'num_of_prev_attempts' in numeric_cols:
            features['credits_per_attempt'] = (
                features['studied_credits'] / 
                features['num_of_prev_attempts'].clip(1)
            )
            
    except Exception as e:
        logger.error(f"Error creating interaction terms: {str(e)}")
        raise

def _monitor_feature_distributions(features: pd.DataFrame) -> None:
    """Monitors distributions of created features."""
    try:
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        
        for col in numeric_features.columns:
            if col in FAIRNESS['protected_attributes']:
                continue
                
            # Calculate statistics by protected group
            for attr in FAIRNESS['protected_attributes']:
                if attr not in features.columns:
                    continue
                    
                group_stats = features.groupby(attr)[col].agg(['mean', 'std'])
                
                # Calculate disparity
                max_mean = group_stats['mean'].max()
                min_mean = group_stats['mean'].min()
                disparity = (max_mean - min_mean) / max_mean if max_mean != 0 else 0
                
                if disparity > FAIRNESS['threshold']:
                    logger.warning(
                        f"High demographic disparity detected in {col} "
                        f"for {attr}: {disparity:.2f}"
                    )
                    logger.info(f"\nGroup statistics for {col}:")
                    logger.info(group_stats.to_string())
                    
    except Exception as e:
        logger.error(f"Error monitoring feature distributions: {str(e)}")
        raise

def _export_demographic_metadata(features: pd.DataFrame, params: Dict) -> None:
    """Exports metadata about demographic features."""
    try:
        metadata_path = Path(DIRS['feature_metadata']) / 'demographic_features.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'feature_count': len(features.columns),
            'protected_attributes': [
                attr for attr in FAIRNESS['protected_attributes']
                if attr in features.columns
            ],
            'group_sizes': {
                attr: features[attr].value_counts().to_dict()
                for attr in FAIRNESS['protected_attributes']
                if attr in features.columns
            },
            'parameters': params,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Exported demographic feature metadata to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error exporting demographic metadata: {str(e)}")
        raise