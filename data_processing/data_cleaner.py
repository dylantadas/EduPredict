import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from scipy import stats
from config import DATA_PROCESSING, PROTECTED_ATTRIBUTES, FAIRNESS, BIAS_MITIGATION

logger = logging.getLogger('edupredict')

def clean_demographic_data(
    student_info: pd.DataFrame,
    missing_value_strategy: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Cleans demographic data and handles missing values.

    Args:
        student_info: DataFrame containing student demographic information
        missing_value_strategy: Dictionary mapping columns to missing value strategies
        logger: Logger for tracking cleaning process

    Returns:
        Cleaned demographic DataFrame
    """
    logger = logger or logging.getLogger('edupredict')
    cleaned_data = student_info.copy()
    original_count = len(cleaned_data)
    protected_cols = list(PROTECTED_ATTRIBUTES.keys())

    # Log initial protected attribute distributions
    logger.info("Initial protected attribute distributions:")
    for col in protected_cols:
        if col in cleaned_data.columns:
            logger.info(f"{col} distribution:\n{cleaned_data[col].value_counts(normalize=True)}")

    # Use missing value strategy from config
    strategy = missing_value_strategy or DATA_PROCESSING.get('missing_value_strategy', 'median')
    default_strategies = {
        'imd_band': 'unknown',
        'disability': 'N',
        'highest_education': 'unknown',
        'age_band': 'unknown',
        'region': 'unknown'
    }

    try:
        # Filter rows with missing final_result 
        cleaned_data = cleaned_data.dropna(subset=['final_result'])
        removed_count = original_count - len(cleaned_data)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows ({removed_count/original_count:.2%}) with missing final_result")
            # Check impact on protected groups
            for col in protected_cols:
                if col in cleaned_data.columns:
                    before_dist = student_info[col].value_counts(normalize=True)
                    after_dist = cleaned_data[col].value_counts(normalize=True)
                    max_diff = max(abs(before_dist - after_dist))
                    if max_diff > FAIRNESS['threshold']:
                        logger.warning(f"Significant change in {col} distribution after filtering: {max_diff:.3f}")

        # Apply data types from config
        for col, dtype in DATA_PROCESSING['dtypes'].items():
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].astype(dtype)

        # Standardize string columns with special handling for protected attributes
        string_columns = ['gender', 'region', 'highest_education', 'imd_band', 'age_band']
        for col in string_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].str.strip().str.lower()
                # Validate protected attribute values
                if col in protected_cols:
                    valid_values = [v.lower() for v in PROTECTED_ATTRIBUTES[col]['values']]
                    invalid_values = cleaned_data[col].unique().tolist()
                    invalid_values = [v for v in invalid_values if v not in valid_values and pd.notna(v)]
                    if invalid_values:
                        logger.warning(f"Invalid values found in {col}: {invalid_values}")

        # Handle missing values according to strategy
        for col in cleaned_data.columns:
            if cleaned_data[col].isnull().any():
                if col in protected_cols:
                    # Use mode for protected attributes to preserve distributions
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])
                    logger.info(f"Filled missing values in protected attribute {col} using mode")
                elif strategy == 'median' and cleaned_data[col].dtype in ['int64', 'float64']:
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                elif strategy == 'mode':
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])

        # Check minimum group sizes for protected attributes
        min_group_size = FAIRNESS['min_group_size']
        for col in protected_cols:
            if col in cleaned_data.columns:
                group_sizes = cleaned_data[col].value_counts()
                small_groups = group_sizes[group_sizes < min_group_size]
                if not small_groups.empty:
                    logger.warning(f"Groups in {col} below minimum size ({min_group_size}): {small_groups.to_dict()}")

        # Log final protected attribute distributions
        logger.info("Final protected attribute distributions:")
        for col in protected_cols:
            if col in cleaned_data.columns:
                logger.info(f"{col} distribution:\n{cleaned_data[col].value_counts(normalize=True)}")

        # Validate cleaned data
        null_cols = cleaned_data.isnull().sum()
        if null_cols.any():
            logger.warning(f"Remaining null values after cleaning: {null_cols[null_cols > 0].to_dict()}")

        return cleaned_data

    except Exception as e:
        logger.error(f"Error cleaning demographic data: {str(e)}")
        raise

def clean_vle_data(
    vle_interactions: pd.DataFrame,
    vle_materials: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Cleans VLE interaction data and removes invalid entries.

    Args:
        vle_interactions: DataFrame of VLE interactions
        vle_materials: DataFrame of VLE materials
        logger: Logger for tracking cleaning process

    Returns:
        Cleaned and merged VLE data
    """
    logger = logger or logging.getLogger('edupredict')
    
    try:
        # Clean materials data first
        clean_materials = vle_materials.copy()
        clean_materials['activity_type'] = clean_materials['activity_type'].fillna('unknown')
        
        # Clean interactions data
        clean_interactions = vle_interactions.copy()
        
        # Remove invalid click counts
        original_count = len(clean_interactions)
        clean_interactions = clean_interactions[clean_interactions['sum_click'] > 0]
        removed_count = original_count - len(clean_interactions)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows with invalid click counts")
        
        # Merge with materials
        cleaned_data = pd.merge(
            clean_interactions,
            clean_materials[['id_site', 'activity_type']],
            on='id_site',
            how='left'
        )
        
        # Sort by student and date
        cleaned_data = cleaned_data.sort_values(['id_student', 'date'])
        
        return cleaned_data

    except Exception as e:
        logger.error(f"Error cleaning VLE data: {str(e)}")
        raise

def clean_assessment_data(
    assessments: pd.DataFrame,
    student_assessments: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Cleans assessment data and handles missing scores.

    Args:
        assessments: DataFrame of assessment information
        student_assessments: DataFrame of student assessment results
        logger: Logger for tracking cleaning process

    Returns:
        Cleaned assessment data
    """
    logger = logger or logging.getLogger('edupredict')
    
    try:
        # Clean student assessment data
        clean_student_assessments = student_assessments.copy()
        
        # Remove invalid scores
        original_count = len(clean_student_assessments)
        clean_student_assessments = clean_student_assessments[
            (clean_student_assessments['score'] >= 0) &
            (clean_student_assessments['score'] <= 100)
        ]
        removed_count = original_count - len(clean_student_assessments)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows with invalid scores")

        # Merge with assessment information
        cleaned_data = pd.merge(
            clean_student_assessments,
            assessments,
            on='id_assessment',
            how='left'
        )
        
        return cleaned_data

    except Exception as e:
        logger.error(f"Error cleaning assessment data: {str(e)}")
        raise

def validate_data_consistency(
    datasets: Dict[str, pd.DataFrame],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Validates consistency across cleaned datasets.

    Args:
        datasets: Dictionary of cleaned datasets
        logger: Logger for tracking validation process

    Returns:
        Dictionary of validation results and issues
    """
    logger = logger or logging.getLogger('edupredict')
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'issues': [],
        'statistics': {}
    }

    try:
        # Check student ID consistency
        student_ids = set(datasets['student_info']['id_student'])
        student_count = len(student_ids)
        validation_results['statistics']['total_students'] = student_count

        for name, df in datasets.items():
            if 'id_student' in df.columns and name != 'student_info':
                dataset_student_ids = set(df['id_student'])
                unknown_ids = dataset_student_ids - student_ids
                if unknown_ids:
                    validation_results['issues'].append({
                        'dataset': name,
                        'issue': 'unknown_student_ids',
                        'count': len(unknown_ids)
                    })
                    validation_results['is_valid'] = False

        # Check value ranges
        if 'student_assessments' in datasets:
            score_stats = datasets['student_assessments']['score'].describe()
            if score_stats['min'] < 0 or score_stats['max'] > 100:
                validation_results['issues'].append({
                    'dataset': 'student_assessments',
                    'issue': 'invalid_scores',
                    'details': {'min': score_stats['min'], 'max': score_stats['max']}
                })
                validation_results['is_valid'] = False

        # Check for temporal consistency in VLE data
        if 'vle_interactions' in datasets:
            vle_df = datasets['vle_interactions']
            if (vle_df['date'] < 0).any():
                validation_results['issues'].append({
                    'dataset': 'vle_interactions',
                    'issue': 'negative_dates'
                })
                validation_results['is_valid'] = False

        return validation_results

    except Exception as e:
        logger.error(f"Error validating data consistency: {str(e)}")
        validation_results['is_valid'] = False
        validation_results['issues'].append({
            'issue': 'validation_error',
            'details': str(e)
        })
        return validation_results

def detect_and_handle_outliers(
    df: pd.DataFrame,
    numerical_cols: List[str],
    method: str = 'iqr',
    threshold: float = 1.5,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Detects and handles outliers in numerical data with fairness considerations.

    Args:
        df: DataFrame to process
        numerical_cols: List of numerical columns to check
        method: Method for outlier detection ('iqr', 'zscore', etc.)
        threshold: Threshold for outlier detection
        logger: Logger for tracking outlier handling

    Returns:
        DataFrame with handled outliers
    """
    logger = logger or logging.getLogger('edupredict')
    cleaned_df = df.copy()
    protected_cols = list(PROTECTED_ATTRIBUTES.keys())
    
    try:
        # First analyze outliers by protected group
        outlier_stats = {col: {} for col in numerical_cols}
        
        for protected_col in protected_cols:
            if protected_col in df.columns:
                logger.info(f"Analyzing outliers by {protected_col} groups:")
                for group in df[protected_col].unique():
                    group_mask = df[protected_col] == group
                    group_df = df[group_mask]
                    
                    for col in numerical_cols:
                        if col not in group_df.columns:
                            continue
                            
                        values = group_df[col].dropna()
                        if len(values) == 0:
                            continue
                            
                        if method == 'iqr':
                            Q1 = values.quantile(0.25)
                            Q3 = values.quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - threshold * IQR
                            upper = Q3 + threshold * IQR
                            outliers = (values < lower) | (values > upper)
                        elif method == 'zscore':
                            z_scores = np.abs(stats.zscore(values))
                            outliers = z_scores > threshold
                        else:
                            raise ValueError(f"Unknown outlier detection method: {method}")
                            
                        n_outliers = outliers.sum()
                        if n_outliers > 0:
                            outlier_rate = n_outliers / len(values)
                            outlier_stats[col].setdefault(protected_col, {})[group] = {
                                'n_outliers': n_outliers,
                                'outlier_rate': outlier_rate,
                                'total_values': len(values)
                            }
                            
                            # Check for significant differences in outlier rates
                            if outlier_rate > FAIRNESS['threshold']:
                                logger.warning(
                                    f"High outlier rate ({outlier_rate:.2%}) for {col} in "
                                    f"{protected_col}={group} group"
                                )
        
        # Handle outliers while maintaining group distributions
        for col in numerical_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            values = df[col].dropna()
            if len(values) == 0:
                continue
                
            # Calculate global bounds
            if method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (values < lower_bound) | (values > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(values))
                outlier_mask = z_scores > threshold
            
            if outlier_mask is not None:
                n_outliers = outlier_mask.sum()
                if n_outliers > 0:
                    logger.info(f"Found {n_outliers} outliers in {col}")
                    
                    # Check impact on protected groups before handling
                    for protected_col in protected_cols:
                        if protected_col in df.columns:
                            before_dist = df[protected_col].value_counts(normalize=True)
                            after_dist = df[~outlier_mask][protected_col].value_counts(normalize=True)
                            max_diff = max(abs(before_dist - after_dist))
                            
                            if max_diff > FAIRNESS['threshold']:
                                logger.warning(
                                    f"Outlier handling would significantly impact {protected_col} "
                                    f"distribution (diff: {max_diff:.3f})"
                                )
                    
                    # Handle outliers using bounds or group-specific medians
                    if method == 'iqr':
                        cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                        cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
                    elif method == 'zscore':
                        for protected_col in protected_cols:
                            if protected_col in df.columns:
                                for group in df[protected_col].unique():
                                    group_mask = (df[protected_col] == group) & outlier_mask
                                    if group_mask.any():
                                        group_median = df.loc[df[protected_col] == group, col].median()
                                        cleaned_df.loc[group_mask, col] = group_median
        
        logger.info(f"Outlier detection summary by protected groups: {outlier_stats}")
        return cleaned_df

    except Exception as e:
        logger.error(f"Error handling outliers: {str(e)}")
        raise