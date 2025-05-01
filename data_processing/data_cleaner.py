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
    Cleans demographic data and handles missing values with enhanced fairness considerations.

    Args:
        student_info: DataFrame containing student demographic information
        missing_value_strategy: Dictionary mapping columns to missing value strategies
        logger: Logger for tracking cleaning process

    Returns:
        Cleaned demographic DataFrame
    """
    logger = logger or logging.getLogger('edupredict')
    cleaned_data = student_info.copy()
    protected_cols = list(PROTECTED_ATTRIBUTES.keys())

    try:
        # Initialize null tracking
        initial_nulls = cleaned_data['num_of_prev_attempts'].isnull().sum()
        logger.info(f"Initial null values in num_of_prev_attempts: {initial_nulls}")

        # Handle num_of_prev_attempts with fairness-aware imputation
        if 'num_of_prev_attempts' in cleaned_data.columns:
            # First pass: Group by protected attributes for fair imputation
            for attr in protected_cols:
                if attr in cleaned_data.columns:
                    # Calculate and validate group means
                    group_means = cleaned_data.groupby(attr)['num_of_prev_attempts'].transform('mean')
                    group_means = group_means.fillna(cleaned_data['num_of_prev_attempts'].mean())
                    
                    # Apply group means where null
                    null_mask = cleaned_data['num_of_prev_attempts'].isnull()
                    cleaned_data.loc[null_mask, 'num_of_prev_attempts'] = group_means[null_mask]
                    
                    remaining_nulls = cleaned_data['num_of_prev_attempts'].isnull().sum()
                    logger.info(f"Remaining nulls after {attr} group imputation: {remaining_nulls}")

            # Second pass: Fill any remaining nulls with global statistics
            if cleaned_data['num_of_prev_attempts'].isnull().any():
                global_mean = cleaned_data['num_of_prev_attempts'].mean()
                if pd.isnull(global_mean):
                    global_mean = 0  # Default to 0 if still NaN
                cleaned_data['num_of_prev_attempts'].fillna(global_mean, inplace=True)
                logger.info(f"Filled remaining nulls with global mean: {global_mean}")

        # Verify final state
        final_nulls = cleaned_data['num_of_prev_attempts'].isnull().sum()
        if final_nulls > 0:
            logger.warning(f"Unexpected remaining null values: {final_nulls}")
        else:
            logger.info("Successfully imputed all null values in num_of_prev_attempts")

        # Rest of demographic cleaning...
        original_count = len(cleaned_data)

        try:
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

            # Normalize numerical values before filtering to prevent bias
            numerical_cols = ['studied_credits', 'num_of_prev_attempts']
            for col in numerical_cols:
                if col in cleaned_data.columns:
                    # Use robust scaling to minimize impact of outliers
                    cleaned_data[col] = (cleaned_data[col] - cleaned_data[col].median()) / (
                        cleaned_data[col].quantile(0.75) - cleaned_data[col].quantile(0.25)
                    )

            # Filter rows with missing final_result using stratified sampling
            if 'final_result' in cleaned_data.columns:
                missing_mask = cleaned_data['final_result'].isna()
                if missing_mask.any():
                    # Calculate sampling weights to maintain demographic proportions
                    weights = np.ones(len(cleaned_data))
                    for col in protected_cols:
                        if col in cleaned_data.columns:
                            group_props = cleaned_data[col].value_counts(normalize=True)
                            weights *= cleaned_data[col].map(lambda x: 1/group_props[x])
                    
                    # Normalize weights
                    weights = weights / weights.sum()
                    
                    # Stratified sampling of rows to remove
                    keep_mask = ~missing_mask | (
                        np.random.random(len(cleaned_data)) < weights
                    )
                    cleaned_data = cleaned_data[keep_mask]
                    
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

            # Standardize protected attributes using PROTECTED_ATTRIBUTES config
            for col in protected_cols:
                if col in cleaned_data.columns:
                    # Map values to standardized format
                    value_map = {v.lower(): v.lower() for v in PROTECTED_ATTRIBUTES[col]['values']}
                    cleaned_data[col] = cleaned_data[col].str.strip().str.lower().map(value_map)
                    
                    # Handle invalid values proportionally
                    invalid_mask = ~cleaned_data[col].isin(value_map.values())
                    if invalid_mask.any():
                        invalid_count = invalid_mask.sum()
                        logger.warning(f"Found {invalid_count} invalid values in {col}")
                        
                        # Distribute invalid values proportionally
                        valid_dist = cleaned_data.loc[~invalid_mask, col].value_counts(normalize=True)
                        invalid_indices = cleaned_data[invalid_mask].index
                        cleaned_data.loc[invalid_indices, col] = np.random.choice(
                            valid_dist.index,
                            size=len(invalid_indices),
                            p=valid_dist.values
                        )

            # Handle missing values with fairness awareness
            for col in cleaned_data.columns:
                if cleaned_data[col].isnull().any():
                    if col in protected_cols:
                        # Preserve group distributions when filling missing values
                        group_dist = cleaned_data[col].value_counts(normalize=True)
                        null_indices = cleaned_data[cleaned_data[col].isnull()].index
                        cleaned_data.loc[null_indices, col] = np.random.choice(
                            group_dist.index,
                            size=len(null_indices),
                            p=group_dist.values
                        )
                        logger.info(f"Filled missing values in protected attribute {col} preserving distributions")
                    elif strategy == 'median' and cleaned_data[col].dtype in ['int64', 'float64']:
                        # Fill numerical missing values while preserving group-specific distributions
                        for protected_col in protected_cols:
                            if protected_col in cleaned_data.columns:
                                group_medians = cleaned_data.groupby(protected_col)[col].transform('median')
                                cleaned_data[col] = cleaned_data[col].fillna(group_medians)
                    elif strategy == 'mode':
                        # Fill categorical missing values while preserving group-specific distributions
                        for protected_col in protected_cols:
                            if protected_col in cleaned_data.columns:
                                group_modes = cleaned_data.groupby(protected_col)[col].transform(
                                    lambda x: x.mode()[0] if not x.mode().empty else None
                                )
                                cleaned_data[col] = cleaned_data[col].fillna(group_modes)
                    else:
                        # Use default strategies from config
                        if col in default_strategies:
                            cleaned_data[col] = cleaned_data[col].fillna(default_strategies[col])

            # Apply bias mitigation if configured
            if BIAS_MITIGATION['method'] != 'none':
                for col in protected_cols:
                    if col in cleaned_data.columns:
                        group_sizes = cleaned_data[col].value_counts()
                        if (group_sizes.max() / group_sizes.min()) > BIAS_MITIGATION['max_ratio']:
                            logger.warning(f"Applying {BIAS_MITIGATION['method']} bias mitigation for {col}")
                            
                            if BIAS_MITIGATION['method'] == 'reweight':
                                # Calculate weights to balance groups
                                weights = 1 / (group_sizes / len(cleaned_data))
                                cleaned_data[f'{col}_weight'] = cleaned_data[col].map(weights)
                            
                            elif BIAS_MITIGATION['method'] in ['oversample', 'undersample']:
                                # Implement sampling in a separate function to maintain clean structure
                                cleaned_data = _apply_sampling_strategy(
                                    cleaned_data, 
                                    col, 
                                    BIAS_MITIGATION['method'],
                                    BIAS_MITIGATION['target_ratios']
                                )

            # Normalize num_of_prev_attempts using fairness-aware scaling
            if 'num_of_prev_attempts' in cleaned_data.columns:
                for protected_col in protected_cols:
                    if protected_col in cleaned_data.columns:
                        # Calculate group-specific statistics
                        group_stats = cleaned_data.groupby(protected_col)['num_of_prev_attempts'].agg(['mean', 'std'])
                        
                        # Apply group-specific normalization
                        for group in cleaned_data[protected_col].unique():
                            mask = cleaned_data[protected_col] == group
                            group_mean = group_stats.loc[group, 'mean']
                            # Clip std to prevent division by zero, with both lower and upper bounds
                            group_std = np.clip(group_stats.loc[group, 'std'], a_min=1e-6, a_max=None)
                            cleaned_data.loc[mask, 'num_of_prev_attempts'] = (
                                (cleaned_data.loc[mask, 'num_of_prev_attempts'] - group_mean) / group_std
                            )
                        
                        # Verify fairness after normalization
                        normalized_means = cleaned_data.groupby(protected_col)['num_of_prev_attempts'].mean()
                        max_diff = normalized_means.max() - normalized_means.min()
                        if max_diff > FAIRNESS['threshold']:
                            logger.warning(
                                f"Normalized num_of_prev_attempts still shows disparity for {protected_col}: {max_diff:.3f}"
                            )

            # Log final protected attribute distributions
            logger.info("Final protected attribute distributions:")
            for col in protected_cols:
                if col in cleaned_data.columns:
                    logger.info(f"{col} distribution:\n{cleaned_data[col].value_counts(normalize=True)}")

            # Final fairness validation
            fairness_metrics = {}
            for col in protected_cols:
                if col in cleaned_data.columns:
                    group_stats = {}
                    for feature in numerical_cols:
                        if feature in cleaned_data.columns:
                            group_means = cleaned_data.groupby(col)[feature].mean()
                            max_disparity = (group_means.max() - group_means.min()) / group_means.max()
                            group_stats[feature] = {
                                'disparity': max_disparity,
                                'threshold_exceeded': max_disparity > FAIRNESS['threshold']
                            }
                    fairness_metrics[col] = group_stats
                    
                    # Log concerning disparities
                    for feature, stats in group_stats.items():
                        if stats['threshold_exceeded']:
                            logger.warning(
                                f"Feature {feature} shows high disparity ({stats['disparity']:.3f}) "
                                f"for protected attribute {col}"
                            )

            # Validate cleaned data
            null_cols = cleaned_data.isnull().sum()
            if null_cols.any():
                logger.warning(f"Remaining null values after cleaning: {null_cols[null_cols > 0].to_dict()}")

            return cleaned_data

        except Exception as e:
            logger.error(f"Error cleaning demographic data: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in demographic cleaning: {str(e)}")
        raise

def _apply_sampling_strategy(
    data: pd.DataFrame,
    protected_col: str,
    method: str,
    target_ratios: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Applies sampling strategy to balance protected groups.
    
    Args:
        data: Input DataFrame
        protected_col: Protected attribute column
        method: Sampling method ('oversample' or 'undersample')
        target_ratios: Optional target ratios for each group
        
    Returns:
        Balanced DataFrame
    """
    group_sizes = data[protected_col].value_counts()
    
    if target_ratios:
        # Use specified target ratios
        target_sizes = {
            group: int(len(data) * ratio)
            for group, ratio in target_ratios.items()
        }
    else:
        if method == 'oversample':
            # Oversample minority groups to match majority
            target_sizes = {group: group_sizes.max() for group in group_sizes.index}
        else:
            # Undersample majority groups to match minority
            target_sizes = {group: group_sizes.min() for group in group_sizes.index}
    
    balanced_dfs = []
    for group in group_sizes.index:
        group_data = data[data[protected_col] == group]
        target_size = target_sizes[group]
        
        if len(group_data) < target_size:
            # Oversample with replacement
            balanced_dfs.append(group_data.sample(n=target_size, replace=True))
        else:
            # Undersample without replacement
            balanced_dfs.append(group_data.sample(n=target_size, replace=False))
    
    return pd.concat(balanced_dfs, ignore_index=True)

def clean_vle_data(
    vle_interactions: pd.DataFrame,
    vle_materials: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Cleans VLE interaction data and removes invalid entries.
    The date field represents days relative to the module start date,
    where negative values indicate interactions before the module started.

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
        
        # Remove invalid click counts and ensure date column exists
        original_count = len(clean_interactions)
        clean_interactions = clean_interactions[clean_interactions['sum_click'] > 0]
        removed_count = original_count - len(clean_interactions)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows with invalid click counts")
        
        # Ensure date column exists
        if 'date' not in clean_interactions.columns:
            logger.error("Required 'date' column missing from VLE interactions")
            raise ValueError("Required 'date' column missing from VLE interactions")
        
        # Sort by date to ensure temporal consistency
        clean_interactions = clean_interactions.sort_values(['id_student', 'date'])
        
        # Merge with materials
        cleaned_data = pd.merge(
            clean_interactions,
            clean_materials[['id_site', 'activity_type']],
            on='id_site',
            how='left'
        )
        
        # Verify date column still exists after merge
        if 'date' not in cleaned_data.columns:
            logger.error("Date column lost during merge operation")
            raise ValueError("Date column lost during merge operation")
        
        # Add additional temporal features if needed
        cleaned_data['day_of_week'] = cleaned_data['date'].abs() % 7  # Use abs() for day of week calculation
        cleaned_data['week_number'] = cleaned_data['date'] // 7  # Keep sign for week numbering
        
        logger.info(f"Final cleaned VLE data shape: {cleaned_data.shape}")
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
    Properly interprets assessment dates relative to module timeline:
    - Negative dates indicate assessments scheduled before module start
    - 0 is module start date
    - Positive dates are days since module start

    Args:
        assessments: DataFrame of assessment information
        student_assessments: DataFrame of student assessment results
        logger: Logger for tracking cleaning process

    Returns:
        Cleaned assessment data with enhanced date interpretation
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
        
        # Validate date fields
        if 'date' not in cleaned_data.columns:
            logger.error("Required 'date' column missing from assessments")
            raise ValueError("Required 'date' column missing from assessments")
            
        if 'date_submitted' not in cleaned_data.columns:
            logger.error("Required 'date_submitted' column missing from assessments")
            raise ValueError("Required 'date_submitted' column missing from assessments")
            
        # Create timeline context fields
        cleaned_data['is_pre_module'] = cleaned_data['date'] < 0
        cleaned_data['time_to_deadline'] = cleaned_data['date'] - cleaned_data['date_submitted']
        cleaned_data['submission_time'] = cleaned_data['date_submitted'] - cleaned_data['date']
        cleaned_data['is_late'] = cleaned_data['submission_time'] > 0
        cleaned_data['days_late'] = cleaned_data['submission_time'].clip(lower=0)
        
        # Group by student and check submission patterns
        submission_patterns = cleaned_data.groupby('id_student').agg({
            'is_late': ['mean', 'sum'],
            'days_late': ['mean', 'max'],
            'time_to_deadline': ['mean', 'std']
        })
        
        # Flatten column names
        submission_patterns.columns = [
            f"{col[0]}_{col[1]}" for col in submission_patterns.columns
        ]
        
        # Add submission pattern metrics back to main dataset
        cleaned_data = cleaned_data.merge(
            submission_patterns,
            left_on='id_student',
            right_index=True,
            how='left'
        )
        
        # Validate temporal consistency
        invalid_dates = cleaned_data[cleaned_data['date_submitted'] < cleaned_data['date']]
        if not invalid_dates.empty:
            logger.warning(
                f"Found {len(invalid_dates)} submissions with date_submitted before due date. "
                "This may indicate data quality issues."
            )
            # Log specific cases for investigation
            for _, row in invalid_dates.iterrows():
                logger.debug(
                    f"Invalid submission timing - Student: {row['id_student']}, "
                    f"Assessment: {row['id_assessment']}, "
                    f"Due: {row['date']}, Submitted: {row['date_submitted']}"
                )
        
        # Log submission timing statistics
        logger.info("\nSubmission timing statistics:")
        logger.info(f"Average days early/late: {cleaned_data['submission_time'].mean():.2f}")
        logger.info(f"Percentage of late submissions: {(cleaned_data['is_late'].mean() * 100):.2f}%")
        
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