import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from scipy import stats
from config import DATA_PROCESSING, PROTECTED_ATTRIBUTES, FAIRNESS, BIAS_MITIGATION, MODULE_CODES, PRESENTATION_CODES

logger = logging.getLogger('edupredict')

def clean_demographic_data(
    student_info: pd.DataFrame,
    missing_value_strategy: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Cleans demographic data and handles missing values with enhanced fairness considerations.
    """
    logger = logger or logging.getLogger('edupredict')
    
    if student_info.empty:
        raise ValueError("Student demographic data cannot be empty")
        
    cleaned_data = student_info.copy()
    protected_cols = list(PROTECTED_ATTRIBUTES.keys())

    try:
        # Initialize default strategies
        strategy = missing_value_strategy or DATA_PROCESSING.get('missing_value_strategy', 'median')
        default_strategies = {
            'imd_band': 'unknown',
            'disability': 'unknown',  # Changed from 'N' to 'unknown' to preserve original distribution
            'highest_education': 'unknown',
            'age_band': 'unknown',
            'region': 'unknown'
        }
        
        # Log initial distributions
        logger.info("Initial protected attribute distributions:")
        for col in protected_cols:
            if col in cleaned_data.columns:
                logger.info(f"{col} distribution:\n{cleaned_data[col].value_counts(normalize=True)}")

        # Handle protected attributes first
        for col in protected_cols:
            if col in cleaned_data.columns:
                # Special handling for age_band merging
                if col == 'age_band':
                    # Merge 35-55 and 55<= into 35+
                    cleaned_data[col] = cleaned_data[col].replace({'35-55': '35+', '55<=': '35+'})
                    logger.info("Merged age bands 35-55 and 55<= into 35+")
                    valid_values = ['0-35', '35+']
                else:
                    valid_values = PROTECTED_ATTRIBUTES[col]['values']
                
                # Special handling for disability - ensure proper values are maintained
                if col == 'disability':
                    # First standardize existing values to match the expected ones
                    # Important: Preserve case for disability values and map correctly
                    value_map = {'Y': 'Y', 'N': 'N', 'y': 'Y', 'n': 'N'}
                    cleaned_data[col] = cleaned_data[col].map(lambda x: value_map.get(x, x))
                    
                    # Log the standardized distribution
                    logger.info(f"Standardized {col} distribution:\n{cleaned_data[col].value_counts(normalize=True)}")
                
                # Include 'unknown' for missing value handling
                all_values = list(valid_values) + ['unknown']
                
                # Create category with all possible values - preserve case for disability
                if col == 'disability':
                    cleaned_data[col] = pd.Categorical(
                        cleaned_data[col],
                        categories=all_values,
                        ordered=False
                    )
                else:
                    cleaned_data[col] = pd.Categorical(
                        cleaned_data[col].str.strip().str.lower(),
                        categories=all_values,
                        ordered=False
                    )

                # Handle missing values using proportional sampling to maintain distributions
                null_mask = cleaned_data[col].isnull()
                if null_mask.any():
                    # Get current distribution excluding nulls
                    current_dist = cleaned_data[col].value_counts(normalize=True).fillna(0)
                    if current_dist.sum() > 0 and len(current_dist) > 1:  # Ensure we have valid probabilities and diversity
                        # If disability column has very skewed distribution or only unknown values,
                        # ensure we maintain a representative distribution
                        if col == 'disability' and ('Y' not in current_dist or current_dist['Y'] < 0.01):
                            # Force minimum representation of disability values
                            logger.info(f"Ensuring minimum representation of disability values")
                            modified_dist = current_dist.copy()
                            if 'Y' not in modified_dist or modified_dist['Y'] < 0.08:
                                modified_dist['Y'] = 0.08  # Ensure at least 8% representation
                            if 'N' not in modified_dist or modified_dist['N'] < 0.40:
                                modified_dist['N'] = 0.40  # Ensure at least 40% representation
                            # Normalize to 1.0
                            modified_dist = modified_dist / modified_dist.sum()
                            
                            # Use modified distribution for sampling
                            cleaned_data.loc[null_mask, col] = np.random.choice(
                                modified_dist.index,
                                size=null_mask.sum(),
                                p=modified_dist.values
                            )
                        else:
                            # Use current distribution for sampling
                            cleaned_data.loc[null_mask, col] = np.random.choice(
                                current_dist.index,
                                size=null_mask.sum(),
                                p=current_dist.values
                            )
                    else:
                        # If no valid distribution, use default value
                        cleaned_data.loc[null_mask, col] = default_strategies[col]
                    
                    logger.info(f"Filled {null_mask.sum()} missing values in {col}")

        # Handle num_of_prev_attempts with fairness-aware imputation
        if 'num_of_prev_attempts' not in cleaned_data.columns:
            cleaned_data['num_of_prev_attempts'] = 0
            logger.info("Added num_of_prev_attempts column with default value 0")
        else:
            initial_nulls = cleaned_data['num_of_prev_attempts'].isnull().sum()
            if initial_nulls > 0:
                logger.info(f"Found {initial_nulls} null values in num_of_prev_attempts")
                
                # Group-aware imputation
                for attr in protected_cols:
                    if attr in cleaned_data.columns:
                        group_stats = cleaned_data.groupby(attr)['num_of_prev_attempts'].agg(['mean', 'std'])
                        for group in cleaned_data[attr].unique():
                            mask = (cleaned_data[attr] == group) & cleaned_data['num_of_prev_attempts'].isnull()
                            if mask.any():
                                group_mean = group_stats.loc[group, 'mean']
                                if pd.isna(group_mean):  # If group stats are invalid, use global mean
                                    group_mean = cleaned_data['num_of_prev_attempts'].mean()
                                group_std = np.clip(group_stats.loc[group, 'std'], a_min=1e-6, a_max=None)
                                cleaned_data.loc[mask, 'num_of_prev_attempts'] = group_mean

        # Handle other categorical columns
        for col in cleaned_data.columns:
            if col not in protected_cols and col in default_strategies:
                if not isinstance(cleaned_data[col].dtype, pd.CategoricalDtype):
                    # Convert to lowercase if string
                    if cleaned_data[col].dtype == object:
                        cleaned_data[col] = cleaned_data[col].str.strip().str.lower()
                    
                    # Get unique values including default
                    unique_vals = list(cleaned_data[col].dropna().unique()) + [default_strategies[col]]
                    # Remove duplicates while preserving order
                    unique_vals = list(dict.fromkeys(unique_vals))
                    
                    # Create categorical with all values
                    cleaned_data[col] = pd.Categorical(
                        cleaned_data[col],
                        categories=unique_vals,
                        ordered=False
                    )

                # Fill missing values
                if cleaned_data[col].isnull().any():
                    default_value = default_strategies[col]
                    if default_value not in cleaned_data[col].cat.categories:
                        cleaned_data[col] = cleaned_data[col].cat.add_categories([default_value])
                    cleaned_data[col] = cleaned_data[col].fillna(default_value)

        # Apply bias mitigation if configured
        if BIAS_MITIGATION['method'] != 'none':
            for col in protected_cols:
                if col in cleaned_data.columns:
                    # Filter out empty groups
                    group_sizes = cleaned_data[col].value_counts()
                    non_empty_groups = group_sizes[group_sizes > 0]
                    
                    if len(non_empty_groups) >= 2:  # Need at least 2 groups to calculate ratio
                        max_ratio = non_empty_groups.max() / non_empty_groups.min()
                        
                        if max_ratio > BIAS_MITIGATION['max_ratio']:
                            logger.warning(f"Applying {BIAS_MITIGATION['method']} bias mitigation for {col}")
                            
                            if BIAS_MITIGATION['method'] == 'reweight':
                                weights = 1 / (non_empty_groups / len(cleaned_data))
                                cleaned_data[f'{col}_weight'] = cleaned_data[col].map(weights)
                            elif BIAS_MITIGATION['method'] in ['oversample', 'undersample']:
                                cleaned_data = _apply_sampling_strategy(
                                    cleaned_data,
                                    col,
                                    BIAS_MITIGATION['method'],
                                    BIAS_MITIGATION['target_ratios']
                                )

        # Validate final state
        logger.info("\nFinal protected attribute distributions:")
        for col in protected_cols:
            if col in cleaned_data.columns:
                final_dist = cleaned_data[col].value_counts(normalize=True)
                logger.info(f"{col} distribution:\n{final_dist}")

        return cleaned_data

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

def validate_module_codes(
    df: pd.DataFrame,
    strategy: str = 'warn',  # Options: 'warn', 'remove', 'replace'
    replacement_values: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Validates and standardizes module and presentation codes.
    
    Args:
        df: DataFrame containing code_module and/or code_presentation columns
        strategy: How to handle invalid codes:
            - 'warn': Keep invalid codes but log warnings (default)
            - 'remove': Remove rows with invalid codes
            - 'replace': Replace invalid codes with specified replacements
        replacement_values: Dictionary mapping invalid codes to valid replacements
        logger: Optional logger instance
    
    Returns:
        Validated DataFrame with standardized codes
    """
    logger = logger or logging.getLogger('edupredict')
    validated = df.copy()
    validation_stats = {
        'code_module': {'valid': 0, 'invalid': 0, 'unique_invalid': set()},
        'code_presentation': {'valid': 0, 'invalid': 0, 'unique_invalid': set()}
    }
    
    try:
        # Validate code_module
        if 'code_module' in validated.columns:
            valid_modules = set(MODULE_CODES['values'])
            invalid_mask = ~validated['code_module'].isin(valid_modules)
            
            if invalid_mask.any():
                invalid_codes = validated.loc[invalid_mask, 'code_module'].unique()
                validation_stats['code_module'].update({
                    'invalid': invalid_mask.sum(),
                    'valid': (~invalid_mask).sum(),
                    'unique_invalid': set(invalid_codes)
                })
                
                logger.warning(
                    f"Found {len(invalid_codes)} unique invalid module codes: {invalid_codes}\n"
                    f"Total invalid entries: {invalid_mask.sum()}"
                )
                
                if strategy == 'remove':
                    validated = validated[~invalid_mask]
                    logger.info(f"Removed {invalid_mask.sum()} rows with invalid module codes")
                elif strategy == 'replace' and replacement_values:
                    for invalid_code in invalid_codes:
                        if invalid_code in replacement_values:
                            mask = validated['code_module'] == invalid_code
                            validated.loc[mask, 'code_module'] = replacement_values[invalid_code]
                            logger.info(f"Replaced module code {invalid_code} with {replacement_values[invalid_code]}")
                
            # Convert to category type with valid codes
            if strategy != 'remove':
                # Include both valid codes and any remaining invalid ones
                all_codes = list(valid_modules | set(validated['code_module'].unique()))
            else:
                all_codes = list(valid_modules)
                
            validated['code_module'] = pd.Categorical(
                validated['code_module'],
                categories=all_codes,
                ordered=False
            )
            
            # Log module code distribution
            module_dist = validated['code_module'].value_counts(normalize=True)
            logger.info(f"\nModule code distribution:\n{module_dist}")
        
        # Validate code_presentation
        if 'code_presentation' in validated.columns:
            valid_presentations = set(PRESENTATION_CODES['values'])
            invalid_mask = ~validated['code_presentation'].isin(valid_presentations)
            
            if invalid_mask.any():
                invalid_codes = validated.loc[invalid_mask, 'code_presentation'].unique()
                validation_stats['code_presentation'].update({
                    'invalid': invalid_mask.sum(),
                    'valid': (~invalid_mask).sum(),
                    'unique_invalid': set(invalid_codes)
                })
                
                logger.warning(
                    f"Found {len(invalid_codes)} unique invalid presentation codes: {invalid_codes}\n"
                    f"Total invalid entries: {invalid_mask.sum()}"
                )
                
                if strategy == 'remove':
                    validated = validated[~invalid_mask]
                    logger.info(f"Removed {invalid_mask.sum()} rows with invalid presentation codes")
                elif strategy == 'replace' and replacement_values:
                    for invalid_code in invalid_codes:
                        if invalid_code in replacement_values:
                            mask = validated['code_presentation'] == invalid_code
                            validated.loc[mask, 'code_presentation'] = replacement_values[invalid_code]
                            logger.info(f"Replaced presentation code {invalid_code} with {replacement_values[invalid_code]}")
            
            # Convert to category type with chronological order
            if strategy != 'remove':
                # Include both valid codes and any remaining invalid ones
                all_codes = list(valid_presentations | set(validated['code_presentation'].unique()))
            else:
                all_codes = list(valid_presentations)
            
            validated['code_presentation'] = pd.Categorical(
                validated['code_presentation'],
                categories=sorted(all_codes),  # Sort chronologically
                ordered=True
            )
            
            # Log presentation code distribution
            pres_dist = validated['code_presentation'].value_counts(normalize=True)
            logger.info(f"\nPresentation code distribution:\n{pres_dist}")

        # Add validation metadata
        if validation_stats['code_module']['invalid'] > 0 or validation_stats['code_presentation']['invalid'] > 0:
            logger.warning("\nValidation Summary:")
            for col, stats in validation_stats.items():
                if stats['invalid'] > 0:
                    logger.warning(
                        f"{col}:\n"
                        f"  Valid entries: {stats['valid']}\n"
                        f"  Invalid entries: {stats['invalid']}\n"
                        f"  Invalid codes: {stats['unique_invalid']}"
                    )
        
        return validated
        
    except Exception as e:
        logger.error(f"Error validating module codes: {str(e)}")
        raise

def clean_vle_data(
    vle_interactions: pd.DataFrame,
    vle_materials: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Cleans VLE interaction data and removes invalid entries.
    The date field represents days relative to the module start date, where:
    - Negative values indicate interactions before module start
    - Zero indicates the module start date
    - Positive values indicate days after module start
    """
    logger = logger or logging.getLogger('edupredict')
    
    try:
        if vle_interactions.empty:
            raise ValueError("VLE interaction data cannot be empty")
            
        # Clean interactions data first
        clean_interactions = vle_interactions.copy()
        
        # Validate module and presentation codes
        clean_interactions = validate_module_codes(clean_interactions, logger=logger)
        
        # Remove invalid click counts (ensure positive integers)
        original_count = len(clean_interactions)
        clean_interactions = clean_interactions[
            (clean_interactions['sum_click'] > 0) & 
            (clean_interactions['sum_click'].astype(float).apply(lambda x: x.is_integer() if isinstance(x, float) else True))
        ]
        removed_count = original_count - len(clean_interactions)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows with invalid click counts (non-positive or non-integer)")
            
        # Convert to integer type after validation
        clean_interactions['sum_click'] = clean_interactions['sum_click'].astype(int)
        
        # Ensure date column exists and analyze temporal distribution
        if 'date' not in clean_interactions.columns:
            logger.error("Required 'date' column missing from VLE interactions")
            raise ValueError("Required 'date' column missing from VLE interactions")
            
        # Log temporal distribution information
        pre_module = clean_interactions['date'] < 0
        if pre_module.any():
            logger.info(
                f"Found {pre_module.sum()} interactions before module start "
                f"(min: {clean_interactions['date'].min()} days)"
            )
        
        # Sort by date to ensure temporal consistency
        clean_interactions = clean_interactions.sort_values(['id_student', 'date'])
        
        # Only merge with materials if the DataFrame is not empty
        if not vle_materials.empty and 'id_site' in vle_materials.columns:
            # Clean materials data
            clean_materials = vle_materials.copy()
            if 'activity_type' in clean_materials.columns:
                clean_materials['activity_type'] = clean_materials['activity_type'].fillna('unknown')
            else:
                clean_materials['activity_type'] = 'unknown'
                logger.warning("activity_type column missing from vle_materials, using default value 'unknown'")
            
            # Merge with materials
            cleaned_data = pd.merge(
                clean_interactions,
                clean_materials[['id_site', 'activity_type']],
                on='id_site',
                how='left'
            )
        else:
            # If materials data is empty, just use interactions data
            cleaned_data = clean_interactions
            cleaned_data['activity_type'] = 'unknown'
            logger.warning("No VLE materials data provided, using default activity_type 'unknown'")
        
        # Add temporal features properly handling negative dates
        cleaned_data['day_of_week'] = cleaned_data['date'].abs() % 7  # Use abs() only for day of week
        cleaned_data['week_number'] = cleaned_data['date'] // 7  # Preserve negative weeks for pre-module activity
        cleaned_data['is_pre_module'] = cleaned_data['date'] < 0  # Flag pre-module activity
        
        logger.info(f"Final cleaned VLE data shape: {cleaned_data.shape}")
        return cleaned_data

    except Exception as e:
        logger.error(f"Error cleaning VLE data: {str(e)}")
        raise

def clean_assessment_data(
    assessment_info: pd.DataFrame,
    student_assessments: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean assessment data with enhanced validation.
    """
    logger = logger or logging.getLogger('edupredict')
    
    try:
        # Clean assessment info
        cleaned_assessments = assessment_info.copy()
        cleaned_submissions = student_assessments.copy()
        
        # Remove final exams with no submissions
        exam_mask = cleaned_assessments['assessment_type'] == 'Exam'
        exam_ids = cleaned_assessments[exam_mask]['id_assessment']
        submissions_per_exam = cleaned_submissions[
            cleaned_submissions['id_assessment'].isin(exam_ids)
        ]['id_assessment'].value_counts()
        
        exams_no_submissions = exam_ids[~exam_ids.isin(submissions_per_exam.index)]
        if not exams_no_submissions.empty:
            logger.info(f"Removing {len(exams_no_submissions)} final exams with no submissions")
            cleaned_assessments = cleaned_assessments[
                ~cleaned_assessments['id_assessment'].isin(exams_no_submissions)
            ]
        
        # Handle invalid scores
        score_mask = (
            (cleaned_submissions['score'] < 0) |
            (cleaned_submissions['score'] > 100)
        )
        invalid_scores = cleaned_submissions[score_mask]
        if not invalid_scores.empty:
            logger.warning(f"Removed {len(invalid_scores)} rows with invalid scores")
            cleaned_submissions = cleaned_submissions[~score_mask]
        
        # Handle banked assessments
        banked_mask = cleaned_submissions['is_banked'] == 1
        banked_count = banked_mask.sum()
        if banked_count > 0:
            logger.info(f"Found {banked_count} banked assessment results")
        
        # Validate assessment weights
        zero_weight_mask = cleaned_assessments['weight'] == 0
        if zero_weight_mask.any():
            n_zero_weights = zero_weight_mask.sum()
            logger.warning(f"Found {n_zero_weights} assessments with zero weight, setting to default weight")
            default_weight = 0.1  # 10% default weight
            cleaned_assessments.loc[zero_weight_mask, 'weight'] = default_weight
        
        # Log cleaning summary
        total_removed = len(student_assessments) - len(cleaned_submissions)
        if total_removed > 0:
            removal_pct = total_removed / len(student_assessments)
            logger.info(f"Total submissions removed in cleaning: {total_removed} ({removal_pct:.1%})")
        
        return cleaned_assessments, cleaned_submissions
        
    except Exception as e:
        logger.error(f"Error in assessment data cleaning: {str(e)}")
        raise

def validate_data_consistency(
    datasets: Dict[str, pd.DataFrame],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Validates data consistency across datasets.
    
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
        'issues': []
    }

    try:
        # Check student ID consistency
        if 'student_info' in datasets:
            student_ids = set(datasets['student_info']['id_student'])
            student_count = len(student_ids)

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

        # Check value ranges for assessment scores
        if 'student_assessments' in datasets:
            score_stats = datasets['student_assessments']['score'].describe()
            if score_stats['min'] < 0 or score_stats['max'] > 100:
                validation_results['issues'].append({
                    'dataset': 'student_assessments',
                    'issue': 'invalid_scores',
                    'details': {'min': score_stats['min'], 'max': score_stats['max']}
                })
                validation_results['is_valid'] = False

        # Check VLE data - negative dates are allowed but warn about extreme values
        if 'vle_interactions' in datasets:
            vle_df = datasets['vle_interactions']
            very_early_mask = vle_df['date'] < -60  # More than 60 days before start
            if very_early_mask.any():
                validation_results['warnings'].append({
                    'dataset': 'vle_interactions',
                    'warning': 'very_early_activity',
                    'details': {
                        'count': int(very_early_mask.sum()),
                        'earliest_day': float(vle_df['date'].min())
                    }
                })
                logger.warning(
                    f"Found {very_early_mask.sum()} VLE interactions more than 60 days "
                    f"before module start (earliest: {vle_df['date'].min()} days)"
                )
            
            very_late_mask = vle_df['date'] > 180  # More than 180 days after start
            if very_late_mask.any():
                validation_results['warnings'].append({
                    'dataset': 'vle_interactions',
                    'warning': 'very_late_activity',
                    'details': {
                        'count': int(very_late_mask.sum()),
                        'latest_day': float(vle_df['date'].max())
                    }
                })

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

def clean_registration_data(
    registration_data: pd.DataFrame,
    vle_data: Optional[pd.DataFrame] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Cleans student registration data with enhanced VLE activity analysis.
    Dates are in days relative to module start:
    - date_registration: Can be negative (registration before module start)
    - date_unregistration: Can be null (completed module) or positive (dropped out)
    """
    logger = logger or logging.getLogger('edupredict')
    cleaned_data = registration_data.copy()
    
    try:
        # Validate module and presentation codes
        cleaned_data = validate_module_codes(cleaned_data, logger=logger)
        
        # Handle missing unregistration dates
        # Null values indicate module completion
        completed = cleaned_data['date_unregistration'].isnull()
        if completed.any():
            logger.info(f"Found {completed.sum()} students who completed the module")
            # For completed students, set unregistration to module length or last VLE activity
            if vle_data is not None:
                # Use last VLE activity date for each student
                last_activity = vle_data.groupby('id_student')['date'].max()
                cleaned_data.loc[completed, 'date_unregistration'] = cleaned_data.loc[completed, 'id_student'].map(last_activity)
                logger.info("Set unregistration dates to last VLE activity for completed students")
            else:
                # Default to typical module length if no VLE data
                cleaned_data.loc[completed, 'date_unregistration'] = 180
                logger.info("Set unregistration dates to default module length (180 days) for completed students")
        
        # Check registration timeline
        early_registrations = cleaned_data[cleaned_data['date_registration'] < -60]
        if not early_registrations.empty:
            logger.warning(
                f"Found {len(early_registrations)} registrations more than 60 days "
                "before module start"
            )
        
        late_registrations = cleaned_data[cleaned_data['date_registration'] > 30]
        if not late_registrations.empty:
            logger.warning(
                f"Found {len(late_registrations)} registrations more than 30 days "
                "after module start"
            )
        
        # Analyze unregistration patterns for dropouts
        dropped = ~completed
        if dropped.any():
            dropout_times = cleaned_data.loc[dropped, 'date_unregistration']
            early_dropouts = dropout_times[dropout_times < 30]
            late_dropouts = dropout_times[dropout_times > 180]
            
            if not early_dropouts.empty:
                logger.warning(
                    f"Found {len(early_dropouts)} early dropouts (< 30 days into module)"
                )
            
            if not late_dropouts.empty:
                logger.warning(
                    f"Found {len(late_dropouts)} late dropouts (> 180 days into module)"
                )
            
            # Calculate average time to dropout
            avg_dropout_time = dropout_times.mean()
            logger.info(f"Average time to dropout: {avg_dropout_time:.1f} days")
        
        # Add derived features
        cleaned_data['completed_module'] = completed
        cleaned_data['enrollment_duration'] = cleaned_data['date_unregistration'] - cleaned_data['date_registration']
        cleaned_data['registration_type'] = pd.cut(
            cleaned_data['date_registration'],
            bins=[-float('inf'), -30.001, -7, 0, float('inf')],
            labels=['very_early', 'early', 'on_time', 'late']
        )
        
        # Add VLE engagement windows if data available
        if vle_data is not None:
            # Calculate pre-module engagement
            vle_by_student = vle_data.groupby('id_student').agg({
                'date': ['min', 'max', 'count'],
                'sum_click': 'sum'
            })
            vle_by_student.columns = ['first_activity', 'last_activity', 'total_activities', 'total_clicks']
            
            # Merge VLE statistics
            cleaned_data = cleaned_data.merge(
                vle_by_student,
                left_on='id_student',
                right_index=True,
                how='left'
            )
            
            # Fill missing VLE data with zeros
            vle_cols = ['total_activities', 'total_clicks']
            cleaned_data[vle_cols] = cleaned_data[vle_cols].fillna(0)
            
            # Calculate engagement metrics
            cleaned_data['days_before_first_activity'] = cleaned_data['first_activity'] - cleaned_data['date_registration']
            cleaned_data['engagement_span'] = cleaned_data['last_activity'] - cleaned_data['first_activity']
            cleaned_data['avg_activities_per_day'] = cleaned_data['total_activities'] / cleaned_data['engagement_span'].clip(lower=1)
            
            logger.info("Added VLE engagement metrics to registration data")
        
        return cleaned_data
        
    except Exception as e:
        logger.error(f"Error cleaning registration data: {str(e)}")
        raise