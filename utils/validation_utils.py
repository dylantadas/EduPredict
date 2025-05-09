from typing import Dict, List, Set, Union, Any, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from config import FAIRNESS, PROTECTED_ATTRIBUTES

logger = logging.getLogger('edupredict')

def validate_directories(dirs: Dict[str, Path]) -> bool:
    """
    Validates that all required directories exist and are properly configured.
    
    Args:
        dirs: Dictionary of directory paths
        
    Returns:
        bool: True if all directories are valid
    """
    try:
        # Check if all paths are Path objects
        if not all(isinstance(path, Path) for path in dirs.values()):
            logger.error("Invalid directory configuration: all paths must be Path objects")
            return False
            
        # Check if all required directories exist
        for name, path in dirs.items():
            if not path.exists():
                logger.warning(f"Directory {name} at {path} does not exist")
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory {path}")
                
        return True
    except Exception as e:
        logger.error(f"Error validating directories: {str(e)}")
        return False

def validate_model_parameters(params: Dict[str, Any]) -> bool:
    """
    Validates model configuration parameters.
    
    Args:
        params: Dictionary of model parameters
        
    Returns:
        bool: True if all parameters are valid
    """
    try:
        # Validate test and validation split sizes
        test_size = params.get('TEST_SIZE', 0)
        val_size = params.get('VALIDATION_SIZE', 0)
        
        if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
            logger.error("Invalid test/validation split configuration")
            return False
            
        # Validate random seed
        if not isinstance(params.get('RANDOM_SEED', 0), int):
            logger.error("Random seed must be an integer")
            return False
            
        # Validate feature engineering parameters
        fe_params = params.get('FEATURE_ENGINEERING', {})
        if not (0 < fe_params.get('correlation_threshold', 1) <= 1):
            logger.error("Correlation threshold must be between 0 and 1")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating model parameters: {str(e)}")
        return False

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

        # Check for temporal consistency in VLE data - allow negative dates but warn about extremes
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

        # Check for timeline consistency across all temporal data
        all_dates = []
        if 'vle_interactions' in datasets:
            all_dates.extend(datasets['vle_interactions']['date'])
        if 'student_assessments' in datasets:
            all_dates.extend(datasets['student_assessments']['date_submitted'])
        if 'assessments' in datasets:
            all_dates.extend(datasets['assessments']['date'])
        if 'student_registration' in datasets:
            all_dates.extend(datasets['student_registration']['date_registration'])
            all_dates.extend(datasets['student_registration']['date_unregistration'].dropna())

        if all_dates:
            date_range = max(all_dates) - min(all_dates)
            if date_range > 365:  # More than one year
                validation_results['warnings'].append({
                    'dataset': 'timeline',
                    'warning': 'extended_timeline',
                    'details': {
                        'duration_days': float(date_range),
                        'start_day': float(min(all_dates)),
                        'end_day': float(max(all_dates))
                    }
                })
                logger.warning(
                    f"Timeline spans {date_range:.1f} days, which may indicate "
                    "data from multiple module presentations"
                )

        return validation_results['is_valid']

    except Exception as e:
        logger.error(f"Error validating data consistency: {str(e)}")
        validation_results['is_valid'] = False
        validation_results['issues'].append({
            'issue': 'validation_error',
            'details': str(e)
        })
        return False

def validate_feature_engineering_inputs(
    data: Dict[str, pd.DataFrame],
    required_columns: Dict[str, List[str]]
) -> bool:
    """
    Validates input data for feature engineering.
    
    Args:
        data: Dictionary of input DataFrames
        required_columns: Dictionary specifying required columns for each DataFrame
        
    Returns:
        bool: True if all inputs are valid
    """
    try:
        for df_name, columns in required_columns.items():
            if df_name not in data:
                logger.error(f"Missing required dataset: {df_name}")
                return False
                
            df = data[df_name]
            missing_cols = [col for col in columns if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns in {df_name}: {missing_cols}")
                return False
                
            # Check for null values in required columns
            null_cols = [col for col in columns if df[col].isnull().any()]
            if null_cols:
                logger.warning(f"Null values found in {df_name} columns: {null_cols}")
        
        return True
    except Exception as e:
        logger.error(f"Error validating feature engineering inputs: {str(e)}")
        return False

def validate_temporal_consistency(
    vle_data: pd.DataFrame,
    assessment_data: Optional[pd.DataFrame] = None,
    submission_data: Optional[pd.DataFrame] = None
) -> Tuple[bool, List[str]]:
    """
    Validates temporal consistency across datasets.
    All date fields represent days relative to module start (day 0).
    Negative values indicate days before module start.
    Typical module duration is one semester (~140 days).
    """
    issues = []
    
    try:
        # Validate VLE timeline
        vle_timeline = {
            'min_date': float(vle_data['date'].min()),
            'max_date': float(vle_data['date'].max()),
            'span': float(vle_data['date'].max() - vle_data['date'].min())
        }
        
        # Check for unreasonable pre-module activity
        # Allow up to 30 days before module start for preparation
        if vle_timeline['min_date'] < -30:
            issues.append(
                f"Unusually early VLE activity detected: {vle_timeline['min_date']:.1f} "
                "days before module start"
            )
        
        # Check activity spans
        # A typical module runs for one semester (~140 days)
        if vle_timeline['span'] > 180:  # Allow some buffer beyond semester length
            issues.append(
                f"VLE activity span ({vle_timeline['span']:.1f} days) exceeds "
                "typical module duration"
            )
        
        # Validate assessment timeline if provided
        if assessment_data is not None:
            assessment_timeline = {
                'min_date': float(assessment_data['date'].min()),
                'max_date': float(assessment_data['date'].max())
            }
            
            # Check assessment scheduling
            if assessment_timeline['min_date'] < -7:  # Assessments shouldn't be due before module starts
                issues.append(
                    f"Assessment scheduled too early: {assessment_timeline['min_date']:.1f} "
                    "days before module start"
                )
            
            # Check for reasonable assessment spacing
            assessment_dates = sorted(assessment_data['date'].unique())
            for i in range(len(assessment_dates) - 1):
                gap = assessment_dates[i+1] - assessment_dates[i]
                if gap < 6:  # Less than 6 days between assessments
                    issues.append(
                        f"Short gap between assessments: {gap:.1f} days between "
                        f"days {assessment_dates[i]:.1f} and {assessment_dates[i+1]:.1f}"
                    )
        
        # Validate submission timeline if provided
        if submission_data is not None and assessment_data is not None:
            # Merge assessment due dates with submissions
            merged_data = pd.merge(
                submission_data,
                assessment_data[['id_assessment', 'date']],
                on='id_assessment'
            )
            
            # Check submission timing
            submission_delay = merged_data['date_submitted'] - merged_data['date']
            early_submissions = submission_delay[submission_delay < -1]  # Allow 1 day early
            very_late_submissions = submission_delay[submission_delay > 30]
            
            if len(early_submissions) > 0:
                issues.append(
                    f"Found {len(early_submissions)} submissions more than 1 day "
                    "before assessment date"
                )
            
            if len(very_late_submissions) > 0:
                issues.append(
                    f"Found {len(very_late_submissions)} submissions more than "
                    "30 days after due date"
                )
        
        return len(issues) == 0, issues
        
    except Exception as e:
        logger.error(f"Error validating temporal consistency: {str(e)}")
        issues.append(f"Validation error: {str(e)}")
        return False, issues

def validate_activity_patterns(
    vle_data: pd.DataFrame,
    assessment_data: Optional[pd.DataFrame] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates activity patterns for reasonableness and consistency.
    
    Args:
        vle_data: DataFrame containing VLE interactions
        assessment_data: Optional DataFrame containing assessments
        
    Returns:
        Tuple of (is_valid, validation results)
    """
    results = {
        'is_valid': True,
        'warnings': [],
        'activity_metrics': {}
    }
    
    try:
        # Analyze daily activity patterns
        daily_activity = vle_data.groupby('date')['sum_click'].agg(['sum', 'count'])
        
        results['activity_metrics'].update({
            'avg_daily_activities': float(daily_activity['count'].mean()),
            'std_daily_activities': float(daily_activity['count'].std()),
            'zero_activity_days': int((daily_activity['count'] == 0).sum()),
            'max_daily_activities': int(daily_activity['count'].max())
        })
        
        # Check for unusual patterns
        if results['activity_metrics']['zero_activity_days'] > 30:
            results['warnings'].append(
                f"High number of zero-activity days: "
                f"{results['activity_metrics']['zero_activity_days']}"
            )
        
        if results['activity_metrics']['std_daily_activities'] > results['activity_metrics']['avg_daily_activities'] * 2:
            results['warnings'].append("Highly variable daily activity detected")
        
        # Analyze student-level patterns
        student_activity = vle_data.groupby('id_student').agg({
            'date': ['nunique', 'min', 'max'],
            'sum_click': ['sum', 'mean']
        })
        
        student_activity['activity_span'] = (
            student_activity['date']['max'] - student_activity['date']['min']
        )
        
        results['activity_metrics']['student_patterns'] = {
            'avg_active_days': float(student_activity['date']['nunique'].mean()),
            'avg_activity_span': float(student_activity['activity_span'].mean()),
            'inactive_students': int(
                (student_activity['date']['nunique'] < 10).sum()
            )
        }
        
        # Check for concerning student patterns
        if results['activity_metrics']['student_patterns']['inactive_students'] > 0:
            results['warnings'].append(
                f"Found {results['activity_metrics']['student_patterns']['inactive_students']} "
                "students with very low activity"
            )
        
        # If assessment data provided, check activity around assessments
        if assessment_data is not None:
            assessment_activity = []
            for assessment_date in assessment_data['date'].unique():
                window_activity = vle_data[
                    (vle_data['date'] >= assessment_date - 7) &
                    (vle_data['date'] <= assessment_date)
                ]
                
                assessment_activity.append({
                    'assessment_date': int(assessment_date),
                    'pre_assessment_activity': len(window_activity),
                    'active_students': window_activity['id_student'].nunique()
                })
            
            results['activity_metrics']['assessment_patterns'] = assessment_activity
            
            # Check for assessments with low pre-assessment activity
            low_activity_assessments = [
                a for a in assessment_activity
                if a['pre_assessment_activity'] < results['activity_metrics']['avg_daily_activities']
            ]
            
            if low_activity_assessments:
                results['warnings'].append(
                    f"Found {len(low_activity_assessments)} assessments with "
                    "unusually low pre-assessment activity"
                )
        
        results['is_valid'] = len(results['warnings']) == 0
        return results['is_valid'], results
        
    except Exception as e:
        logger.error(f"Error validating activity patterns: {str(e)}")
        results['warnings'].append(f"Validation error: {str(e)}")
        results['is_valid'] = False
        return False, results

def validate_demographic_fairness(
    activity_data: pd.DataFrame,
    demographic_data: pd.DataFrame,
    protected_attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validates fairness of activity patterns across demographic groups.
    
    Args:
        activity_data: DataFrame containing activity data
        demographic_data: DataFrame containing demographic information
        protected_attributes: Optional list of protected attributes to check
        
    Returns:
        Dictionary containing fairness validation results
    """
    protected_attributes = protected_attributes or PROTECTED_ATTRIBUTES
    results = {
        'is_fair': True,
        'disparities': {},
        'warnings': []
    }
    
    try:
        # Merge activity and demographic data
        merged_data = pd.merge(
            activity_data,
            demographic_data[['id_student'] + list(protected_attributes)],
            on='id_student'
        )
        
        # Calculate activity metrics by student
        student_metrics = merged_data.groupby('id_student').agg({
            'date': ['nunique', 'min', 'max'],
            'sum_click': ['sum', 'mean']
        })
        
        student_metrics['activity_span'] = (
            student_metrics['date']['max'] - student_metrics['date']['min']
        )
        
        # Analyze disparities across protected groups
        for attr in protected_attributes:
            group_metrics = {}
            
            for metric in ['active_days', 'total_clicks', 'activity_span']:
                if metric == 'active_days':
                    group_stats = student_metrics['date']['nunique']
                elif metric == 'total_clicks':
                    group_stats = student_metrics['sum_click']['sum']
                else:  # activity_span
                    group_stats = student_metrics['activity_span']
                
                group_means = merged_data.groupby(attr)['id_student'].apply(
                    lambda x: group_stats[x].mean()
                )
                
                max_mean = group_means.max()
                min_mean = group_means.min()
                disparity = (max_mean - min_mean) / max_mean if max_mean != 0 else 0
                
                group_metrics[metric] = {
                    'disparity': float(disparity),
                    'group_means': group_means.to_dict(),
                    'threshold_exceeded': disparity > FAIRNESS['threshold']
                }
                
                if disparity > FAIRNESS['threshold']:
                    results['warnings'].append(
                        f"High disparity in {metric} for {attr}: {disparity:.3f}"
                    )
            
            results['disparities'][attr] = group_metrics
        
        results['is_fair'] = len(results['warnings']) == 0
        return results
        
    except Exception as e:
        logger.error(f"Error validating demographic fairness: {str(e)}")
        results['warnings'].append(f"Validation error: {str(e)}")
        results['is_fair'] = False
        return results