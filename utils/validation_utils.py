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

def validate_data_consistency(datasets: Dict[str, pd.DataFrame]) -> bool:
    """
    Validates consistency between different datasets.
    
    Args:
        datasets: Dictionary of pandas DataFrames
        
    Returns:
        bool: True if data is consistent
    """
    try:
        warnings = []
        
        # Get reference sets
        student_ids = set(datasets['student_info']['id_student'])
        
        # Check VLE data consistency
        if 'vle_interactions' in datasets:
            vle_student_ids = set(datasets['vle_interactions']['id_student'])
            if not vle_student_ids.issubset(student_ids):
                warnings.append("VLE interactions contain unknown student IDs")
                
            if 'vle_materials' in datasets:
                vle_ids = set(datasets['vle_materials']['id_site'])
                interaction_vle_ids = set(datasets['vle_interactions']['id_site'])
                if not interaction_vle_ids.issubset(vle_ids):
                    warnings.append("VLE interactions contain unknown activity IDs")
        
        # Check assessment data consistency
        if all(k in datasets for k in ['assessments', 'student_assessments']):
            assessment_ids = set(datasets['assessments']['id_assessment'])
            student_assessment_ids = set(datasets['student_assessments']['id_assessment'])
            if not student_assessment_ids.issubset(assessment_ids):
                warnings.append("Student assessment data contain unknown assessment IDs")
                
            assessment_student_ids = set(datasets['student_assessments']['id_student'])
            if not assessment_student_ids.issubset(student_ids):
                warnings.append("Assessment data contain unknown student IDs")
        
        # Log any warnings
        for warning in warnings:
            logger.warning(warning)
            
        # Return True even with warnings, as they may not be critical
        return True
    except Exception as e:
        logger.error(f"Error validating data consistency: {str(e)}")
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
    Ensures proper interpretation of dates relative to module start.
    
    Args:
        vle_data: DataFrame containing VLE interactions
        assessment_data: Optional DataFrame containing assessments
        submission_data: Optional DataFrame containing submissions
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    try:
        # Validate VLE timeline
        vle_timeline = {
            'min_date': vle_data['date'].min(),
            'max_date': vle_data['date'].max(),
            'span': vle_data['date'].max() - vle_data['date'].min()
        }
        
        # Check for unreasonable pre-module activity
        if vle_timeline['min_date'] < -90:  # More than 90 days before module start
            issues.append(
                f"Unusually early VLE activity detected: {vle_timeline['min_date']} "
                "days before module start"
            )
        
        # Check activity spans
        if vle_timeline['span'] > 365:  # More than a year
            issues.append(
                f"VLE activity span ({vle_timeline['span']} days) exceeds "
                "expected module duration"
            )
        
        # Validate assessment timeline if provided
        if assessment_data is not None:
            assessment_timeline = {
                'min_date': assessment_data['date'].min(),
                'max_date': assessment_data['date'].max()
            }
            
            # Check assessment scheduling
            if assessment_timeline['min_date'] < -30:  # Assessments >30 days before start
                issues.append(
                    f"Assessment scheduled too early: {assessment_timeline['min_date']} "
                    "days before module start"
                )
            
            # Check alignment with VLE activity
            assessment_dates = assessment_data['date'].unique()
            for assessment_date in assessment_dates:
                window_start = assessment_date - 14
                window_end = assessment_date
                
                # Check for sufficient pre-assessment activity
                pre_assessment_activity = vle_data[
                    (vle_data['date'] >= window_start) &
                    (vle_data['date'] <= window_end)
                ]
                
                if len(pre_assessment_activity) == 0:
                    issues.append(
                        f"No VLE activity detected in two weeks before "
                        f"assessment on day {assessment_date}"
                    )
        
        # Validate submission timeline if provided
        if submission_data is not None and assessment_data is not None:
            merged_data = pd.merge(
                submission_data,
                assessment_data[['id_assessment', 'date']],
                on='id_assessment'
            )
            
            # Check submission timing
            submission_delay = merged_data['date_submitted'] - merged_data['date']
            early_submissions = submission_delay[submission_delay < 0]
            very_late_submissions = submission_delay[submission_delay > 30]
            
            if len(early_submissions) > 0:
                issues.append(
                    f"Found {len(early_submissions)} submissions before "
                    "assessment date"
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