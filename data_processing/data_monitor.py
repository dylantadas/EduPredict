import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import json
import os
from pathlib import Path
from collections import defaultdict
from config import DATA_PROCESSING, DIRS, FEATURE_ENGINEERING, FAIRNESS, PROTECTED_ATTRIBUTES

logger = logging.getLogger('edupredict')

def check_distribution_shifts(
    new_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    columns: List[str] = None,
    protected_cols: Optional[List[str]] = None,
    threshold: float = None,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Checks for distribution shifts with special attention to protected attributes.
    
    Args:
        new_data: New dataset to check
        reference_data: Reference dataset for comparison
        columns: List of columns to check (default: all numeric columns)
        protected_cols: List of protected attribute columns
        threshold: Threshold for significant shift detection
        output_dir: Directory to save the report
        logger: Logger instance for logging messages
        
    Returns:
        Dictionary with shift detection results"""
    logger = logger or logging.getLogger('edupredict')
    threshold = threshold or FAIRNESS['threshold']
    protected_cols = protected_cols or FAIRNESS['protected_attributes']
    output_dir = output_dir or DIRS['reports']

    try:
        results = {
            'distribution_shifts': {},
            'protected_attribute_shifts': {},
            'summary': {
                'significant_shifts': [],
                'protected_shifts': [],
                'shift_detected': False
            }
        }

        # First check protected attributes
        for col in protected_cols:
            if col in new_data.columns and col in reference_data.columns:
                ref_dist = reference_data[col].value_counts(normalize=True)
                new_dist = new_data[col].value_counts(normalize=True)
                
                # Calculate distribution differences
                diffs = {
                    cat: abs(ref_dist.get(cat, 0) - new_dist.get(cat, 0))
                    for cat in set(ref_dist.index) | set(new_dist.index)
                }
                max_diff = max(diffs.values())
                
                results['protected_attribute_shifts'][col] = {
                    'max_difference': max_diff,
                    'category_differences': diffs,
                    'shift_detected': max_diff > threshold
                }
                
                if max_diff > threshold:
                    results['summary']['protected_shifts'].append(col)
                    logger.warning(
                        f"Significant shift detected in protected attribute {col}: "
                        f"max difference = {max_diff:.3f}"
                    )

        # Then check other columns
        if columns is None:
            columns = new_data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in new_data.columns or col not in reference_data.columns:
                logger.warning(f"Column {col} not found in both datasets")
                continue

            # Calculate basic statistics
            ref_stats = reference_data[col].describe()
            new_stats = new_data[col].describe()
            
            # Perform statistical tests
            ks_statistic, p_value = stats.ks_2samp(
                reference_data[col].dropna(),
                new_data[col].dropna()
            )
            
            # Calculate relative changes
            rel_changes = {
                'mean': abs(new_stats['mean'] - ref_stats['mean']) / abs(ref_stats['mean']) if ref_stats['mean'] != 0 else 0,
                'std': abs(new_stats['std'] - ref_stats['std']) / abs(ref_stats['std']) if ref_stats['std'] != 0 else 0,
                'median': abs(new_stats['50%'] - ref_stats['50%']) / abs(ref_stats['50%']) if ref_stats['50%'] != 0 else 0
            }
            
            # Check shifts by protected groups
            group_shifts = {}
            for prot_col in protected_cols:
                if prot_col in new_data.columns and prot_col in reference_data.columns:
                    group_shifts[prot_col] = {}
                    for group in set(reference_data[prot_col].unique()) | set(new_data[prot_col].unique()):
                        ref_group = reference_data[reference_data[prot_col] == group][col].dropna()
                        new_group = new_data[new_data[prot_col] == group][col].dropna()
                        
                        if len(ref_group) > 0 and len(new_group) > 0:
                            _, group_p_value = stats.ks_2samp(ref_group, new_group)
                            group_shifts[prot_col][group] = {
                                'p_value': group_p_value,
                                'shift_detected': group_p_value < 0.05
                            }
            
            shift_detected = (p_value < 0.05) or any(change > threshold for change in rel_changes.values())
            
            results['distribution_shifts'][col] = {
                'ks_test': {
                    'statistic': ks_statistic,
                    'p_value': p_value
                },
                'relative_changes': rel_changes,
                'protected_group_shifts': group_shifts,
                'shift_detected': shift_detected
            }
            
            if shift_detected:
                results['summary']['significant_shifts'].append(col)
                results['summary']['shift_detected'] = True

        # Save results if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / 'distribution_shift_report.json'
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved distribution shift report to {output_path}")

        return results

    except Exception as e:
        logger.error(f"Error checking distributions: {str(e)}")
        raise

def detect_data_quality_issues(
    datasets: Dict[str, pd.DataFrame],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Detects potential data quality issues, especially focusing on temporal consistency.
    
    Args:
        datasets: Dictionary of DataFrames to check
        logger: Optional logger instance
        
    Returns:
        Dictionary containing quality metrics, issues found and recommendations
    """
    logger = logger or logging.getLogger('edupredict')
    issues = defaultdict(list)
    recommendations = []
    quality_metrics = {
        'completeness': {},
        'validity': {},
        'consistency': {},
        'protected_attributes': {}
    }
    
    try:
        # Check VLE data temporal consistency
        if 'vle_interactions' in datasets:
            vle_data = datasets['vle_interactions']
            
            # Check date range and distribution
            date_range = vle_data['date'].agg(['min', 'max'])
            if date_range['min'] < -60:  # More than 60 days before module start
                msg = f"Unusually early activity detected: {date_range['min']} days before module start"
                issues['vle_interactions'].append(msg)
                recommendations.append(f"Review and validate early VLE activity data before module start")
            
            # Check for activity gaps
            vle_by_student = vle_data.sort_values(['id_student', 'date'])
            time_gaps = vle_by_student.groupby('id_student')['date'].diff()
            large_gaps = time_gaps[time_gaps > 30]  # Gaps > 30 days
            if not large_gaps.empty:
                msg = f"Found {len(large_gaps)} large activity gaps (>30 days)"
                issues['vle_interactions'].append(msg)
                recommendations.append("Consider investigating patterns in activity gaps")
            
            # Add activity metrics to quality metrics
            quality_metrics['completeness']['vle_activity'] = {
                'total_days': len(vle_data['date'].unique()),
                'total_interactions': len(vle_data),
                'students_with_activity': vle_data['id_student'].nunique()
            }
            
            # Check for unusual activity patterns
            activity_by_day = vle_data.groupby('date')['sum_click'].sum()
            zero_activity_days = activity_by_day[activity_by_day == 0]
            if len(zero_activity_days) > 0:
                msg = f"Found {len(zero_activity_days)} days with zero activity"
                issues['vle_interactions'].append(msg)
                recommendations.append("Investigate days with zero activity to ensure data completeness")
        
        # Check assessment submission timing
        if 'student_assessments' in datasets and 'assessments' in datasets:
            assessments = datasets['assessments']
            submissions = datasets['student_assessments']
            
            merged = pd.merge(
                submissions,
                assessments[['id_assessment', 'date']],
                on='id_assessment'
            )
            
            # Check for submissions before due dates
            early_submissions = merged[merged['date_submitted'] < merged['date']]
            if len(early_submissions) > 0:
                msg = f"Found {len(early_submissions)} submissions before assessment date"
                issues['student_assessments'].append(msg)
                recommendations.append("Validate assessment submission dates against due dates")
            
            # Add assessment metrics to quality metrics
            quality_metrics['completeness']['assessments'] = {
                'total_submissions': len(submissions),
                'students_with_submissions': submissions['id_student'].nunique(),
                'submission_rate': len(submissions) / len(assessments) if len(assessments) > 0 else 0
            }
            
            # Check for extremely late submissions
            very_late = merged[merged['date_submitted'] - merged['date'] > 30]
            if len(very_late) > 0:
                msg = f"Found {len(very_late)} submissions more than 30 days late"
                issues['student_assessments'].append(msg)
                recommendations.append("Review policy for handling extremely late submissions")
        
        # Check for timeline inconsistencies across datasets
        if all(k in datasets for k in ['vle_interactions', 'student_assessments', 'assessments']):
            vle_timeline = datasets['vle_interactions']['date']
            assessment_timeline = datasets['assessments']['date']
            submission_timeline = datasets['student_assessments']['date_submitted']
            
            # Create overall timeline bounds
            min_date = min(vle_timeline.min(), assessment_timeline.min())
            max_date = max(
                vle_timeline.max(),
                assessment_timeline.max(),
                submission_timeline.max()
            )
            
            # Add timeline metrics
            quality_metrics['consistency']['timeline'] = {
                'start_date': int(min_date),
                'end_date': int(max_date),
                'duration_days': int(max_date - min_date)
            }
            
            # Check for activity outside module boundaries
            module_length = 365  # Assume maximum module length of 1 year
            if max_date - min_date > module_length:
                msg = f"Activity span ({max_date - min_date} days) exceeds expected module length"
                issues['timeline'].append(msg)
                recommendations.append("Review and validate activity timestamps that exceed expected module duration")
            
            # Check for synchronized starting points
            if abs(vle_timeline.min() - assessment_timeline.min()) > 30:
                msg = "VLE and assessment timelines have significantly different start points"
                issues['timeline'].append(msg)
                recommendations.append("Investigate misalignment between VLE and assessment timelines")
        
        # Check protected attributes if present in student_info
        if 'student_info' in datasets:
            protected_metrics = {}
            for attr in PROTECTED_ATTRIBUTES:
                if attr in datasets['student_info'].columns:
                    attr_data = datasets['student_info'][attr]
                    null_count = attr_data.isnull().sum()
                    if null_count > 0:
                        msg = f"Found {null_count} missing values in protected attribute {attr}"
                        issues['demographics'].append(msg)
                        recommendations.append(f"Address missing values in protected attribute {attr} using fairness-aware imputation")
                    
                    # Add protected attribute metrics
                    protected_metrics[attr] = {
                        'missing_rate': null_count / len(attr_data),
                        'value_counts': attr_data.value_counts(normalize=True).to_dict(),
                        'unique_values': attr_data.nunique()
                    }
            
            quality_metrics['protected_attributes'] = protected_metrics

        # Add validity metrics for numeric columns
        for name, df in datasets.items():
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                quality_metrics['validity'][name] = {
                    col: df[col].describe().to_dict()
                    for col in numeric_cols
                }

        return {
            'quality_metrics': quality_metrics,
            'issues': dict(issues),
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Error detecting data quality issues: {str(e)}")
        raise

def analyze_temporal_patterns(
    vle_data: pd.DataFrame,
    assessment_data: Optional[pd.DataFrame] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Analyzes temporal patterns in student activity data.
    All date fields represent days relative to module start (day 0).
    Negative values indicate days before module start.
    
    Args:
        vle_data: DataFrame containing VLE interaction data
        assessment_data: Optional DataFrame containing assessment data
        logger: Optional logger instance
        
    Returns:
        Dictionary containing temporal pattern analysis
    """
    logger = logger or logging.getLogger('edupredict')
    patterns = {}
    
    try:
        # Analyze pre-module engagement (negative days)
        pre_module = vle_data[vle_data['date'] < 0]
        patterns['pre_module_engagement'] = {
            'total_activities': len(pre_module),
            'unique_students': pre_module['id_student'].nunique(),
            'avg_activities_per_student': len(pre_module) / pre_module['id_student'].nunique()
            if pre_module['id_student'].nunique() > 0 else 0
        }
        
        # Analyze activity distribution
        vle_by_day = vle_data.groupby('date')['sum_click'].agg(['sum', 'count'])
        patterns['daily_patterns'] = {
            'avg_daily_activities': float(vle_by_day['count'].mean()),
            'std_daily_activities': float(vle_by_day['count'].std()),
            'peak_activity_day': float(vle_by_day['sum'].idxmax()),
            'zero_activity_days': int((vle_by_day['count'] == 0).sum())
        }
        
        # Analyze student engagement consistency
        student_activity = vle_data.groupby('id_student').agg({
            'date': ['min', 'max', 'count'],
            'sum_click': 'sum'
        })
        
        # Calculate student-level metrics
        patterns['student_engagement'] = {
            'avg_engagement_span': float(
                (student_activity['date']['max'] - student_activity['date']['min']).mean()
            ),
            'avg_activities_per_student': float(student_activity['date']['count'].mean()),
            'engagement_consistency': float(student_activity['date']['count'].std())
        }
        
        # If assessment data provided, analyze temporal relationships
        if assessment_data is not None:
            assessment_dates = assessment_data['date'].unique()
            assessment_patterns = []
            
            for assessment_date in assessment_dates:
                # Look at 7-day window before assessment
                window_start = assessment_date - 7
                window_end = assessment_date
                
                window_activity = vle_data[
                    (vle_data['date'] >= window_start) &
                    (vle_data['date'] <= window_end)
                ]
                
                if len(window_activity) > 0:
                    avg_daily_activity = len(window_activity) / 8  # 7 days + assessment day
                    assessment_patterns.append({
                        'assessment_day': float(assessment_date),
                        'pre_assessment_activity': len(window_activity),
                        'avg_daily_activity': float(avg_daily_activity),
                        'unique_active_students': window_activity['id_student'].nunique()
                    })
            
            patterns['assessment_patterns'] = assessment_patterns
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error analyzing temporal patterns: {str(e)}")
        raise

def detect_anomalies(
    data: pd.DataFrame,
    feature_cols: List[str],
    protected_cols: Optional[List[str]] = None,
    method: str = 'isolation_forest',
    contamination: float = 0.1,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Detects anomalies with fairness considerations.
    
    Args:
        data: DataFrame to analyze
        feature_cols: List of feature columns for anomaly detection
        protected_cols: List of protected attribute columns
        method: Anomaly detection method ('isolation_forest' or 'robust_covariance')
        contamination: Expected proportion of outliers in the data
        output_dir: Directory to save the report
        logger: Logger instance for logging messages
        
    Returns:
        DataFrame with anomaly scores and labels"""
    logger = logger or logging.getLogger('edupredict')
    protected_cols = protected_cols or FAIRNESS['protected_attributes']
    output_dir = output_dir or DIRS['reports']

    try:
        # Prepare feature matrix
        X = data[feature_cols].copy()
        X = X.fillna(X.mean())
        
        # Track anomaly rates by protected group
        group_anomaly_rates = {}
        
        # Detect anomalies for each protected group separately
        for col in protected_cols:
            if col in data.columns:
                group_anomaly_rates[col] = {}
                
                for group in data[col].unique():
                    group_mask = data[col] == group
                    group_X = X[group_mask]
                    
                    if len(group_X) < 10:  # Skip very small groups
                        logger.warning(f"Group {group} in {col} too small for anomaly detection")
                        continue
                    
                    # Initialize detector
                    if method == 'isolation_forest':
                        detector = IsolationForest(
                            contamination=contamination,
                            random_state=42
                        )
                    elif method == 'robust_covariance':
                        detector = EllipticEnvelope(
                            contamination=contamination,
                            random_state=42
                        )
                    else:
                        raise ValueError(f"Unknown anomaly detection method: {method}")
                    
                    # Fit and predict
                    group_scores = detector.fit_predict(group_X)
                    anomaly_rate = (group_scores == -1).mean()
                    group_anomaly_rates[col][group] = anomaly_rate
                    
                    # Check for significant differences
                    if abs(anomaly_rate - contamination) > FAIRNESS['threshold']:
                        logger.warning(
                            f"Anomaly rate for {col}={group} ({anomaly_rate:.3f}) "
                            f"differs significantly from expected rate ({contamination})"
                        )
        
        # Global anomaly detection
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
        elif method == 'robust_covariance':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
        
        scores = detector.fit_predict(X)
        
        # Add results to original data
        result_df = data.copy()
        result_df['anomaly_score'] = scores
        result_df['is_anomaly'] = scores == -1
        
        # Calculate feature contributions for anomalies
        if method == 'isolation_forest':
            anomaly_mask = result_df['is_anomaly']
            if anomaly_mask.any():
                feature_scores = pd.DataFrame(
                    index=feature_cols,
                    columns=['contribution_score']
                )
                
                for feature in feature_cols:
                    z_scores = np.abs(stats.zscore(X.loc[anomaly_mask, feature]))
                    feature_scores.loc[feature, 'contribution_score'] = z_scores.mean()
                
                # Normalize scores
                feature_scores['contribution_score'] = (
                    feature_scores['contribution_score'] / 
                    feature_scores['contribution_score'].sum()
                )
                
                result_df['anomaly_details'] = result_df.apply(
                    lambda row: feature_scores.to_dict()['contribution_score']
                    if row['is_anomaly'] else None,
                    axis=1
                )
        
        # Save results and fairness analysis
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save anomaly detection results
            result_df[['anomaly_score', 'is_anomaly']].to_parquet(
                output_dir / 'anomaly_scores.parquet'
            )
            
            # Save feature contributions
            if 'anomaly_details' in result_df.columns:
                with open(output_dir / 'anomaly_details.json', 'w') as f:
                    json.dump(
                        result_df[result_df['is_anomaly']]['anomaly_details'].to_dict(),
                        f,
                        indent=2
                    )
            
            # Save group-level analysis
            fairness_report = {
                'group_anomaly_rates': group_anomaly_rates,
                'overall_rate': contamination,
                'threshold': FAIRNESS['threshold'],
                'method': method,
                'groups_with_significant_differences': [
                    f"{col}={group}"
                    for col, groups in group_anomaly_rates.items()
                    for group, rate in groups.items()
                    if abs(rate - contamination) > FAIRNESS['threshold']
                ]
            }
            
            with open(output_dir / 'anomaly_fairness_report.json', 'w') as f:
                json.dump(fairness_report, f, indent=2)
            
            logger.info(f"Saved anomaly detection results to {output_dir}")

        return result_df

    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise