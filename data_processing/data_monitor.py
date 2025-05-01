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
    data: pd.DataFrame,
    protected_cols: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Detects data quality issues with focus on protected attributes.
    
    Args:
        data: DataFrame to analyze
        protected_cols: List of protected attribute columns
        output_dir: Directory to save the report
        logger: Logger instance for logging messages
        
    Returns:
        Dictionary with data quality report
    """
    logger = logger or logging.getLogger('edupredict')
    protected_cols = protected_cols or FAIRNESS['protected_attributes']
    output_dir = output_dir or DIRS['reports']
    
    try:
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum() / 1024**2  # MB
            },
            'quality_metrics': {
                'completeness': {},
                'validity': {},
                'consistency': {},
                'protected_attributes': {}
            },
            'recommendations': []
        }

        # Analyze protected attributes first
        for col in protected_cols:
            if col in data.columns:
                col_stats = {
                    'missing_rate': data[col].isnull().mean(),
                    'value_counts': data[col].value_counts(normalize=True).to_dict(),
                    'unique_values': data[col].nunique(),
                    'issues': []
                }
                
                # Check against known valid values
                if col in PROTECTED_ATTRIBUTES:
                    valid_values = PROTECTED_ATTRIBUTES[col]['values']
                    invalid_values = [
                        v for v in data[col].unique() 
                        if v not in valid_values and pd.notna(v)
                    ]
                    if invalid_values:
                        col_stats['issues'].append({
                            'type': 'invalid_values',
                            'details': invalid_values
                        })
                
                # Check group sizes
                group_sizes = data[col].value_counts()
                small_groups = group_sizes[group_sizes < FAIRNESS['min_group_size']]
                if not small_groups.empty:
                    col_stats['issues'].append({
                        'type': 'small_groups',
                        'details': small_groups.to_dict()
                    })
                
                report['quality_metrics']['protected_attributes'][col] = col_stats
                
                # Add recommendations if needed
                if col_stats['missing_rate'] > 0:
                    report['recommendations'].append(
                        f"Handle missing values in protected attribute {col} "
                        f"({col_stats['missing_rate']:.1%} missing)"
                    )
                if 'issues' in col_stats and col_stats['issues']:
                    report['recommendations'].append(
                        f"Address quality issues in protected attribute {col}: "
                        f"{[issue['type'] for issue in col_stats['issues']]}"
                    )

        # General completeness checks
        missing_stats = data.isnull().sum()
        report['quality_metrics']['completeness'] = {
            'missing_values': missing_stats[missing_stats > 0].to_dict(),
            'missing_rate': (missing_stats / len(data)).to_dict()
        }

        # Validity checks for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        report['quality_metrics']['validity']['numeric_ranges'] = {
            col: {
                'min': data[col].min(),
                'max': data[col].max(),
                'mean': data[col].mean(),
                'std': data[col].std()
            } for col in numeric_cols
        }

        # Consistency checks
        categorical_cols = data.select_dtypes(include=['category', 'object']).columns
        report['quality_metrics']['consistency']['value_counts'] = {
            col: data[col].value_counts().to_dict()
            for col in categorical_cols if col not in protected_cols
        }

        # Check for duplicate rows
        duplicates = data.duplicated().sum()
        report['quality_metrics']['consistency']['duplicate_rows'] = duplicates

        # Generate summary
        report['summary'] = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'complete_columns': sum(missing_stats == 0),
            'columns_with_missing': sum(missing_stats > 0),
            'duplicate_rows': duplicates,
            'protected_attributes_with_issues': sum(
                1 for col_stats in report['quality_metrics']['protected_attributes'].values()
                if col_stats['issues']
            )
        }

        # Save report
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = output_dir / 'data_quality_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Generated data quality report at {report_path}")

        return report

    except Exception as e:
        logger.error(f"Error generating data quality report: {str(e)}")
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