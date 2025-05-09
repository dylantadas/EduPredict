import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import itertools
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

from config import FAIRNESS, PROTECTED_ATTRIBUTES, EVALUATION, DIRS

# Set up the logger
logger = logging.getLogger('edupredict2')

def calculate_group_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray, 
    group_values: np.ndarray,
    metrics: Optional[List[str]] = None,
    min_group_size: int = 10,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Calculates performance metrics for each demographic group.
    
    Args:
        y_true: True target values
        y_pred: Predicted binary values
        y_prob: Predicted probabilities
        group_values: Group values for each sample
        metrics: List of metrics to calculate (None = all)
        min_group_size: Minimum group size to include
        logger: Logger for tracking metric calculation
        
    Returns:
        DataFrame with metrics by group
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    # Use default metrics from config if none specified
    if metrics is None:
        metrics = EVALUATION.get('metrics', ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    logger.info(f"Calculating group metrics for {len(np.unique(group_values))} groups...")
    
    # List to store results for each group
    group_metrics_list = []
    
    # Calculate metrics for each group
    for group in np.unique(group_values):
        # Filter by group
        group_indices = np.where(group_values == group)[0]
        
        # Skip if group size is below minimum threshold
        if len(group_indices) < min_group_size:
            logger.warning(f"Group '{group}' has only {len(group_indices)} samples, which is below the minimum threshold of {min_group_size}. Skipping.")
            continue
        
        # Filter data for this group
        group_y_true = y_true[group_indices]
        group_y_pred = y_pred[group_indices]
        group_y_prob = y_prob[group_indices]
        
        # Skip if group has only one class (can't compute some metrics)
        if len(np.unique(group_y_true)) < 2:
            logger.warning(f"Group '{group}' has only one class in true values. Skipping.")
            continue
        
        # Calculate metrics for this group
        group_result = {
            'group': group,
            'count': len(group_indices),
            'positive_rate': np.mean(group_y_pred)
        }
        
        # Calculate selected metrics
        if 'accuracy' in metrics:
            group_result['accuracy'] = accuracy_score(group_y_true, group_y_pred)
            
        if 'precision' in metrics:
            group_result['precision'] = precision_score(group_y_true, group_y_pred, zero_division=0)
            
        if 'recall' in metrics:
            group_result['recall'] = recall_score(group_y_true, group_y_pred, zero_division=0)
            
        if 'f1' in metrics:
            group_result['f1'] = f1_score(group_y_true, group_y_pred, zero_division=0)
            
        if ('auc' in metrics) or ('auc_roc' in metrics):
            try:
                group_result['auc_roc'] = roc_auc_score(group_y_true, group_y_prob)
            except ValueError:
                # Handle case where there's only one class in the prediction
                logger.warning(f"Could not calculate AUC for group '{group}'. Using 0.5 as default.")
                group_result['auc_roc'] = 0.5
        
        # Calculate true positive rate (TPR) and false positive rate (FPR) for fairness metrics
        cm = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1])
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # True positive rate (TPR) = Recall = TP / (TP + FN)
            if tp + fn > 0:
                group_result['true_positive_rate'] = tp / (tp + fn)
            else:
                group_result['true_positive_rate'] = 0.0
            
            # False positive rate (FPR) = FP / (FP + TN)
            if fp + tn > 0:
                group_result['false_positive_rate'] = fp / (fp + tn)
            else:
                group_result['false_positive_rate'] = 0.0
        else:
            group_result['true_positive_rate'] = 0.0
            group_result['false_positive_rate'] = 0.0
        
        group_metrics_list.append(group_result)
    
    # Convert list to DataFrame
    if group_metrics_list:
        result_df = pd.DataFrame(group_metrics_list)
        logger.info(f"Calculated metrics for {len(result_df)} groups")
        return result_df
    else:
        logger.warning("No groups met the criteria for metrics calculation")
        return pd.DataFrame(columns=['group', 'count', 'positive_rate'])


def calculate_fairness_metrics(
    group_metrics: pd.DataFrame, 
    fairness_metrics: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Calculates fairness metrics from group-level metrics.
    
    Args:
        group_metrics: DataFrame with group metrics
        fairness_metrics: List of fairness metrics to calculate
        logger: Logger for tracking metric calculation
        
    Returns:
        Dictionary of fairness metrics
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    # Default fairness metrics if none specified
    if fairness_metrics is None:
        fairness_metrics = [
            'demographic_parity_difference',
            'disparate_impact_ratio',
            'equal_opportunity_difference',
            'average_odds_difference'
        ]
    
    logger.info(f"Calculating fairness metrics: {', '.join(fairness_metrics)}")
    
    # Ensure required columns are present
    required_columns = ['positive_rate']
    if 'equal_opportunity_difference' in fairness_metrics or 'average_odds_difference' in fairness_metrics:
        required_columns.extend(['true_positive_rate', 'false_positive_rate'])
    
    for col in required_columns:
        if col not in group_metrics.columns:
            logger.error(f"Required column '{col}' not found in group_metrics DataFrame")
            return {'error': f"Missing required column: {col}"}
    
    # Initialize results dictionary
    fairness_results = {}
    
    # Demographic Parity Difference (maximum difference in positive prediction rates)
    if 'demographic_parity_difference' in fairness_metrics:
        try:
            max_positive_rate = group_metrics['positive_rate'].max()
            min_positive_rate = group_metrics['positive_rate'].min()
            fairness_results['demographic_parity_difference'] = max_positive_rate - min_positive_rate
            
            # Add group-specific info 
            fairness_results['max_positive_rate_group'] = group_metrics.loc[group_metrics['positive_rate'].idxmax(), 'group']
            fairness_results['min_positive_rate_group'] = group_metrics.loc[group_metrics['positive_rate'].idxmin(), 'group']
        except Exception as e:
            logger.error(f"Error calculating demographic parity difference: {str(e)}")
            fairness_results['demographic_parity_difference'] = float('nan')
    
    # Disparate Impact Ratio (minimum ratio of positive prediction rates)
    if 'disparate_impact_ratio' in fairness_metrics:
        try:
            max_positive_rate = group_metrics['positive_rate'].max()
            min_positive_rate = group_metrics['positive_rate'].min()
            
            # Avoid division by zero
            if max_positive_rate > 0:
                fairness_results['disparate_impact_ratio'] = min_positive_rate / max_positive_rate
            else:
                fairness_results['disparate_impact_ratio'] = 1.0
                
            # Add group-specific info
            fairness_results['min_impact_ratio_groups'] = [
                group_metrics.loc[group_metrics['positive_rate'].idxmin(), 'group'],
                group_metrics.loc[group_metrics['positive_rate'].idxmax(), 'group']
            ]
        except Exception as e:
            logger.error(f"Error calculating disparate impact ratio: {str(e)}")
            fairness_results['disparate_impact_ratio'] = float('nan')
    
    # Equal Opportunity Difference (maximum difference in true positive rates)
    if 'equal_opportunity_difference' in fairness_metrics:
        try:
            max_tpr = group_metrics['true_positive_rate'].max()
            min_tpr = group_metrics['true_positive_rate'].min()
            fairness_results['equal_opportunity_difference'] = max_tpr - min_tpr
            
            # Add group-specific info
            fairness_results['max_tpr_group'] = group_metrics.loc[group_metrics['true_positive_rate'].idxmax(), 'group']
            fairness_results['min_tpr_group'] = group_metrics.loc[group_metrics['true_positive_rate'].idxmin(), 'group']
        except Exception as e:
            logger.error(f"Error calculating equal opportunity difference: {str(e)}")
            fairness_results['equal_opportunity_difference'] = float('nan')
    
    # Average Odds Difference (average difference in FPR and TPR)
    if 'average_odds_difference' in fairness_metrics:
        try:
            # Calculate maximum difference in true positive rates
            max_tpr = group_metrics['true_positive_rate'].max()
            min_tpr = group_metrics['true_positive_rate'].min()
            tpr_diff = max_tpr - min_tpr
            
            # Calculate maximum difference in false positive rates
            max_fpr = group_metrics['false_positive_rate'].max()
            min_fpr = group_metrics['false_positive_rate'].min()
            fpr_diff = max_fpr - min_fpr
            
            # Average of the two differences
            fairness_results['average_odds_difference'] = (tpr_diff + fpr_diff) / 2
        except Exception as e:
            logger.error(f"Error calculating average odds difference: {str(e)}")
            fairness_results['average_odds_difference'] = float('nan')
    
    # Calculate the statistical parity across groups
    if 'statistical_parity' in fairness_metrics:
        try:
            # Calculate the standard deviation of positive rates across groups
            fairness_results['statistical_parity_std'] = group_metrics['positive_rate'].std()
        except Exception as e:
            logger.error(f"Error calculating statistical parity: {str(e)}")
            fairness_results['statistical_parity_std'] = float('nan')
    
    # Theil index (measure of inequality)
    if 'theil_index' in fairness_metrics:
        try:
            # Weighted average of positive rates 
            overall_rate = np.average(group_metrics['positive_rate'], weights=group_metrics['count'])
            
            # Avoid division by zero
            if overall_rate > 0:
                # Log terms for each group
                log_terms = []
                for _, row in group_metrics.iterrows():
                    rate_ratio = row['positive_rate'] / overall_rate
                    if rate_ratio > 0:  # Avoid log(0)
                        log_term = row['count'] * (row['positive_rate'] / overall_rate) * np.log(row['positive_rate'] / overall_rate)
                        log_terms.append(log_term)
                
                # Calculate Theil index
                theil = sum(log_terms) / sum(group_metrics['count'])
                fairness_results['theil_index'] = theil
            else:
                fairness_results['theil_index'] = 0.0
        except Exception as e:
            logger.error(f"Error calculating Theil index: {str(e)}")
            fairness_results['theil_index'] = float('nan')
    
    logger.info(f"Fairness metrics calculated: {list(fairness_results.keys())}")
    return fairness_results


def evaluate_model_fairness(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray, 
    protected_attributes: Dict[str, np.ndarray],
    thresholds: Optional[Dict[str, float]] = None,
    metrics: Optional[List[str]] = None,
    fairness_metrics: Optional[List[str]] = None,
    min_group_size: int = 10,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict]:
    """
    Evaluates model fairness across multiple protected attributes.
    
    Args:
        y_true: True target values
        y_pred: Predicted binary values
        y_prob: Predicted probabilities
        protected_attributes: Dictionary mapping attribute names to values
        thresholds: Dictionary of fairness thresholds
        metrics: List of performance metrics to calculate
        fairness_metrics: List of fairness metrics to calculate
        min_group_size: Minimum group size to include
        logger: Logger for tracking fairness evaluation
        
    Returns:
        Dictionary of fairness results by attribute
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    # Use default thresholds from config if none specified
    if thresholds is None:
        thresholds = FAIRNESS.get('thresholds', {
            'demographic_parity_difference': 0.1,
            'disparate_impact_ratio': 0.8,
            'equal_opportunity_difference': 0.1,
            'average_odds_difference': 0.1
        })
    
    # Use default group size from config if not specified
    if min_group_size is None:
        min_group_size = FAIRNESS.get('min_group_size', 10)
    
    logger.info(f"Evaluating fairness across {len(protected_attributes)} protected attributes")
    
    # Initialize results dictionary
    fairness_results = {}
    
    # Evaluate fairness for each protected attribute
    for attr_name, attr_values in protected_attributes.items():
        logger.info(f"Evaluating fairness for attribute: {attr_name}")
        
        try:
            # Calculate group metrics
            group_metrics_df = calculate_group_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                group_values=attr_values,
                metrics=metrics,
                min_group_size=min_group_size,
                logger=logger
            )
            
            # Skip if no groups met the criteria
            if len(group_metrics_df) <= 1:
                logger.warning(f"Not enough valid groups for attribute '{attr_name}'. Skipping fairness calculation.")
                fairness_results[attr_name] = {
                    'error': f"Not enough valid groups for attribute '{attr_name}'",
                    'group_metrics': group_metrics_df.to_dict('records') if not group_metrics_df.empty else []
                }
                continue
            
            # Calculate fairness metrics
            fairness_metrics_result = calculate_fairness_metrics(
                group_metrics=group_metrics_df,
                fairness_metrics=fairness_metrics,
                logger=logger
            )
            
            # Check if thresholds are violated
            violations = {}
            for metric, value in fairness_metrics_result.items():
                if metric in thresholds:
                    threshold = thresholds[metric]
                    
                    # Special handling for disparate impact ratio (higher is better)
                    if metric == 'disparate_impact_ratio':
                        if value < threshold:
                            violations[metric] = {
                                'value': value,
                                'threshold': threshold,
                                'violation': True
                            }
                    # For other metrics, lower is better
                    elif value > threshold:
                        violations[metric] = {
                            'value': value,
                            'threshold': threshold,
                            'violation': True
                        }
            
            # Store results for this attribute
            fairness_results[attr_name] = {
                'group_metrics': group_metrics_df.to_dict('records'),
                'fairness_metrics': fairness_metrics_result,
                'violations': violations,
                'has_violations': len(violations) > 0
            }
            
            # Log any violations
            if violations:
                logger.warning(f"Fairness violations detected for {attr_name}: {list(violations.keys())}")
                
        except Exception as e:
            logger.error(f"Error evaluating fairness for attribute '{attr_name}': {str(e)}")
            fairness_results[attr_name] = {'error': str(e)}
    
    logger.info(f"Fairness evaluation completed for {len(fairness_results)} attributes")
    return fairness_results


def analyze_subgroup_fairness(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray, 
    protected_attributes: Dict[str, np.ndarray],
    metrics: List[str] = ['f1', 'accuracy'],
    fairness_metrics: Optional[List[str]] = None,
    min_group_size: int = 30,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Analyzes model fairness for intersectional subgroups.
    
    Args:
        y_true: True target values
        y_pred: Predicted binary values
        y_prob: Predicted probabilities
        protected_attributes: Dictionary mapping attribute names to values
        metrics: List of metrics to calculate
        fairness_metrics: List of fairness metrics to calculate
        min_group_size: Minimum group size to include
        logger: Logger for tracking subgroup analysis
        
    Returns:
        DataFrame with subgroup metrics
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    logger.info("Analyzing intersectional subgroup fairness...")
    
    # Need at least 2 protected attributes for intersectional analysis
    if len(protected_attributes) < 2:
        logger.warning("Intersectional analysis requires at least 2 protected attributes")
        return pd.DataFrame()
    
    # Create subgroup identifier by combining protected attributes
    attr_names = list(protected_attributes.keys())
    
    # Ensure all arrays have the same length
    lengths = [len(arr) for arr in protected_attributes.values()]
    if len(set(lengths)) > 1:
        logger.error("Protected attributes have different lengths")
        return pd.DataFrame()
    
    n_samples = lengths[0]
    
    # Create array to hold subgroup labels
    subgroups = np.empty(n_samples, dtype=object)
    
    # Initialize with first attribute
    for i in range(n_samples):
        subgroups[i] = str(protected_attributes[attr_names[0]][i])
    
    # Append other attributes to create intersectional subgroup labels
    for j in range(1, len(attr_names)):
        attr_name = attr_names[j]
        attr_values = protected_attributes[attr_name]
        
        for i in range(n_samples):
            subgroups[i] = f"{subgroups[i]}_{attr_values[i]}"
    
    logger.info(f"Created {len(np.unique(subgroups))} intersectional subgroups")
    
    # Calculate metrics for each subgroup
    subgroup_metrics = calculate_group_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        group_values=subgroups,
        metrics=metrics,
        min_group_size=min_group_size,
        logger=logger
    )
    
    # Add attribute components to the DataFrame
    if not subgroup_metrics.empty:
        # Split the subgroup identifiers back into individual attributes
        for i, attr_name in enumerate(attr_names):
            subgroup_metrics[attr_name] = subgroup_metrics['group'].apply(
                lambda x: x.split('_')[i] if isinstance(x, str) else None
            )
    
    # Calculate fairness metrics across subgroups
    if fairness_metrics is not None and len(subgroup_metrics) > 1:
        fairness_results = calculate_fairness_metrics(
            group_metrics=subgroup_metrics,
            fairness_metrics=fairness_metrics,
            logger=logger
        )
        
        logger.info(f"Intersectional fairness metrics: {fairness_results}")
        
        # Add fairness metrics to the DataFrame as metadata
        if not subgroup_metrics.empty:
            subgroup_metrics.attrs['fairness_metrics'] = fairness_results
    
    logger.info(f"Subgroup analysis completed with {len(subgroup_metrics)} valid subgroups")
    return subgroup_metrics


def mitigate_bias_with_thresholds(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    protected_attribute: np.ndarray,
    attribute_name: str,
    metric: str = 'demographic_parity',
    tolerance: float = 0.01,
    max_iterations: int = 100,
    initial_threshold: float = 0.5,
    logger: Optional[logging.Logger] = None
) -> Dict[Any, float]:
    """
    Finds group-specific thresholds to mitigate bias.
    
    Args:
        y_true: True target values
        y_prob: Predicted probabilities
        protected_attribute: Group values for each sample
        attribute_name: Name of protected attribute
        metric: Fairness metric to optimize
        tolerance: Tolerance for fairness gap
        max_iterations: Maximum iterations
        initial_threshold: Starting threshold value
        logger: Logger for tracking bias mitigation
        
    Returns:
        Dictionary mapping groups to optimal thresholds
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    logger.info(f"Finding optimal thresholds for {attribute_name} to mitigate {metric} bias")
    
    # Get unique groups
    unique_groups = np.unique(protected_attribute)
    
    if len(unique_groups) < 2:
        logger.warning("At least 2 groups are required for threshold optimization")
        return {unique_groups[0]: initial_threshold}
    
    # Initialize thresholds for each group
    thresholds = {group: initial_threshold for group in unique_groups}
    
    # Binary search parameters
    learning_rate = 0.1
    min_lr = 0.001
    
    # Function to compute positive prediction rates with current thresholds
    def compute_positive_rates():
        group_rates = {}
        for group in unique_groups:
            group_mask = (protected_attribute == group)
            group_preds = (y_prob[group_mask] >= thresholds[group]).astype(int)
            group_rates[group] = np.mean(group_preds)
        return group_rates
    
    # Track progress
    iteration = 0
    best_gap = float('inf')
    best_thresholds = thresholds.copy()
    
    # Iteratively adjust thresholds
    while iteration < max_iterations:
        # Compute positive prediction rates for each group
        positive_rates = compute_positive_rates()
        
        # Get the maximum difference between rates
        if metric == 'demographic_parity':
            max_rate = max(positive_rates.values())
            min_rate = min(positive_rates.values())
            gap = max_rate - min_rate
            
            # Track best solution
            if gap < best_gap:
                best_gap = gap
                best_thresholds = thresholds.copy()
            
            # Check if we've reached the tolerance
            if gap <= tolerance:
                logger.info(f"Demographic parity achieved within tolerance after {iteration} iterations")
                break
            
            # Adjust thresholds
            for group in unique_groups:
                if positive_rates[group] == max_rate:
                    # Increase threshold to reduce positive rate
                    thresholds[group] = min(1.0, thresholds[group] + learning_rate)
                elif positive_rates[group] == min_rate:
                    # Decrease threshold to increase positive rate
                    thresholds[group] = max(0.0, thresholds[group] - learning_rate)
        
        # Equal opportunity (equal true positive rates)
        elif metric == 'equal_opportunity':
            tpr_rates = {}
            for group in unique_groups:
                group_mask = (protected_attribute == group)
                group_true = y_true[group_mask]
                group_probs = y_prob[group_mask]
                
                # Only consider positive instances
                positive_mask = (group_true == 1)
                if sum(positive_mask) > 0:
                    # Calculate true positive rate with current threshold
                    group_preds = (group_probs[positive_mask] >= thresholds[group]).astype(int)
                    tpr_rates[group] = np.mean(group_preds)
                else:
                    tpr_rates[group] = 0.0
            
            # Calculate gap
            if tpr_rates:
                max_tpr = max(tpr_rates.values())
                min_tpr = min(tpr_rates.values())
                gap = max_tpr - min_tpr
                
                # Track best solution
                if gap < best_gap:
                    best_gap = gap
                    best_thresholds = thresholds.copy()
                
                # Check if we've reached the tolerance
                if gap <= tolerance:
                    logger.info(f"Equal opportunity achieved within tolerance after {iteration} iterations")
                    break
                
                # Adjust thresholds
                for group in unique_groups:
                    if tpr_rates[group] == max_tpr:
                        # Increase threshold to reduce TPR
                        thresholds[group] = min(1.0, thresholds[group] + learning_rate)
                    elif tpr_rates[group] == min_tpr:
                        # Decrease threshold to increase TPR
                        thresholds[group] = max(0.0, thresholds[group] - learning_rate)
        
        # Update learning rate
        iteration += 1
        if iteration % 10 == 0:
            learning_rate = max(min_lr, learning_rate * 0.9)
    
    # Use best thresholds found
    if iteration == max_iterations:
        logger.warning(f"Maximum iterations reached. Using best thresholds found with gap: {best_gap:.4f}")
        thresholds = best_thresholds
    
    # Log final thresholds
    logger.info(f"Optimized thresholds for {attribute_name}: {thresholds}")
    
    return thresholds


def compare_fairness_across_models(
    model_results: Dict[str, Dict[str, Dict]],
    protected_attributes: List[str],
    fairness_metrics: List[str] = ['demographic_parity_difference', 'disparate_impact_ratio', 'equal_opportunity_difference'],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Compares fairness metrics across multiple models.
    
    Args:
        model_results: Dictionary mapping model names to fairness results
        protected_attributes: List of protected attributes to compare
        fairness_metrics: List of fairness metrics to compare
        logger: Logger for tracking comparison
        
    Returns:
        DataFrame comparing fairness across models
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    logger.info(f"Comparing fairness across {len(model_results)} models and {len(protected_attributes)} protected attributes")
    
    # List to store comparison rows
    comparison_rows = []
    
    # For each model and protected attribute combination
    for model_name, model_result in model_results.items():
        for attr_name in protected_attributes:
            if attr_name not in model_result:
                logger.warning(f"Protected attribute {attr_name} not found for model {model_name}")
                continue
            
            attr_result = model_result[attr_name]
            
            # Skip if there was an error
            if 'error' in attr_result:
                logger.warning(f"Error in results for model {model_name}, attribute {attr_name}: {attr_result['error']}")
                continue
            
            # Get fairness metrics
            if 'fairness_metrics' not in attr_result:
                logger.warning(f"No fairness metrics found for model {model_name}, attribute {attr_name}")
                continue
            
            fairness_result = attr_result['fairness_metrics']
            
            # Create comparison row
            row = {
                'model': model_name,
                'protected_attribute': attr_name
            }
            
            # Add fairness metrics
            for metric in fairness_metrics:
                if metric in fairness_result:
                    row[metric] = fairness_result[metric]
                else:
                    row[metric] = None
            
            # Add flag for violations
            row['has_violations'] = attr_result.get('has_violations', False)
            
            comparison_rows.append(row)
    
    # Convert to DataFrame
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        
        # Add ranking columns for each fairness metric
        for metric in fairness_metrics:
            if metric in comparison_df.columns:
                if metric == 'disparate_impact_ratio':
                    # For disparate impact, higher is better
                    comparison_df[f'{metric}_rank'] = comparison_df.groupby('protected_attribute')[metric].rank(ascending=False)
                else:
                    # For other metrics, lower is better
                    comparison_df[f'{metric}_rank'] = comparison_df.groupby('protected_attribute')[metric].rank()
        
        logger.info(f"Fairness comparison completed with {len(comparison_df)} entries")
        return comparison_df
    else:
        logger.warning("No valid comparison data found")
        return pd.DataFrame(columns=['model', 'protected_attribute'] + fairness_metrics)


def generate_fairness_report(
    fairness_results: Dict[str, Dict], 
    thresholds: Optional[Dict[str, float]] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Generates detailed fairness evaluation report.
    
    Args:
        fairness_results: Dictionary of fairness results
        thresholds: Dictionary of fairness thresholds
        output_path: Path to save report
        logger: Logger for tracking report generation
        
    Returns:
        Report text or path to saved report
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    # Use default thresholds from config if none specified
    if thresholds is None:
        thresholds = FAIRNESS.get('thresholds', {
            'demographic_parity_difference': 0.1,
            'disparate_impact_ratio': 0.8,
            'equal_opportunity_difference': 0.1,
            'average_odds_difference': 0.1
        })
    
    logger.info("Generating fairness evaluation report")
    
    # Generate report text
    report_lines = []
    report_lines.append("# Fairness Evaluation Report")
    report_lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Overall summary
    total_attributes = len(fairness_results)
    attributes_with_violations = sum(1 for attr_result in fairness_results.values() 
                                    if attr_result.get('has_violations', False))
    
    report_lines.append("## Summary")
    report_lines.append(f"- Total protected attributes evaluated: {total_attributes}")
    report_lines.append(f"- Attributes with fairness violations: {attributes_with_violations}")
    report_lines.append(f"- Fairness thresholds used: {thresholds}\n")
    
    # Detailed results by attribute
    report_lines.append("## Detailed Results by Protected Attribute")
    
    for attr_name, attr_result in fairness_results.items():
        report_lines.append(f"\n### {attr_name}")
        
        # Handle errors
        if 'error' in attr_result:
            report_lines.append(f"Error: {attr_result['error']}")
            continue
        
        # Fairness metrics
        if 'fairness_metrics' in attr_result:
            report_lines.append("\n#### Fairness Metrics")
            fairness_metrics = attr_result['fairness_metrics']
            for metric_name, metric_value in fairness_metrics.items():
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    report_lines.append(f"- {metric_name}: {metric_value:.4f}")
                elif isinstance(metric_value, dict) or isinstance(metric_value, list):
                    # Skip complex values in text report
                    continue
                else:
                    report_lines.append(f"- {metric_name}: {metric_value}")
        
        # Violations
        if 'violations' in attr_result and attr_result['violations']:
            report_lines.append("\n#### Fairness Violations")
            violations = attr_result['violations']
            for metric_name, violation_info in violations.items():
                threshold = violation_info['threshold']
                value = violation_info['value']
                report_lines.append(f"- {metric_name}: {value:.4f} (threshold: {threshold:.4f})")
        
        # Group metrics
        if 'group_metrics' in attr_result and attr_result['group_metrics']:
            report_lines.append("\n#### Group Performance")
            group_metrics = attr_result['group_metrics']
            
            # Create a table header
            metrics_keys = [k for k in group_metrics[0].keys() if k not in ['group', 'count']]
            report_lines.append("| Group | Count | " + " | ".join(metrics_keys) + " |")
            report_lines.append("|-------|-------|" + "-|"*len(metrics_keys))
            
            # Add table rows
            for group_data in group_metrics:
                group = group_data['group']
                count = group_data['count']
                metrics_values = []
                for key in metrics_keys:
                    val = group_data.get(key)
                    if isinstance(val, float):
                        metrics_values.append(f"{val:.4f}")
                    else:
                        metrics_values.append(str(val))
                
                report_lines.append(f"| {group} | {count} | " + " | ".join(metrics_values) + " |")
    
    # Recommendations
    report_lines.append("\n## Recommendations")
    
    if attributes_with_violations > 0:
        report_lines.append("\nBased on the fairness evaluation, the following recommendations are made:")
        
        for attr_name, attr_result in fairness_results.items():
            if attr_result.get('has_violations', False) and 'violations' in attr_result:
                report_lines.append(f"\n### For {attr_name}:")
                
                violations = attr_result['violations']
                for metric_name, violation_info in violations.items():
                    if metric_name == 'demographic_parity_difference':
                        report_lines.append(f"- The model shows significant disparity in prediction rates across {attr_name} groups. Consider:")
                        report_lines.append("  - Applying bias mitigation techniques such as reweighting or thresholding")
                        report_lines.append("  - Retraining with a balanced dataset")
                        report_lines.append("  - Using bias mitigation algorithms during training")
                    
                    elif metric_name == 'disparate_impact_ratio':
                        report_lines.append(f"- The model shows disparate impact with respect to {attr_name}. Consider:")
                        report_lines.append("  - Applying group-specific thresholds")
                        report_lines.append("  - Investigating and addressing data representation issues")
                    
                    elif metric_name == 'equal_opportunity_difference':
                        report_lines.append(f"- True positive rates differ significantly across {attr_name} groups. Consider:")
                        report_lines.append("  - Customizing thresholds by group")
                        report_lines.append("  - Using adversarial debiasing techniques")
                        report_lines.append("  - Exploring post-processing techniques to equalize true positive rates")
                    
                    elif metric_name == 'average_odds_difference':
                        report_lines.append(f"- Both true positive and false positive rates show disparities across {attr_name} groups. Consider:")
                        report_lines.append("  - Using a fairness-aware algorithm that explicitly optimizes for equalized odds")
                        report_lines.append("  - Exploring calibrated equalized odds post-processing")
    else:
        report_lines.append("\nNo fairness violations were detected based on the specified thresholds.")
        report_lines.append("However, continuous monitoring of fairness metrics is recommended as the model is deployed and updated.")
    
    # Join all lines into a single report text
    report_text = "\n".join(report_lines)
    
    # Save to file if path provided
    if output_path:
        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write report
            with open(output_path, 'w') as f:
                f.write(report_text)
                
            logger.info(f"Fairness report saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving fairness report: {str(e)}")
    
    # Return report text if not saved to file
    return report_text