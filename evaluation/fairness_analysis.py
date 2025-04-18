import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from ..config import FAIRNESS_THRESHOLDS, BIAS_MITIGATION


def calculate_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group_values: np.ndarray
) -> pd.DataFrame:
    """Calculates performance metrics for each demographic group."""
    
    unique_groups = np.unique(group_values)
    metrics = []
    
    for group in unique_groups:
        # get indices for this group
        group_mask = (group_values == group)
        
        # only proceed if enough samples
        if np.sum(group_mask) < 10:
            continue
            
        # get group data
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        group_y_prob = y_prob[group_mask]
        
        # calculate basic metrics
        group_metrics = {
            'group': group,
            'count': int(np.sum(group_mask)),
            'positive_rate_true': group_y_true.mean(),
            'positive_rate_pred': group_y_pred.mean(),
            'accuracy': accuracy_score(group_y_true, group_y_pred),
            'precision': precision_score(group_y_true, group_y_pred, zero_division=0),
            'recall': recall_score(group_y_true, group_y_pred, zero_division=0),
            'f1': f1_score(group_y_true, group_y_pred, zero_division=0),
        }
        
        # calculate auc if both classes present
        if len(np.unique(group_y_true)) > 1:
            group_metrics['auc'] = roc_auc_score(group_y_true, group_y_prob)
        else:
            group_metrics['auc'] = np.nan
            
        # calculate tpr and fpr
        group_metrics['true_positive_rate'] = recall_score(
            group_y_true, group_y_pred, zero_division=0
        )
        
        if np.sum(group_y_true == 0) > 0:
            group_metrics['false_positive_rate'] = np.sum(
                (group_y_true == 0) & (group_y_pred == 1)
            ) / np.sum(group_y_true == 0)
        else:
            group_metrics['false_positive_rate'] = np.nan
            
        metrics.append(group_metrics)
    
    return pd.DataFrame(metrics)


def calculate_fairness_metrics(group_metrics: pd.DataFrame) -> Dict:
    """Calculates fairness metrics from group-level performance metrics."""
    
    # demographic parity metrics (prediction rates)
    pred_rates = group_metrics['positive_rate_pred']
    demo_parity_diff = pred_rates.max() - pred_rates.min()
    
    if pred_rates.max() > 0:
        disparate_impact = pred_rates.min() / pred_rates.max()
    else:
        disparate_impact = 1.0
        
    # equal opportunity metrics (true positive rates)
    tprs = group_metrics['true_positive_rate']
    equal_opp_diff = tprs.max() - tprs.min()
    
    # equalized odds metrics (both TPR and FPR)
    fprs = group_metrics['false_positive_rate']
    eq_odds_diff_fpr = fprs.max() - fprs.min() if not fprs.isna().any() else np.nan
    eq_odds_diff = np.sqrt(equal_opp_diff**2 + eq_odds_diff_fpr**2) if not np.isnan(eq_odds_diff_fpr) else equal_opp_diff
    
    # performance metrics
    performance_diffs = {
        'accuracy_diff': group_metrics['accuracy'].max() - group_metrics['accuracy'].min(),
        'precision_diff': group_metrics['precision'].max() - group_metrics['precision'].min(),
        'recall_diff': group_metrics['recall'].max() - group_metrics['recall'].min(),
        'f1_diff': group_metrics['f1'].max() - group_metrics['f1'].min(),
        'auc_diff': group_metrics['auc'].max() - group_metrics['auc'].min() if not group_metrics['auc'].isna().any() else np.nan
    }
    
    # combine all metrics
    fairness_metrics = {
        'demographic_parity_difference': demo_parity_diff,
        'disparate_impact_ratio': disparate_impact,
        'equal_opportunity_difference': equal_opp_diff,
        'equalized_odds_difference_fpr': eq_odds_diff_fpr,
        'equalized_odds_difference': eq_odds_diff,
        **performance_diffs,
        'min_group_size': group_metrics['count'].min(),
        'max_group_size': group_metrics['count'].max(),
        'n_groups': len(group_metrics)
    }
    
    return fairness_metrics


def calculate_composite_fairness_score(
    metrics: List[Dict[str, float]],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate weighted composite fairness score."""
    if weights is None:
        weights = {
            'demographic_parity_difference': 1/3,
            'equalized_odds': 1/3,
            'disparate_impact_ratio': 1/3
        }
    
    scores = []
    
    for metric in metrics:
        if metric['metric'] == 'demographic_parity_difference':
            score = 1.0 - min(1.0, metric['value'] / 0.1)
            scores.append(weights['demographic_parity_difference'] * score)
            
        elif metric['metric'] == 'equalized_odds':
            avg_diff = (metric['tpr_difference'] + metric['fpr_difference']) / 2
            score = 1.0 - min(1.0, avg_diff / 0.1)
            scores.append(weights['equalized_odds'] * score)
            
        elif metric['metric'] == 'disparate_impact_ratio':
            score = 1.0 - min(1.0, abs(1.0 - metric['value']))
            scores.append(weights['disparate_impact_ratio'] * score)
    
    return sum(scores)


def evaluate_model_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    protected_attributes: Dict[str, np.ndarray],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict:
    """Evaluates model fairness across multiple protected attributes."""
    
    if thresholds is None:
        thresholds = FAIRNESS_THRESHOLDS
    
    fairness_results = {}
    
    for attr_name, attr_values in protected_attributes.items():
        # calculate group metrics
        group_metrics = calculate_group_metrics(
            y_true, y_pred, y_prob, attr_values
        )
        
        # calculate fairness metrics
        fairness_metrics = calculate_fairness_metrics(group_metrics)
        
        # evaluate against thresholds
        threshold_results = {}
        for metric_name, threshold in thresholds.items():
            if metric_name in fairness_metrics:
                if metric_name == 'disparate_impact_ratio':
                    # for disparate impact, higher is better
                    threshold_results[metric_name] = fairness_metrics[metric_name] >= threshold
                else:
                    # for differences, lower is better
                    threshold_results[metric_name] = fairness_metrics[metric_name] <= threshold
        
        # combine results
        fairness_results[attr_name] = {
            'group_metrics': group_metrics,
            'fairness_metrics': fairness_metrics,
            'threshold_results': threshold_results,
            'passes_all_thresholds': all(threshold_results.values())
        }
    
    return fairness_results


def detect_demographic_imbalance(
    data: pd.DataFrame,
    protected_attributes: Dict[str, str],
    intersectional: bool = True
) -> Dict[str, float]:
    """Detects imbalances in demographic representation."""
    
    if intersectional:
        # Create intersectional groups
        data['group'] = data[protected_attributes.keys()].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        group_counts = data['group'].value_counts()
    else:
        # Check each attribute separately
        group_counts = {
            attr: data[attr].value_counts()
            for attr in protected_attributes
        }
    
    # Calculate imbalance metrics
    metrics = {
        'max_ratio': group_counts.max() / group_counts.min(),
        'std_dev': group_counts.std() / group_counts.mean(),
        'smallest_group': group_counts.min(),
        'largest_group': group_counts.max()
    }
    
    return metrics


def create_sample_weights(
    data: pd.DataFrame,
    protected_attributes: Dict[str, str],
    strategy: str = 'group_balanced'
) -> np.ndarray:
    """Creates sample weights to balance demographic representation."""
    
    if strategy == 'group_balanced':
        # Calculate intersectional group frequencies
        data['group'] = data[protected_attributes.keys()].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        group_counts = data['group'].value_counts()
        
        # Create weights inversely proportional to frequency
        weights = 1 / group_counts[data['group']].values
        weights = weights * (len(data) / weights.sum())  # Normalize
        
    else:
        raise ValueError(f"Unsupported weighting strategy: {strategy}")
    
    return weights


def resample_training_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    protected_attributes: Dict[str, str],
    method: str = 'reweight',
    random_state: int = 42
) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Resamples or reweights training data to mitigate bias."""
    
    if method == 'none':
        return X, y, None
    
    # Create intersectional groups for stratification
    if isinstance(X, pd.DataFrame):
        groups = X[protected_attributes.keys()].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
    else:
        # Assume protected attributes are provided in X
        groups = np.array(['_'.join(map(str, row)) 
                         for row in X[:, :len(protected_attributes)]])
    
    if method == 'reweight':
        # Calculate sample weights
        weights = create_sample_weights(
            pd.DataFrame({'group': groups}),
            {'group': 'group'},
            strategy='group_balanced'
        )
        return X, y, weights
    
    elif method == 'oversample':
        # Use SMOTE with group-based stratification
        sampler = SMOTE(
            sampling_strategy='not majority',
            random_state=random_state,
            k_neighbors=min(5, min(np.bincount(groups.astype('category').codes)) - 1)
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None
    
    elif method == 'undersample':
        # Use random undersampling
        sampler = RandomUnderSampler(
            sampling_strategy='majority',
            random_state=random_state
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled, None
    
    else:
        raise ValueError(f"Unsupported resampling method: {method}")


def evaluate_bias_mitigation(
    original_results: Dict,
    mitigated_results: Dict,
    protected_attributes: Dict[str, str]
) -> Dict:
    """Compares fairness metrics before and after bias mitigation."""
    
    comparison = {}
    
    # Compare key metrics
    metrics_to_compare = [
        'demographic_parity_difference',
        'disparate_impact_ratio',
        'equal_opportunity_difference'
    ]
    
    for attr in protected_attributes:
        comparison[attr] = {}
        for metric in metrics_to_compare:
            original = original_results[attr]['fairness_metrics'].get(metric, 0)
            mitigated = mitigated_results[attr]['fairness_metrics'].get(metric, 0)
            
            comparison[attr][metric] = {
                'before': original,
                'after': mitigated,
                'improvement': mitigated - original if metric != 'disparate_impact_ratio'
                              else mitigated - original
            }
    
    # Calculate overall improvement score
    total_improvement = 0
    metric_count = 0
    
    for attr_results in comparison.values():
        for metric, values in attr_results.items():
            if metric != 'disparate_impact_ratio':
                # For difference metrics, reduction is improvement
                total_improvement += -values['improvement']
            else:
                # For ratio metrics, closer to 1 is better
                total_improvement += -abs(1 - values['after']) + abs(1 - values['before'])
            metric_count += 1
    
    comparison['overall_improvement'] = total_improvement / metric_count if metric_count > 0 else 0
    
    return comparison