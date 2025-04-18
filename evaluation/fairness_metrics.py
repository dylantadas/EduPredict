from typing import Dict, List, Optional
import numpy as np
import pandas as pd

def calculate_demographic_parity(
    y_pred: np.ndarray,
    protected_attribute: np.ndarray
) -> Dict[str, float]:
    """Calculate demographic parity difference."""
    unique_values = np.unique(protected_attribute)
    pred_rates = {}
    
    for value in unique_values:
        mask = (protected_attribute == value)
        pred_rates[value] = y_pred[mask].mean()
    
    max_diff = max(pred_rates.values()) - min(pred_rates.values())
    
    return {
        'metric': 'demographic_parity_difference',
        'value': max_diff,
        'group_rates': pred_rates
    }

def calculate_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: np.ndarray
) -> Dict[str, float]:
    """Calculate equalized odds differences."""
    unique_values = np.unique(protected_attribute)
    tpr_rates = {}
    fpr_rates = {}
    
    for value in unique_values:
        mask = (protected_attribute == value)
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        
        # True Positive Rate
        pos_mask = (group_true == 1)
        if pos_mask.any():
            tpr_rates[value] = (group_pred[pos_mask] == 1).mean()
        
        # False Positive Rate
        neg_mask = (group_true == 0)
        if neg_mask.any():
            fpr_rates[value] = (group_pred[neg_mask] == 1).mean()
    
    tpr_diff = max(tpr_rates.values()) - min(tpr_rates.values())
    fpr_diff = max(fpr_rates.values()) - min(fpr_rates.values())
    
    return {
        'metric': 'equalized_odds',
        'tpr_difference': tpr_diff,
        'fpr_difference': fpr_diff,
        'group_tpr': tpr_rates,
        'group_fpr': fpr_rates
    }

def calculate_disparate_impact(
    y_pred: np.ndarray,
    protected_attribute: np.ndarray
) -> Dict[str, float]:
    """Calculate disparate impact ratio."""
    unique_values = np.unique(protected_attribute)
    pred_rates = {}
    
    for value in unique_values:
        mask = (protected_attribute == value)
        pred_rates[value] = y_pred[mask].mean()
    
    max_rate = max(pred_rates.values())
    min_rate = min(pred_rates.values())
    
    ratio = min_rate / max_rate if max_rate > 0 else 1.0
    
    return {
        'metric': 'disparate_impact_ratio',
        'value': ratio,
        'group_rates': pred_rates
    }

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
            # Convert to score (0 = unfair, 1 = fair)
            score = 1.0 - min(1.0, metric['value'] / 0.1)  # 0.1 as max acceptable difference
            scores.append(weights['demographic_parity_difference'] * score)
            
        elif metric['metric'] == 'equalized_odds':
            # Average TPR and FPR differences
            avg_diff = (metric['tpr_difference'] + metric['fpr_difference']) / 2
            score = 1.0 - min(1.0, avg_diff / 0.1)
            scores.append(weights['equalized_odds'] * score)
            
        elif metric['metric'] == 'disparate_impact_ratio':
            # Score based on how close to 1.0 the ratio is
            score = 1.0 - min(1.0, abs(1.0 - metric['value']))
            scores.append(weights['disparate_impact_ratio'] * score)
    
    return sum(scores)
