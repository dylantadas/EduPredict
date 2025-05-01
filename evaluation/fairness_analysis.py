import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix
import json
from config import FAIRNESS, PROTECTED_ATTRIBUTES, DIRS

logger = logging.getLogger('edupredict')

def calculate_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_features: pd.DataFrame,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Calculates comprehensive fairness metrics across protected groups.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        protected_features: DataFrame with protected attributes
        threshold: Classification threshold
        
    Returns:
        Dictionary of fairness metrics by protected attribute
    """
    try:
        metrics = {}
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        for attr in FAIRNESS['protected_attributes']:
            if attr not in protected_features.columns:
                continue
                
            groups = protected_features[attr].unique()
            group_metrics = {}
            
            # Calculate metrics for each group
            for group in groups:
                group_mask = protected_features[attr] == group
                if group_mask.sum() < FAIRNESS['min_group_size']:
                    logger.warning(
                        f"Group {group} in {attr} has fewer than "
                        f"{FAIRNESS['min_group_size']} samples"
                    )
                    continue
                
                tn, fp, fn, tp = confusion_matrix(
                    y_true[group_mask],
                    y_pred_binary[group_mask]
                ).ravel()
                
                # True positive rate (recall)
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                # False positive rate
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                # Positive predictive value (precision)
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                # Selection rate
                selection_rate = (tp + fp) / (tp + fp + tn + fn)
                
                group_metrics[group] = {
                    'true_positive_rate': tpr,
                    'false_positive_rate': fpr,
                    'positive_predictive_value': ppv,
                    'selection_rate': selection_rate,
                    'sample_size': int(group_mask.sum())
                }
            
            # Calculate disparity metrics
            rates = pd.DataFrame(group_metrics).T
            if not rates.empty:
                max_selection_rate = rates['selection_rate'].max()
                min_selection_rate = rates['selection_rate'].min()
                
                disparities = {
                    'demographic_parity_difference': max_selection_rate - min_selection_rate,
                    'disparate_impact_ratio': min_selection_rate / max_selection_rate if max_selection_rate > 0 else 1,
                    'equal_opportunity_difference': rates['true_positive_rate'].max() - rates['true_positive_rate'].min(),
                    'average_odds_difference': (
                        (rates['true_positive_rate'].max() - rates['true_positive_rate'].min()) +
                        (rates['false_positive_rate'].max() - rates['false_positive_rate'].min())
                    ) / 2
                }
                
                # Check against thresholds
                threshold_violations = []
                for metric, value in disparities.items():
                    if metric in FAIRNESS['thresholds']:
                        threshold = FAIRNESS['thresholds'][metric]
                        if (metric.endswith('_ratio') and value < threshold) or \
                           (metric.endswith('_difference') and value > threshold):
                            threshold_violations.append(
                                f"{metric}: {value:.3f} (threshold: {threshold})"
                            )
                
                if threshold_violations:
                    logger.warning(
                        f"Fairness violations detected for {attr}:\n" +
                        "\n".join(threshold_violations)
                    )
                
                metrics[attr] = {
                    'group_metrics': group_metrics,
                    'disparities': disparities,
                    'threshold_violations': threshold_violations
                }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating fairness metrics: {str(e)}")
        raise

def analyze_bias_patterns(
    features: pd.DataFrame,
    predictions: np.ndarray,
    labels: np.ndarray,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Analyzes bias patterns in model predictions across demographic groups.
    
    Args:
        features: Input features including protected attributes
        predictions: Model predictions
        labels: True labels
        metadata: Optional metadata about the analysis
        
    Returns:
        Dictionary containing bias analysis results
    """
    try:
        analysis = {
            'overall_metrics': {},
            'group_metrics': {},
            'bias_patterns': [],
            'recommendations': []
        }
        
        # Analyze each protected attribute
        for attr in FAIRNESS['protected_attributes']:
            if attr not in features.columns:
                continue
                
            groups = features[attr].unique()
            group_performance = {}
            
            # Calculate performance metrics for each group
            for group in groups:
                group_mask = features[attr] == group
                if group_mask.sum() < FAIRNESS['min_group_size']:
                    continue
                
                group_preds = predictions[group_mask]
                group_labels = labels[group_mask]
                
                # Basic metrics
                accuracy = (group_preds == group_labels).mean()
                error_rate = 1 - accuracy
                
                # Error analysis
                fp_mask = (group_preds == 1) & (group_labels == 0)
                fn_mask = (group_preds == 0) & (group_labels == 1)
                
                group_performance[group] = {
                    'size': int(group_mask.sum()),
                    'accuracy': float(accuracy),
                    'error_rate': float(error_rate),
                    'false_positive_rate': float(fp_mask.sum() / (group_labels == 0).sum()) if (group_labels == 0).sum() > 0 else 0,
                    'false_negative_rate': float(fn_mask.sum() / (group_labels == 1).sum()) if (group_labels == 1).sum() > 0 else 0
                }
            
            # Analyze performance disparities
            if group_performance:
                error_rates = [g['error_rate'] for g in group_performance.values()]
                max_disparity = max(error_rates) - min(error_rates)
                
                # Record significant disparities
                if max_disparity > FAIRNESS['threshold']:
                    worst_group = max(group_performance.items(), key=lambda x: x[1]['error_rate'])
                    best_group = min(group_performance.items(), key=lambda x: x[1]['error_rate'])
                    
                    analysis['bias_patterns'].append({
                        'attribute': attr,
                        'disparity': max_disparity,
                        'worst_performing': {
                            'group': worst_group[0],
                            'error_rate': worst_group[1]['error_rate']
                        },
                        'best_performing': {
                            'group': best_group[0],
                            'error_rate': best_group[1]['error_rate']
                        }
                    })
                    
                    # Add recommendations based on disparity type
                    if attr in PROTECTED_ATTRIBUTES:
                        attribute_info = PROTECTED_ATTRIBUTES[attr]
                        if attribute_info.get('sensitive', False):
                            analysis['recommendations'].append(
                                f"High disparity detected in sensitive attribute {attr}. "
                                f"Consider applying bias mitigation techniques."
                            )
            
            analysis['group_metrics'][attr] = group_performance
        
        # Export analysis results
        if metadata:
            analysis['metadata'] = metadata
        
        output_path = DIRS['reports_fairness'] / 'bias_analysis.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing bias patterns: {str(e)}")
        raise

def generate_fairness_report(
    fairness_metrics: Dict[str, Dict[str, Any]],
    bias_analysis: Dict[str, Any],
    output_path: Optional[Path] = None
) -> str:
    """
    Generates a comprehensive fairness analysis report.
    
    Args:
        fairness_metrics: Dictionary of fairness metrics
        bias_analysis: Dictionary of bias analysis results
        output_path: Optional path to save the report
        
    Returns:
        Path to the generated report
    """
    try:
        report = []
        report.append("# Fairness Analysis Report\n")
        
        # Overall summary
        report.append("## Summary")
        violations = []
        for attr, metrics in fairness_metrics.items():
            if metrics['threshold_violations']:
                violations.extend([
                    f"- {attr}: {violation}"
                    for violation in metrics['threshold_violations']
                ])
        
        if violations:
            report.append("\n### Fairness Violations")
            report.extend(violations)
        else:
            report.append("\nNo fairness violations detected.")
        
        # Detailed metrics by protected attribute
        report.append("\n## Detailed Metrics by Protected Attribute")
        for attr, metrics in fairness_metrics.items():
            report.append(f"\n### {attr}")
            
            # Group metrics table
            report.append("\n#### Group Performance")
            report.append("\n| Group | Sample Size | Selection Rate | TPR | FPR | PPV |")
            report.append("|-------|--------------|----------------|-----|-----|-----|")
            
            for group, group_metrics in metrics['group_metrics'].items():
                report.append(
                    f"| {group} | {group_metrics['sample_size']} | "
                    f"{group_metrics['selection_rate']:.3f} | "
                    f"{group_metrics['true_positive_rate']:.3f} | "
                    f"{group_metrics['false_positive_rate']:.3f} | "
                    f"{group_metrics['positive_predictive_value']:.3f} |"
                )
            
            # Disparity metrics
            report.append("\n#### Disparity Metrics")
            for metric, value in metrics['disparities'].items():
                report.append(f"- {metric}: {value:.3f}")
        
        # Bias patterns
        if bias_analysis['bias_patterns']:
            report.append("\n## Identified Bias Patterns")
            for pattern in bias_analysis['bias_patterns']:
                report.append(f"\n### {pattern['attribute']}")
                report.append(f"- Disparity: {pattern['disparity']:.3f}")
                report.append(f"- Worst performing group: {pattern['worst_performing']['group']} "
                            f"(error rate: {pattern['worst_performing']['error_rate']:.3f})")
                report.append(f"- Best performing group: {pattern['best_performing']['group']} "
                            f"(error rate: {pattern['best_performing']['error_rate']:.3f})")
        
        # Recommendations
        if bias_analysis['recommendations']:
            report.append("\n## Recommendations")
            for rec in bias_analysis['recommendations']:
                report.append(f"- {rec}")
        
        # Save report
        report_text = "\n".join(report)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        else:
            output_path = DIRS['reports_fairness'] / 'fairness_report.md'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error generating fairness report: {str(e)}")
        raise