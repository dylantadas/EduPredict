import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import argparse
from json import JSONEncoder

from config import EVALUATION, FAIRNESS, DIRS, PROTECTED_ATTRIBUTES

# Set up the logger
logger = logging.getLogger('edupredict2')

class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def generate_performance_report(
    metrics: Dict[str, Any], 
    output_path: str
) -> str:
    """
    Generates report on model performance.
    
    Args:
        metrics: Dictionary of performance metrics
        output_path: Path to save report
        
    Returns:
        Path to saved report
    """
    logger.info("Generating performance report")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize report content
    report_lines = []
    report_lines.append("# Model Performance Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add model summary section
    report_lines.append("\n## Model Performance Summary")
    
    # Add key metrics if available
    key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    available_metrics = [m for m in key_metrics if m in metrics]
    
    if available_metrics:
        report_lines.append("\n### Key Performance Metrics")
        report_lines.append("| Metric | Value |")
        report_lines.append("|--------|-------|")
        
        for metric in available_metrics:
            value = metrics.get(metric)
            if isinstance(value, (int, float)):
                report_lines.append(f"| {metric} | {value:.4f} |")
    
    # Add confusion matrix if available
    if 'confusion_matrix' in metrics:
        report_lines.append("\n### Confusion Matrix")
        cm = metrics['confusion_matrix']
        
        if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
            # Format the confusion matrix for display
            report_lines.append("```")
            report_lines.append("              | Predicted Negative | Predicted Positive |")
            report_lines.append("--------------|-------------------|-------------------|")
            report_lines.append(f"Actual Negative |       {cm[0][0]}       |       {cm[0][1]}       |")
            report_lines.append(f"Actual Positive |       {cm[1][0]}       |       {cm[1][1]}       |")
            report_lines.append("```")
            
            # Extract and display confusion matrix components
            if all(key in metrics for key in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']):
                tp = metrics['true_positives']
                fp = metrics['false_positives']
                tn = metrics['true_negatives']
                fn = metrics['false_negatives']
                
                report_lines.append("\n**Confusion Matrix Components:**")
                report_lines.append(f"- True Positives (TP): {tp}")
                report_lines.append(f"- False Positives (FP): {fp}")
                report_lines.append(f"- True Negatives (TN): {tn}")
                report_lines.append(f"- False Negatives (FN): {fn}")
    
    # Add class distribution if available
    if 'class_distribution' in metrics:
        report_lines.append("\n### Class Distribution")
        
        class_dist = metrics['class_distribution']
        if 'true' in class_dist and 'pred' in class_dist:
            true_dist = class_dist['true']
            pred_dist = class_dist['pred']
            
            report_lines.append("| Class | Actual Count | Predicted Count |")
            report_lines.append("|-------|-------------|----------------|")
            
            for i, (true_count, pred_count) in enumerate(zip(true_dist, pred_dist)):
                report_lines.append(f"| {i} | {true_count} | {pred_count} |")
            
            # Add class percentages
            true_total = sum(true_dist)
            pred_total = sum(pred_dist)
            
            report_lines.append("\n**Class Percentages:**")
            for i, (true_count, pred_count) in enumerate(zip(true_dist, pred_dist)):
                true_pct = true_count / true_total * 100 if true_total > 0 else 0
                pred_pct = pred_count / pred_total * 100 if pred_total > 0 else 0
                report_lines.append(f"- Class {i}: Actual {true_pct:.1f}%, Predicted {pred_pct:.1f}%")
    
    # Add detailed performance metrics if available
    if 'classification_report' in metrics:
        report_lines.append("\n### Detailed Classification Report")
        
        # Format classification report
        report_dict = metrics['classification_report']
        if isinstance(report_dict, dict):
            # Header
            report_lines.append("| Class | Precision | Recall | F1-Score | Support |")
            report_lines.append("|-------|-----------|--------|----------|---------|")
            
            # Per-class metrics
            for class_label, class_metrics in report_dict.items():
                if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                    precision = class_metrics.get('precision', 'N/A')
                    recall = class_metrics.get('recall', 'N/A')
                    f1 = class_metrics.get('f1-score', 'N/A')
                    support = class_metrics.get('support', 'N/A')
                    
                    if isinstance(precision, float):
                        precision = f"{precision:.4f}"
                    if isinstance(recall, float):
                        recall = f"{recall:.4f}"
                    if isinstance(f1, float):
                        f1 = f"{f1:.4f}"
                    
                    report_lines.append(f"| {class_label} | {precision} | {recall} | {f1} | {support} |")
            
            # Add averages
            for avg_type in ['accuracy', 'macro avg', 'weighted avg']:
                if avg_type in report_dict:
                    avg_metrics = report_dict[avg_type]
                    
                    if avg_type == 'accuracy':
                        accuracy = avg_metrics
                        if isinstance(accuracy, float):
                            accuracy = f"{accuracy:.4f}"
                        report_lines.append(f"| **Accuracy** | | | {accuracy} | |")
                    else:
                        precision = avg_metrics.get('precision', 'N/A')
                        recall = avg_metrics.get('recall', 'N/A')
                        f1 = avg_metrics.get('f1-score', 'N/A')
                        support = avg_metrics.get('support', 'N/A')
                        
                        if isinstance(precision, float):
                            precision = f"{precision:.4f}"
                        if isinstance(recall, float):
                            recall = f"{recall:.4f}"
                        if isinstance(f1, float):
                            f1 = f"{f1:.4f}"
                        
                        report_lines.append(f"| **{avg_type}** | {precision} | {recall} | {f1} | {support} |")
    
    # Add confidence intervals if available
    if 'confidence_intervals' in metrics:
        report_lines.append("\n### Confidence Intervals (95%)")
        
        ci_dict = metrics['confidence_intervals']
        if isinstance(ci_dict, dict):
            # Header
            report_lines.append("| Metric | Mean | Lower Bound | Upper Bound | Std. Dev. |")
            report_lines.append("|--------|------|-------------|-------------|----------|")
            
            # Add each metric's CI
            for metric, ci_values in ci_dict.items():
                mean_val = ci_values.get('mean', 'N/A')
                lower = ci_values.get('lower', 'N/A')
                upper = ci_values.get('upper', 'N/A')
                std = ci_values.get('std', 'N/A')
                
                if isinstance(mean_val, float):
                    mean_val = f"{mean_val:.4f}"
                if isinstance(lower, float):
                    lower = f"{lower:.4f}"
                if isinstance(upper, float):
                    upper = f"{upper:.4f}"
                if isinstance(std, float):
                    std = f"{std:.4f}"
                
                report_lines.append(f"| {metric} | {mean_val} | {lower} | {upper} | {std} |")
    
    # Add interpretation and recommendations
    report_lines.append("\n## Interpretation and Recommendations")
    
    # Interpret accuracy
    if 'accuracy' in metrics:
        accuracy = metrics['accuracy']
        
        if accuracy >= 0.9:
            report_lines.append("\n- **Accuracy**: The model has excellent overall accuracy.")
        elif accuracy >= 0.8:
            report_lines.append("\n- **Accuracy**: The model has good overall accuracy.")
        elif accuracy >= 0.7:
            report_lines.append("\n- **Accuracy**: The model has acceptable overall accuracy.")
        else:
            report_lines.append("\n- **Accuracy**: The model's overall accuracy is below acceptable thresholds. Further model improvements are recommended.")
    
    # Interpret precision and recall
    if all(m in metrics for m in ['precision', 'recall']):
        precision = metrics['precision']
        recall = metrics['recall']
        
        # Check for precision-recall imbalance
        if precision - recall > 0.2:
            report_lines.append("\n- **Precision vs. Recall**: The model has significantly higher precision than recall, indicating conservative predictions. " +
                               "The model may be missing positive cases. Consider adjusting the classification threshold to improve recall.")
        elif recall - precision > 0.2:
            report_lines.append("\n- **Precision vs. Recall**: The model has significantly higher recall than precision, indicating liberal predictions. " +
                               "The model may be generating too many false positives. Consider adjusting the classification threshold to improve precision.")
        else:
            report_lines.append("\n- **Precision vs. Recall**: The model has a good balance between precision and recall.")
    
    # Add general recommendations
    report_lines.append("\n### Recommendations")
    
    if 'accuracy' in metrics and metrics['accuracy'] < 0.7:
        report_lines.append("\n1. **Model Improvement**: Consider the following approaches to improve model performance:")
        report_lines.append("   - Feature engineering to create more informative predictors")
        report_lines.append("   - Trying different algorithms or ensemble methods")
        report_lines.append("   - Hyperparameter tuning to optimize model configuration")
        report_lines.append("   - Collecting more training data if possible")
    
    if 'class_distribution' in metrics:
        class_dist = metrics['class_distribution']
        if 'true' in class_dist and len(class_dist['true']) >= 2:
            # Check for class imbalance
            if class_dist['true'][0] / sum(class_dist['true']) > 0.7 or class_dist['true'][1] / sum(class_dist['true']) > 0.7:
                report_lines.append("\n2. **Class Imbalance**: The dataset shows significant class imbalance. Consider:")
                report_lines.append("   - Resampling techniques (oversampling or undersampling)")
                report_lines.append("   - Using class weights to penalize misclassification of the minority class")
                report_lines.append("   - Trying algorithms that are robust to class imbalance")
    
    # Include ROC and PR curve information if available
    if 'auc_roc' in metrics:
        report_lines.append("\n3. **Threshold Tuning**: The model's threshold can be adjusted based on the business requirements:")
        report_lines.append("   - Increase threshold to improve precision (fewer false positives)")
        report_lines.append("   - Decrease threshold to improve recall (fewer false negatives)")
        report_lines.append("   - Review the ROC curve to find the optimal threshold")
    
    # Join lines to create the report text
    report_text = "\n".join(report_lines)
    
    # Save the report
    try:
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Performance report saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving performance report: {str(e)}")
    
    return output_path


def generate_fairness_report(
    fairness_results: Dict[str, Dict], 
    thresholds: Optional[Dict[str, float]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generates report on fairness evaluation.
    
    Args:
        fairness_results: Dictionary of fairness results
        thresholds: Dictionary of fairness thresholds
        output_path: Path to save report
        
    Returns:
        Report text or path to saved report
    """
    logger.info("Generating fairness report")
    
    # Use default thresholds from config if none specified
    if thresholds is None:
        thresholds = FAIRNESS.get('thresholds', {
            'demographic_parity_difference': 0.1,
            'disparate_impact_ratio': 0.8,
            'equal_opportunity_difference': 0.1,
            'average_odds_difference': 0.1
        })
    
    # Initialize report content
    report_lines = []
    report_lines.append("# Fairness Evaluation Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
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
    
    # Add information about protected attributes
    report_lines.append("\n## Protected Attributes Information")
    
    for attr_name, attr_info in PROTECTED_ATTRIBUTES.items():
        report_lines.append(f"\n### {attr_name}")
        report_lines.append(f"- Values: {', '.join(str(v) for v in attr_info.get('values', []))}")
        report_lines.append(f"- Sensitive: {attr_info.get('sensitive', True)}")
        report_lines.append(f"- Balanced threshold: {attr_info.get('balanced_threshold', 'N/A')}")
    
    # Join lines to create the report text
    report_text = "\n".join(report_lines)
    
    # Save the report if path is provided
    if output_path:
        try:
            # Ensure directory exists
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Write report
            with open(output_path, 'w') as f:
                f.write(report_text)
                
            logger.info(f"Fairness report saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving fairness report: {str(e)}")
    
    return report_text


def compare_model_versions(
    metrics_list: List[Dict[str, Any]], 
    model_names: List[str]
) -> pd.DataFrame:
    """
    Compares performance of different model versions.
    
    Args:
        metrics_list: List of metric dictionaries
        model_names: List of model names
        
    Returns:
        DataFrame comparing models
    """
    logger.info(f"Comparing {len(model_names)} model versions")
    
    # Check if lists have same length
    if len(metrics_list) != len(model_names):
        logger.error("metrics_list and model_names must have same length")
        return pd.DataFrame()
    
    # Define metrics to compare
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    
    # Create list to hold comparison rows
    comparison_rows = []
    
    # Create a row for each model
    for i, (metrics, name) in enumerate(zip(metrics_list, model_names)):
        row = {'model': name}
        
        # Add each metric if available
        for metric in metrics_to_compare:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[metric] = value
        
        # Add any available confidence intervals
        if 'confidence_intervals' in metrics:
            ci_dict = metrics['confidence_intervals']
            for metric, ci in ci_dict.items():
                if isinstance(ci, dict):
                    # Add mean and confidence interval bounds
                    if 'mean' in ci:
                        row[f"{metric}_mean"] = ci['mean']
                    if 'lower' in ci and 'upper' in ci:
                        row[f"{metric}_ci"] = f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"
        
        comparison_rows.append(row)
    
    # Convert to DataFrame
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        
        # Sort by F1 score if available, otherwise by accuracy
        if 'f1' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1', ascending=False)
        elif 'accuracy' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        
        # Calculate improvements relative to first model
        if len(comparison_df) > 1:
            for metric in metrics_to_compare:
                if metric in comparison_df.columns:
                    baseline = comparison_df.iloc[0][metric]
                    comparison_df[f"{metric}_change"] = comparison_df[metric].apply(
                        lambda x: ((x - baseline) / baseline) * 100 if baseline != 0 else float('inf')
                    )
        
        logger.info(f"Model comparison completed for {len(comparison_df)} models")
        return comparison_df
    else:
        logger.warning("No model comparison data available")
        return pd.DataFrame(columns=['model'] + metrics_to_compare)


def export_evaluation_results(
    results: Dict[str, Any], 
    filepath: str
) -> None:
    """
    Exports evaluation results to file.
    
    Args:
        results: Dictionary of evaluation results
        filepath: Path to save results
        
    Returns:
        None
    """
    logger.info(f"Exporting evaluation results to {filepath}")
    
    # Ensure directory exists
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy arrays and other non-serializable objects to lists or strings
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    # Process dictionary recursively
    def process_dict(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = process_dict(v)
            elif isinstance(v, list):
                result[k] = [make_serializable(item) if not isinstance(item, dict) else process_dict(item) for item in v]
            else:
                result[k] = make_serializable(v)
        return result
    
    # Create serializable version of results
    serializable_results = process_dict(results)
    
    # Add metadata
    serializable_results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'format_version': '1.0'
    }
    
    # Save to file
    try:
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.json':
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        elif ext == '.csv' and 'metrics' in serializable_results:
            # If it's a CSV, try to convert main metrics to a dataframe
            metrics_df = pd.DataFrame([serializable_results['metrics']])
            metrics_df.to_csv(filepath, index=False)
        else:
            # Default to JSON with custom extension
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results exported to {filepath}")
    except Exception as e:
        logger.error(f"Error exporting evaluation results: {str(e)}")


def run_reporting_pipeline(
    data_results: Dict[str, Any],
    feature_results: Dict[str, Any],
    model_results: Dict[str, Any],
    fairness_results: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, str]:
    """
    Execute comprehensive reporting and documentation pipeline.
    """
    try:
        logger.info("Starting reporting pipeline...")
        reports_dir = dirs['reports']
        reports_dir.mkdir(exist_ok=True)

        # Generate initial data quality report
        logger.info("Generating data quality report...")
        data_report = {
            'data_quality': data_results.get('quality_report', {}),
            'data_splits': {
                'train_size': len(data_results.get('splits', {}).get('train', [])),
                'val_size': len(data_results.get('splits', {}).get('validation', [])),
                'test_size': len(data_results.get('splits', {}).get('test', [])),
            },
            'demographics': {},
            'recommendations': []
        }

        # Add demographic distributions if available
        if 'clean_data' in data_results and 'demographics' in data_results['clean_data']:
            for attr in ['gender', 'age_band', 'imd_band', 'region', 'disability']:
                if attr in data_results['clean_data']['demographics']:
                    dist = data_results['clean_data']['demographics'][attr].value_counts(normalize=True).to_dict()
                    data_report['demographics'][attr] = dist

        # Add recommendations based on data quality
        if 'quality_report' in data_results and 'recommendations' in data_results['quality_report']:
            data_report['recommendations'].extend(data_results['quality_report']['recommendations'])

        # Save data quality report
        report_path = reports_dir / 'data_quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(data_report, f, indent=2, cls=NumpyJSONEncoder)
        logger.info(f"Saved data quality report to {report_path}")

        # Add feature engineering results if available
        if feature_results:
            feature_report_path = reports_dir / 'feature_report.json'
            with open(feature_report_path, 'w') as f:
                json.dump(feature_results, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Saved feature engineering report to {feature_report_path}")

        # Add model results if available
        if model_results:
            model_report_path = reports_dir / 'model_report.json'
            with open(model_report_path, 'w') as f:
                json.dump(model_results, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Saved model performance report to {model_report_path}")

        # Add fairness results if available
        if fairness_results:
            fairness_report_path = reports_dir / 'fairness_report.json'
            with open(fairness_report_path, 'w') as f:
                json.dump(fairness_results, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Saved fairness analysis report to {fairness_report_path}")

        return {
            'data_quality_report': str(report_path),
            'feature_report': str(feature_report_path) if feature_results else None,
            'model_report': str(model_report_path) if model_results else None,
            'fairness_report': str(fairness_report_path) if fairness_results else None
        }

    except Exception as e:
        logger.error(f"Error in reporting pipeline: {str(e)}")
        raise