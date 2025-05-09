import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from config import EVALUATION, RANDOM_SEED, FEATURE_ENGINEERING

# Set up the logger
logger = logging.getLogger('edupredict2')

def calculate_model_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray, 
    metrics: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Calculates comprehensive model performance metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted binary values
        y_prob: Predicted probabilities
        metrics: List of metrics to calculate (None = all)
        logger: Logger for tracking metric calculation
        
    Returns:
        Dictionary of performance metrics
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    logger.info("Calculating model performance metrics...")
    
    # Use default metrics from config if none specified
    if metrics is None:
        metrics = EVALUATION.get('metrics', ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    
    # Initialize results dictionary
    results = {}
    
    # Basic performance metrics
    if 'accuracy' in metrics:
        results['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    if 'precision' in metrics:
        results['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    
    if 'recall' in metrics:
        results['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    
    if 'f1' in metrics:
        results['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    
    if 'auc' in metrics:
        try:
            results['auc_roc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            results['auc_roc'] = float('nan')
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Extract components
        if len(cm) >= 2 and len(cm[0]) >= 2:
            tn, fp, fn, tp = cm.ravel()
            results['true_negatives'] = int(tn)
            results['false_positives'] = int(fp)
            results['false_negatives'] = int(fn)
            results['true_positives'] = int(tp)
    
    # Classification report
    if 'classification_report' in metrics:
        report = classification_report(y_true, y_pred, output_dict=True)
        results['classification_report'] = report
    
    # ROC curve points
    if 'roc_curve' in metrics:
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            results['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        except Exception as e:
            logger.warning(f"Could not calculate ROC curve points: {str(e)}")
    
    # Precision-recall curve points
    if 'pr_curve' in metrics:
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            results['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist() if len(thresholds) > 0 else []
            }
            results['average_precision'] = float(average_precision_score(y_true, y_prob))
        except Exception as e:
            logger.warning(f"Could not calculate PR curve points: {str(e)}")
    
    # Class distribution
    results['class_distribution'] = {
        'true': np.bincount(y_true.astype(int)).tolist(),
        'pred': np.bincount(y_pred.astype(int)).tolist()
    }
    
    logger.info(f"Model metrics calculated successfully: {', '.join(results.keys())}")
    return results


def analyze_feature_importance(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_estimators: int = 100, 
    random_state: int = 0, 
    importance_threshold: float = 0.01,
    plot: bool = True,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Analyzes feature importance using random forest classifier.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_estimators: Number of trees
        random_state: Random seed
        importance_threshold: Minimum importance threshold
        plot: Whether to plot importances
        output_path: Path to save importance visualization
        logger: Logger for tracking importance analysis
        
    Returns:
        DataFrame with feature importances
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    logger.info("Analyzing feature importance...")
    
    # Use default threshold from config if not specified
    if importance_threshold is None:
        importance_threshold = FEATURE_ENGINEERING.get('importance_threshold', 0.01)
    
    # Select only numeric columns for importance analysis
    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'bool']).columns
    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found for feature importance analysis")
        return pd.DataFrame(columns=['feature', 'importance'])
    
    X_numeric = X[numeric_cols]
    
    # Train random forest to get feature importance
    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state
        )
        model.fit(X_numeric, y)
        importances = model.feature_importances_
    except Exception as e:
        logger.error(f"Error training model for feature importance: {str(e)}")
        return pd.DataFrame(columns=['feature', 'importance'])
    
    # Create and sort feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': numeric_cols,
        'importance': importances
    })
    
    feature_importance = feature_importance.sort_values(
        'importance', ascending=False
    ).reset_index(drop=True)
    
    # Filter by threshold
    significant_features = feature_importance[feature_importance['importance'] > importance_threshold]
    
    if len(significant_features) == 0:
        logger.warning(f"No features exceed importance threshold of {importance_threshold}")
        # Return all features anyway, but log the warning
        significant_features = feature_importance
    
    # Plot feature importances
    if plot:
        try:
            plt.figure(figsize=(12, 8))
            
            # Limit to top 30 features for readability
            plot_data = significant_features.head(30)
            
            # Create the barplot
            ax = sns.barplot(
                x='importance', 
                y='feature', 
                data=plot_data
            )
            plt.title('Feature Importance Analysis (Top 30 Features)')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Save if output path provided
            if output_path:
                # Ensure directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {output_path}")
            
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error plotting feature importance: {str(e)}")
    
    logger.info(f"Feature importance analysis completed with {len(significant_features)} significant features")
    return significant_features


def analyze_feature_correlations(
    X: pd.DataFrame, 
    threshold: float = 0.85, 
    plot: bool = True,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Analyzes correlation between numeric features.
    
    Args:
        X: Feature DataFrame
        threshold: Correlation threshold
        plot: Whether to plot correlation matrix
        output_path: Path to save correlation visualization
        logger: Logger for tracking correlation analysis
        
    Returns:
        DataFrame with correlated feature pairs
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    logger.info("Analyzing feature correlations...")
    
    # Use default threshold from config if not specified
    if threshold is None:
        threshold = FEATURE_ENGINEERING.get('correlation_threshold', 0.85)
    
    # Select only numeric columns for correlation analysis
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found for correlation analysis")
        return pd.DataFrame(columns=['feature1', 'feature2', 'correlation'])
    
    X_numeric = X[numeric_cols]
    
    # Calculate correlation matrix
    try:
        corr_matrix = X_numeric.corr()
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {str(e)}")
        return pd.DataFrame(columns=['feature1', 'feature2', 'correlation'])
    
    # Plot correlation heatmap
    if plot:
        try:
            plt.figure(figsize=(14, 12))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Generate heatmap with customized appearance
            sns.heatmap(
                corr_matrix, 
                mask=mask,
                annot=False, 
                cmap='coolwarm', 
                center=0,
                vmin=-1, 
                vmax=1,
                linewidths=0.5
            )
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            # Save if output path provided
            if output_path:
                # Ensure directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation matrix plot saved to {output_path}")
            
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error plotting correlation matrix: {str(e)}")
    
    # Find highly correlated pairs
    correlated_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlated_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    # Convert to DataFrame
    if correlated_pairs:
        result = pd.DataFrame(correlated_pairs)
        logger.info(f"Correlation analysis completed with {len(result)} highly correlated feature pairs")
        return result
    else:
        logger.info(f"Correlation analysis completed with no feature pairs exceeding threshold {threshold}")
        return pd.DataFrame(columns=['feature1', 'feature2', 'correlation'])


def plot_roc_curves(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    group_values: Optional[np.ndarray] = None,
    group_name: str = "Group",
    protected_attributes: Optional[Dict[str, np.ndarray]] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Plots ROC curves with optional demographic stratification.
    
    Args:
        y_true: True target values
        y_prob: Predicted probabilities
        group_values: Group values for each sample
        group_name: Name of group variable
        protected_attributes: Dictionary mapping attribute names to values
        output_path: Path to save ROC visualization
        logger: Logger for tracking visualization
        
    Returns:
        Dictionary with AUC values
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    logger.info("Plotting ROC curves...")
    
    # If protected_attributes is provided and group_values is not, use the first protected attribute
    if group_values is None and protected_attributes is not None:
        if len(protected_attributes) > 0:
            first_attr_name = list(protected_attributes.keys())[0]
            group_values = protected_attributes[first_attr_name]
            group_name = first_attr_name
            logger.info(f"Using protected attribute '{first_attr_name}' for group-based ROC curves")
    
    auc_values = {}
    
    try:
        plt.figure(figsize=(10, 8))
        
        # If no group values, plot overall ROC curve
        if group_values is None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            auc_values['overall'] = auc
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        else:
            # Plot ROC for each group
            unique_groups = np.unique(group_values)
            
            for group in unique_groups:
                # Filter by group
                group_mask = (group_values == group)
                
                # Skip groups with insufficient data
                if sum(group_mask) < 10 or len(np.unique(y_true[group_mask])) < 2:
                    logger.warning(f"Skipping group '{group}' for ROC curve due to insufficient data")
                    continue
                
                try:
                    fpr, tpr, _ = roc_curve(y_true[group_mask], y_prob[group_mask])
                    auc = roc_auc_score(y_true[group_mask], y_prob[group_mask])
                    auc_values[str(group)] = auc
                    plt.plot(fpr, tpr, lw=2, label=f'{group_name} = {group} (AUC = {auc:.3f})')
                except Exception as e:
                    logger.warning(f"Error calculating ROC for group {group}: {str(e)}")
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Customize appearance
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve' + 
                 (f' by {group_name}' if group_values is not None else ''))
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save if output path provided
        if output_path:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting ROC curves: {str(e)}")
    
    logger.info("ROC curve plotting completed")
    return auc_values


def plot_precision_recall_curves(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    group_values: Optional[np.ndarray] = None,
    group_name: str = "Group",
    protected_attributes: Optional[Dict[str, np.ndarray]] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Plots precision-recall curves with optional demographic stratification.
    
    Args:
        y_true: True target values
        y_prob: Predicted probabilities
        group_values: Group values for each sample
        group_name: Name of group variable
        protected_attributes: Dictionary mapping attribute names to values
        output_path: Path to save PR visualization
        logger: Logger for tracking visualization
        
    Returns:
        Dictionary with average precision values
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    logger.info("Plotting precision-recall curves...")
    
    # If protected_attributes is provided and group_values is not, use the first protected attribute
    if group_values is None and protected_attributes is not None:
        if len(protected_attributes) > 0:
            first_attr_name = list(protected_attributes.keys())[0]
            group_values = protected_attributes[first_attr_name]
            group_name = first_attr_name
            logger.info(f"Using protected attribute '{first_attr_name}' for group-based precision-recall curves")
    
    avg_precision_values = {}
    
    try:
        plt.figure(figsize=(10, 8))
        
        # If no group values, plot overall precision-recall curve
        if group_values is None:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            avg_precision_values['overall'] = avg_precision
            plt.plot(recall, precision, lw=2, label=f'Precision-Recall curve (Avg Precision = {avg_precision:.3f})')
        else:
            # Plot precision-recall for each group
            unique_groups = np.unique(group_values)
            
            for group in unique_groups:
                # Filter by group
                group_mask = (group_values == group)
                
                # Skip groups with insufficient data
                if sum(group_mask) < 10 or len(np.unique(y_true[group_mask])) < 2:
                    logger.warning(f"Skipping group '{group}' for precision-recall curve due to insufficient data")
                    continue
                
                try:
                    precision, recall, _ = precision_recall_curve(y_true[group_mask], y_prob[group_mask])
                    avg_precision = average_precision_score(y_true[group_mask], y_prob[group_mask])
                    avg_precision_values[str(group)] = avg_precision
                    plt.plot(recall, precision, lw=2, label=f'{group_name} = {group} (Avg Precision = {avg_precision:.3f})')
                except Exception as e:
                    logger.warning(f"Error calculating precision-recall for group {group}: {str(e)}")
        
        # Customize appearance
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve' + 
                 (f' by {group_name}' if group_values is not None else ''))
        plt.legend(loc="lower left")
        plt.grid(True)
        
        # Save if output path provided
        if output_path:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve plot saved to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting precision-recall curves: {str(e)}")
    
    logger.info("Precision-recall curve plotting completed")
    return avg_precision_values


def calculate_confidence_intervals(
    metrics: Dict[str, float], 
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray, 
    n_bootstrap: int = 1000, 
    confidence: float = 0.95,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculates confidence intervals for performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        y_true: True target values
        y_pred: Predicted binary values
        y_prob: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed
        logger: Logger for tracking interval calculation
        
    Returns:
        Dictionary with confidence intervals for each metric
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
    
    logger.info(f"Calculating confidence intervals using {n_bootstrap} bootstrap samples...")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # List of metrics to calculate intervals for
    metrics_to_bootstrap = {
        'accuracy': lambda yt, yp, ypr: accuracy_score(yt, yp),
        'precision': lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
        'recall': lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
        'f1': lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
        'auc_roc': lambda yt, yp, ypr: roc_auc_score(yt, ypr) if len(np.unique(yt)) > 1 else float('nan')
    }
    
    # Filter to only include metrics provided in the input
    metrics_to_bootstrap = {k: v for k, v in metrics_to_bootstrap.items() if k in metrics}
    
    bootstrap_results = {metric: [] for metric in metrics_to_bootstrap}
    n_samples = len(y_true)
    
    # Perform bootstrap sampling
    for i in range(n_bootstrap):
        try:
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
            y_prob_sample = y_prob[indices]
            
            # Compute metrics for this bootstrap sample
            for metric_name, metric_func in metrics_to_bootstrap.items():
                bootstrap_results[metric_name].append(metric_func(y_true_sample, y_pred_sample, y_prob_sample))
        except Exception as e:
            logger.warning(f"Error in bootstrap sample {i}: {str(e)}")
    
    # Calculate confidence intervals
    confidence_intervals = {}
    alpha = (1 - confidence) / 2
    
    for metric_name, bootstrap_values in bootstrap_results.items():
        bootstrap_values = np.array(bootstrap_values)
        bootstrap_values = bootstrap_values[~np.isnan(bootstrap_values)]  # Remove NaNs
        
        if len(bootstrap_values) > 0:
            lower_percentile = alpha * 100
            upper_percentile = (1 - alpha) * 100
            
            lower = np.percentile(bootstrap_values, lower_percentile)
            upper = np.percentile(bootstrap_values, upper_percentile)
            
            confidence_intervals[metric_name] = {
                'lower': float(lower),
                'upper': float(upper),
                'mean': float(np.mean(bootstrap_values)),
                'std': float(np.std(bootstrap_values))
            }
        else:
            logger.warning(f"No valid bootstrap samples for metric {metric_name}")
            confidence_intervals[metric_name] = {
                'lower': float('nan'),
                'upper': float('nan'),
                'mean': float('nan'),
                'std': float('nan')
            }
    
    logger.info(f"Confidence intervals calculated for {len(confidence_intervals)} metrics")
    return confidence_intervals