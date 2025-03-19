import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

def analyze_feature_importance(X: pd.DataFrame, y: pd.Series, 
                              n_estimators: int = 100, 
                              random_state: int = 0,
                              plot: bool = True) -> pd.DataFrame:
    """Analyzes feature importance using random forest classifier."""
    
    # select numeric columns for importance analysis
    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'bool']).columns
    X_numeric = X[numeric_cols]
    
    # train random forest classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state
    )
    model.fit(X_numeric, y)
    importances = model.feature_importances_
    
    # convert to df for visualization
    feature_importance = pd.DataFrame({
        'Feature': numeric_cols,
        'Importance': importances
    })
    feature_importance = feature_importance.sort_values(
        'Importance', ascending=False
    )
    
    # plot feature importances
    if plot:
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=feature_importance.head(20)
        )
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    return feature_importance


def analyze_feature_correlations(X: pd.DataFrame, 
                                threshold: float = 0.85, 
                                plot: bool = True) -> pd.DataFrame:
    """Analyzes correlation between numeric features."""
    
    # select numeric columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X_numeric = X[numeric_cols]
    
    # calculate correlation matrix
    corr_matrix = X_numeric.corr()
    
    # plot correlation heatmap
    if plot:
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=False, 
            cmap='coolwarm', 
            center=0,
            vmin=-1, 
            vmax=1
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    # find highly correlated pairs
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlated_pairs.append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    # convert to df
    if correlated_pairs:
        return pd.DataFrame(correlated_pairs)
    else:
        return pd.DataFrame(columns=['Feature1', 'Feature2', 'Correlation'])


def calculate_model_metrics(y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           y_prob: np.ndarray) -> Dict:
    """Calculates comprehensive model performance metrics."""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # add class distribution metrics
    y_true_counts = np.bincount(y_true)
    y_pred_counts = np.bincount(y_pred)
    
    metrics['class_distribution_true'] = {
        0: int(y_true_counts[0]),
        1: int(y_true_counts[1]) if len(y_true_counts) > 1 else 0
    }
    
    metrics['class_distribution_pred'] = {
        0: int(y_pred_counts[0]),
        1: int(y_pred_counts[1]) if len(y_pred_counts) > 1 else 0
    }
    
    return metrics


def calculate_fairness_metrics(y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              y_prob: np.ndarray,
                              protected_attributes: Dict[str, np.ndarray]) -> Dict:
    """Calculates fairness metrics across protected attributes."""
    
    fairness_metrics = {}
    
    for attr_name, attr_values in protected_attributes.items():
        attr_metrics = {}
        
        # get unique values for the protected attribute
        unique_values = np.unique(attr_values)
        
        # calculate metrics for each group
        group_metrics = {}
        
        for value in unique_values:
            # get indices for this group
            group_indices = (attr_values == value)
            
            # skip groups with too few samples
            if np.sum(group_indices) < 10:
                continue
            
            # calculate metrics for this group
            group_y_true = y_true[group_indices]
            group_y_pred = y_pred[group_indices]
            group_y_prob = y_prob[group_indices]
            
            # basic performance metrics
            metrics = {
                'count': int(np.sum(group_indices)),
                'accuracy': accuracy_score(group_y_true, group_y_pred),
                'precision': precision_score(group_y_true, group_y_pred, zero_division=0),
                'recall': recall_score(group_y_true, group_y_pred, zero_division=0),
                'f1': f1_score(group_y_true, group_y_pred, zero_division=0)
            }
            
            # auc only if both classes present
            if len(np.unique(group_y_true)) > 1:
                metrics['auc'] = roc_auc_score(group_y_true, group_y_prob)
            else:
                metrics['auc'] = np.nan
            
            # fairness-specific metrics
            metrics['positive_rate'] = np.mean(group_y_pred)
            metrics['true_positive_rate'] = recall_score(
                group_y_true, group_y_pred, zero_division=0
            )
            metrics['false_positive_rate'] = np.sum(
                (group_y_true == 0) & (group_y_pred == 1)
            ) / max(1, np.sum(group_y_true == 0))
            
            group_metrics[value] = metrics
        
        # calculate disparate impact metrics across groups
        if len(group_metrics) > 1:
            positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
            tpr_values = [metrics['true_positive_rate'] for metrics in group_metrics.values()]
            fpr_values = [metrics['false_positive_rate'] for metrics in group_metrics.values()]
            
            # demographic parity difference
            max_pr = max(positive_rates)
            min_pr = min(positive_rates)
            
            # disparate impact ratio
            disparate_impact = min_pr / max_pr if max_pr > 0 else 1.0
            
            # equal opportunity difference
            equal_opp_diff = max(tpr_values) - min(tpr_values)
            
            # equalized odds differences
            eq_odds_diff_tpr = max(tpr_values) - min(tpr_values)
            eq_odds_diff_fpr = max(fpr_values) - min(fpr_values)
            
            attr_metrics['demographic_parity_difference'] = max_pr - min_pr
            attr_metrics['disparate_impact_ratio'] = disparate_impact
            attr_metrics['equal_opportunity_difference'] = equal_opp_diff
            attr_metrics['equalized_odds_diff_tpr'] = eq_odds_diff_tpr
            attr_metrics['equalized_odds_diff_fpr'] = eq_odds_diff_fpr
            
        attr_metrics['group_metrics'] = group_metrics
        fairness_metrics[attr_name] = attr_metrics
    
    return fairness_metrics


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                  group_values: Optional[np.ndarray] = None,
                  group_name: str = "Group") -> None:
    """Plots roc curves (optionally by demographic group)."""
    
    plt.figure(figsize=(10, 8))
    
    # if no group values, plot overall roc
    if group_values is None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_prob):.3f})')
    else:
        # plot roc for each group
        for group in np.unique(group_values):
            group_mask = (group_values == group)
            
            # only proceed if enough samples and both classes present
            if sum(group_mask) >= 10 and len(np.unique(y_true[group_mask])) > 1:
                fpr, tpr, _ = roc_curve(y_true[group_mask], y_prob[group_mask])
                auc = roc_auc_score(y_true[group_mask], y_prob[group_mask])
                plt.plot(fpr, tpr, lw=2, label=f'{group_name} = {group} (AUC = {auc:.3f})')
    
    # plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve' +
             (f' by {group_name}' if group_values is not None else ''))
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_precision_recall_curves(y_true: np.ndarray, y_prob: np.ndarray,
                               group_values: Optional[np.ndarray] = None,
                               group_name: str = "Group") -> None:
    """Plots precision-recall curves (optionally by demographic group)."""
    
    plt.figure(figsize=(10, 8))
    
    # if no group values, just plot overall PR curve
    if group_values is None:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, lw=2, 
                label=f'Precision-Recall curve (Avg Precision = {np.mean(precision):.3f})')
    else:
        # plot PR curve for each group
        for group in np.unique(group_values):
            group_mask = (group_values == group)
            
            # only proceed if enough samples and both classes present
            if sum(group_mask) >= 10 and len(np.unique(y_true[group_mask])) > 1:
                precision, recall, _ = precision_recall_curve(y_true[group_mask], y_prob[group_mask])
                avg_precision = np.mean(precision)
                plt.plot(recall, precision, lw=2, 
                        label=f'{group_name} = {group} (Avg Prec = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve' +
             (f' by {group_name}' if group_values is not None else ''))
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def plot_fairness_metrics(fairness_results: Dict, metric_name: str = 'f1') -> None:
    """Plots fairness metrics across demographic groups."""
    
    fig, axes = plt.subplots(len(fairness_results), 1, figsize=(12, 4*len(fairness_results)))
    
    if len(fairness_results) == 1:
        axes = [axes]
    
    for i, (attr_name, attr_metrics) in enumerate(fairness_results.items()):
        group_metrics = attr_metrics['group_metrics']
        
        # extract metrics for plotting
        groups = list(group_metrics.keys())
        metric_values = [m[metric_name] for m in group_metrics.values()]
        
        # sort by metric value
        sorted_idx = np.argsort(metric_values)
        sorted_groups = [groups[i] for i in sorted_idx]
        sorted_values = [metric_values[i] for i in sorted_idx]
        
        # plot
        sns.barplot(x=sorted_groups, y=sorted_values, ax=axes[i])
        axes[i].set_title(f'{metric_name.upper()} by {attr_name}')
        axes[i].set_xlabel(attr_name)
        axes[i].set_ylabel(metric_name)
        
        # add variance metrics
        var_metrics = f"Max diff: {max(metric_values) - min(metric_values):.3f}, "
        var_metrics += f"Variance: {np.var(metric_values):.3f}, "
        var_metrics += f"Mean: {np.mean(metric_values):.3f}"
        
        axes[i].text(0.5, -0.15, var_metrics, transform=axes[i].transAxes, 
                    ha='center', va='center', fontsize=10)
        
        # rotate x labels if many groups
        if len(groups) > 5:
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()