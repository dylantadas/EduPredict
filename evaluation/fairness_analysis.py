import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from ..config import FAIRNESS_THRESHOLDS


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


def visualize_fairness_metrics(
    fairness_results: Dict,
    save_path: Optional[str] = None
) -> None:
    """Visualizes fairness metrics across demographic groups."""
    
    # prepare data for plotting
    plot_data = []
    
    for attr_name, results in fairness_results.items():
        fairness_metrics = results['fairness_metrics']
        
        # key metrics for visualization
        metrics_to_plot = [
            'demographic_parity_difference',
            'disparate_impact_ratio',
            'equal_opportunity_difference',
            'accuracy_diff',
            'f1_diff'
        ]
        
        for metric in metrics_to_plot:
            if metric in fairness_metrics:
                plot_data.append({
                    'attribute': attr_name,
                    'metric': metric,
                    'value': fairness_metrics[metric]
                })
    
    # convert to dataframe
    plot_df = pd.DataFrame(plot_data)
    
    # create visualization
    plt.figure(figsize=(12, 8))
    
    # create heatmap-style plot
    pivot_df = plot_df.pivot(index='attribute', columns='metric', values='value')
    
    # determine color scale based on metric (some higher is better, some lower is better)
    cmap = plt.cm.RdYlGn.copy()
    
    # create mask for metrics where higher is better (disparate impact)
    higher_better_mask = np.zeros_like(pivot_df.values, dtype=bool)
    if 'disparate_impact_ratio' in pivot_df.columns:
        col_idx = pivot_df.columns.get_loc('disparate_impact_ratio')
        higher_better_mask[:, col_idx] = True
    
    # plot heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        cmap=cmap,
        center=0.5,  # center colormap
        vmin=0, 
        vmax=1,
        fmt='.3f',
        linewidths=0.5,
        cbar_kws={'label': 'Metric Value'}
    )
    
    plt.title('Fairness Metrics by Protected Attribute')
    plt.tight_layout()
    
    # save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved fairness visualization to {save_path}")
    
    plt.show()
    
    # create a bar chart comparing key metrics across attributes
    plt.figure(figsize=(14, 6))
    
    # plot difference metrics (not ratios)
    diff_metrics = ['demographic_parity_difference', 'equal_opportunity_difference', 
                   'accuracy_diff', 'f1_diff']
    diff_df = plot_df[plot_df['metric'].isin(diff_metrics)]
    
    sns.barplot(x='attribute', y='value', hue='metric', data=diff_df)
    plt.title('Fairness Difference Metrics by Protected Attribute')
    plt.ylabel('Difference (lower is better)')
    plt.axhline(y=0.05, color='r', linestyle='--', label='5% Threshold')
    plt.legend(title='Metric')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # save if path provided
    if save_path:
        diff_path = save_path.replace('.png', '_differences.png')
        plt.savefig(diff_path, dpi=300, bbox_inches='tight')
        print(f"Saved fairness differences visualization to {diff_path}")
    
    plt.show()


def compare_group_performance(
    fairness_results: Dict,
    metric: str = 'f1',
    save_path: Optional[str] = None
) -> None:
    """Visualizes model performance across demographic groups for a specific metric."""
    
    plt.figure(figsize=(12, 4 * len(fairness_results)))
    
    # create subplot for each attribute
    fig, axes = plt.subplots(len(fairness_results), 1, figsize=(10, 4 * len(fairness_results)))
    
    if len(fairness_results) == 1:
        axes = [axes]
    
    # plot performance for each attribute
    for i, (attr_name, results) in enumerate(fairness_results.items()):
        group_metrics = results['group_metrics']
        
        # sort by metric value
        sorted_df = group_metrics.sort_values(metric)
        
        # plot
        sns.barplot(x='group', y=metric, data=sorted_df, ax=axes[i])
        axes[i].set_title(f'{metric.upper()} by {attr_name}')
        axes[i].set_xlabel(attr_name)
        axes[i].set_ylabel(metric)
        
        # add overall metric value as reference line
        overall = sorted_df[metric].mean()
        axes[i].axhline(y=overall, color='r', linestyle='--', 
                      label=f'Overall: {overall:.3f}')
        axes[i].legend()
        
        # add value annotations
        for j, bar in enumerate(axes[i].patches):
            axes[i].text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}',
                ha='center'
            )
        
        # set min max for y axis
        axes[i].set_ylim(0, min(1.0, sorted_df[metric].max() * 1.2))
    
    plt.tight_layout()
    
    # save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved group performance visualization to {save_path}")
    
    plt.show()


def generate_fairness_report(
    fairness_results: Dict,
    thresholds: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> str:
    """Generates text report summarizing fairness evaluation results."""
    
    if thresholds is None:
        thresholds = {
            'demographic_parity_difference': 0.05,
            'disparate_impact_ratio': 0.8,
            'equal_opportunity_difference': 0.05
        }
    
    report_lines = ["# Fairness Evaluation Report\n"]
    
    # overall summary
    all_pass = all(results.get('passes_all_thresholds', False) 
                  for results in fairness_results.values())
    
    if all_pass:
        report_lines.append("## SUMMARY: Model passes all fairness thresholds\n")
    else:
        report_lines.append("## SUMMARY: Model fails some fairness thresholds\n")
    
    # threshold details
    report_lines.append("### Fairness Thresholds\n")
    for metric, threshold in thresholds.items():
        if metric == 'disparate_impact_ratio':
            report_lines.append(f"- {metric}: ≥ {threshold}")
        else:
            report_lines.append(f"- {metric}: ≤ {threshold}")
    report_lines.append("")
    
    # results by attribute
    report_lines.append("### Results by Protected Attribute\n")
    
    for attr_name, results in fairness_results.items():
        fairness_metrics = results['fairness_metrics']
        threshold_results = results.get('threshold_results', {})
        
        report_lines.append(f"#### {attr_name}\n")
        
        # threshold results
        for metric, passed in threshold_results.items():
            status = "PASS" if passed else "FAIL"
            value = fairness_metrics[metric]
            threshold = thresholds[metric]
            
            if metric == 'disparate_impact_ratio':
                comparison = ">=" if value >= threshold else "<"
                report_lines.append(f"- {status} {metric}: {value:.3f} {comparison} {threshold}")
            else:
                comparison = "<=" if value <= threshold else ">"
                report_lines.append(f"- {status} {metric}: {value:.3f} {comparison} {threshold}")
        
        # additional metrics
        report_lines.append("\nAdditional metrics:")
        for metric, value in fairness_metrics.items():
            if metric not in threshold_results:
                report_lines.append(f"- {metric}: {value:.3f}")
        
        # group details
        group_metrics = results['group_metrics']
        report_lines.append(f"\nGroups: {len(group_metrics)}")
        report_lines.append(f"Group sizes: min={fairness_metrics['min_group_size']}, "
                          f"max={fairness_metrics['max_group_size']}\n")
    
    # recommendations
    report_lines.append("### Recommendations\n")
    
    if all_pass:
        report_lines.append("- Model meets all fairness criteria. Continuous monitoring is recommended.")
    else:
        # add specific recommendations based on failures
        failing_attributes = []
        failing_metrics = []
        
        for attr_name, results in fairness_results.items():
            threshold_results = results.get('threshold_results', {})
            for metric, passed in threshold_results.items():
                if not passed:
                    if attr_name not in failing_attributes:
                        failing_attributes.append(attr_name)
                    if metric not in failing_metrics:
                        failing_metrics.append(metric)
        
        if 'demographic_parity_difference' in failing_metrics:
            report_lines.append("- **Prediction rate imbalance**: Consider post-processing techniques like "
                              "threshold adjustment for different groups.")
        
        if 'disparate_impact_ratio' in failing_metrics:
            report_lines.append("- **Disparate impact detected**: Additional pre-processing of training data "
                              "may be needed to address underlying biases.")
        
        if 'equal_opportunity_difference' in failing_metrics:
            report_lines.append("- **Equal opportunity violation**: Model has different true positive rates "
                              "across groups. Consider adjusting the loss function to penalize these disparities.")
        
        # add attribute-specific recommendations
        for attr in failing_attributes:
            report_lines.append(f"- **Issues with {attr}**: Consider collecting more training data for "
                              f"underrepresented groups in {attr}.")
    
    # conclusion
    report_lines.append("\n### Conclusion\n")
    if all_pass:
        report_lines.append("The model demonstrates fair performance across all demographic groups "
                          "based on the defined thresholds. Regular monitoring is recommended to "
                          "ensure continued fairness as the model is deployed and updated.")
    else:
        report_lines.append("The model requires additional fairness interventions before deployment. "
                          "Consider implementing the recommendations above to address the specific "
                          "fairness concerns identified in this evaluation.")
    
    # generate the full report
    full_report = "\n".join(report_lines)
    
    # save to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(full_report)
        print(f"Saved fairness report to {save_path}")
    
    return full_report


def analyze_subgroup_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    protected_attributes: Dict[str, np.ndarray],
    metrics: List[str] = ['f1', 'accuracy'],
    min_group_size: int = 30
) -> pd.DataFrame:
    """analyzes model fairness for intersectional subgroups (e.g., gender AND age)"""
    
    # prepare data for analysis
    data = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    
    # add protected attributes
    for attr_name, attr_values in protected_attributes.items():
        data[attr_name] = attr_values
    
    # create all subgroup combinations
    attr_names = list(protected_attributes.keys())
    if len(attr_names) < 2:
        raise ValueError("At least two protected attributes are required for subgroup analysis")
    
    # start with subgroups of size 2
    subgroups = []
    for i in range(len(attr_names)):
        for j in range(i+1, len(attr_names)):
            subgroups.append([attr_names[i], attr_names[j]])
    
    # calculate metrics for each subgroup
    subgroup_results = []
    
    for subgroup_attrs in subgroups:
        # create subgroup identifier
        data['subgroup'] = data[subgroup_attrs].apply(
            lambda x: '_'.join(map(str, x)), axis=1
        )
        
        # get unique subgroups
        unique_subgroups = data['subgroup'].unique()
        
        for subgroup in unique_subgroups:
            # get subgroup data
            subgroup_mask = (data['subgroup'] == subgroup)
            subgroup_size = np.sum(subgroup_mask)
            
            # skip if too small
            if subgroup_size < min_group_size:
                continue
            
            # calculate metrics
            sg_y_true = data.loc[subgroup_mask, 'y_true'].values
            sg_y_pred = data.loc[subgroup_mask, 'y_pred'].values
            sg_y_prob = data.loc[subgroup_mask, 'y_prob'].values
            
            # only proceed if both classes present
            if len(np.unique(sg_y_true)) < 2:
                continue
            
            # calculate basic metrics
            subgroup_metrics = {
                'subgroup': subgroup,
                'attributes': '+'.join(subgroup_attrs),
                'size': subgroup_size,
                'positive_rate_true': sg_y_true.mean(),
                'positive_rate_pred': sg_y_pred.mean()
            }
            
            # add requested metrics
            for metric in metrics:
                if metric == 'accuracy':
                    subgroup_metrics[metric] = accuracy_score(sg_y_true, sg_y_pred)
                elif metric == 'precision':
                    subgroup_metrics[metric] = precision_score(sg_y_true, sg_y_pred, zero_division=0)
                elif metric == 'recall':
                    subgroup_metrics[metric] = recall_score(sg_y_true, sg_y_pred, zero_division=0)
                elif metric == 'f1':
                    subgroup_metrics[metric] = f1_score(sg_y_true, sg_y_pred, zero_division=0)
                elif metric == 'auc':
                    subgroup_metrics[metric] = roc_auc_score(sg_y_true, sg_y_prob)
            
            subgroup_results.append(subgroup_metrics)
    
    if not subgroup_results:
        return pd.DataFrame()
    
    # convert to dataframe
    subgroup_df = pd.DataFrame(subgroup_results)
    
    # calculate overall metrics for comparison
    overall_metrics = {}
    for metric in ['positive_rate_true', 'positive_rate_pred'] + metrics:
        if metric == 'positive_rate_true':
            overall_metrics[metric] = y_true.mean()
        elif metric == 'positive_rate_pred':
            overall_metrics[metric] = y_pred.mean()
        elif metric == 'accuracy':
            overall_metrics[metric] = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            overall_metrics[metric] = precision_score(y_true, y_pred)
        elif metric == 'recall':
            overall_metrics[metric] = recall_score(y_true, y_pred)
        elif metric == 'f1':
            overall_metrics[metric] = f1_score(y_true, y_pred)
        elif metric == 'auc':
            overall_metrics[metric] = roc_auc_score(y_true, y_prob)
    
    # add relative differences
    for metric in ['positive_rate_pred'] + metrics:
        if metric in overall_metrics:
            overall_value = overall_metrics[metric]
            subgroup_df[f'{metric}_diff'] = subgroup_df[metric] - overall_value
            subgroup_df[f'{metric}_ratio'] = subgroup_df[metric] / overall_value
    
    return subgroup_df


def mitigate_bias_with_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    protected_attribute: np.ndarray,
    metric: str = 'demographic_parity',
    tolerance: float = 0.01,
    max_iterations: int = 100
) -> Dict[Any, float]:
    """finds group-specific thresholds to mitigate bias"""
    
    # find unique groups
    unique_groups = np.unique(protected_attribute)
    
    if len(unique_groups) < 2:
        return {unique_groups[0]: 0.5}
    
    # start with default threshold for all groups
    thresholds = {group: 0.5 for group in unique_groups}
    
    # iteratively adjust thresholds
    for iteration in range(max_iterations):
        # apply current thresholds to get predictions
        y_pred = np.zeros_like(y_true)
        
        for group in unique_groups:
            group_mask = (protected_attribute == group)
            group_threshold = thresholds[group]
            y_pred[group_mask] = (y_prob[group_mask] >= group_threshold).astype(int)
        
        # calculate current fairness metric
        if metric == 'demographic_parity':
            # calculate prediction rates by group
            group_pred_rates = {}
            for group in unique_groups:
                group_mask = (protected_attribute == group)
                group_pred_rates[group] = y_pred[group_mask].mean()
            
            # check if groups are within tolerance
            min_rate = min(group_pred_rates.values())
            max_rate = max(group_pred_rates.values())
            
            if max_rate - min_rate <= tolerance:
                break
            
            # adjust thresholds based on prediction rates
            target_rate = sum(y_pred) / len(y_pred)  # overall prediction rate
            
            for group in unique_groups:
                if group_pred_rates[group] > target_rate:
                    # increase threshold to reduce positive predictions
                    thresholds[group] += 0.01
                elif group_pred_rates[group] < target_rate:
                    # decrease threshold to increase positive predictions
                    thresholds[group] -= 0.01
                
                # ensure thresholds are in [0, 1]
                thresholds[group] = max(0, min(1, thresholds[group]))
        
        elif metric == 'equal_opportunity':
            # calculate TPR by group
            group_tpr = {}
            for group in unique_groups:
                group_mask = (protected_attribute == group)
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                # only consider groups with positive examples
                if np.sum(group_y_true) > 0:
                    group_tpr[group] = np.sum((group_y_true == 1) & (group_y_pred == 1)) / np.sum(group_y_true)
                else:
                    group_tpr[group] = 0
            
            # check if groups are within tolerance
            if group_tpr:
                min_tpr = min(group_tpr.values())
                max_tpr = max(group_tpr.values())
                
                if max_tpr - min_tpr <= tolerance:
                    break
                
                # adjust thresholds based on TPR
                for group in unique_groups:
                    if group in group_tpr:
                        if group_tpr[group] > max_tpr - tolerance:
                            # increase threshold to reduce TP
                            thresholds[group] += 0.01
                        elif group_tpr[group] < min_tpr + tolerance:
                            # decrease threshold to increase TP
                            thresholds[group] -= 0.01
                        
                        # ensure thresholds are in [0, 1]
                        thresholds[group] = max(0, min(1, thresholds[group]))
        
        else:
            raise ValueError(f"Unsupported fairness metric: {metric}")
    
    return thresholds