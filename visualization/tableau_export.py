import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def prepare_student_risk_data(
    student_info: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    student_ids: np.ndarray,
    demographic_cols: List[str]
) -> pd.DataFrame:
    """Prepare student risk predictions with demographic information."""
    
    risk_data = pd.DataFrame({
        'student_id': student_ids,
        'risk_prediction': predictions,
        'risk_probability': probabilities
    })
    
    # Merge with demographic information
    risk_data = risk_data.merge(
        student_info[['id_student'] + demographic_cols],
        left_on='student_id',
        right_on='id_student',
        how='left'
    )
    
    # Create binary risk category
    risk_data['risk_category'] = risk_data['risk_prediction'].map({
        0: 'No Risk',    # Pass/Distinction
        1: 'At Risk'     # Fail/Withdraw
    })
    
    return risk_data


def prepare_temporal_engagement_data(
    vle_data: pd.DataFrame,
    risk_data: pd.DataFrame,
    time_window: int = 7
) -> pd.DataFrame:
    """Prepare temporal engagement data for visualization."""
    
    # Calculate engagement metrics per time window
    engagement = vle_data.groupby(
        ['id_student', pd.Grouper(key='date', freq=f'{time_window}D')]
    ).agg({
        'sum_click': ['sum', 'mean', 'count'],
        'activity_type': 'nunique'
    }).reset_index()
    
    # Flatten column names
    engagement.columns = [
        'student_id', 'date', 'total_clicks', 
        'avg_clicks', 'activities', 'unique_activities'
    ]
    
    # Merge with risk predictions
    engagement = engagement.merge(
        risk_data[['student_id', 'risk_category']],
        on='student_id',
        how='left'
    )
    
    return engagement


def prepare_assessment_performance_data(
    assessment_data: pd.DataFrame,
    student_info: pd.DataFrame,
    risk_data: pd.DataFrame
) -> pd.DataFrame:
    """Prepare assessment performance data for visualization."""
    
    # Calculate assessment metrics
    performance = assessment_data.groupby('id_student').agg({
        'score': ['mean', 'min', 'max', 'std'],
        'weight': 'mean',
        'date_submitted': lambda x: (x - assessment_data['date']).mean()
    }).reset_index()
    
    # Flatten column names
    performance.columns = [
        'student_id', 'avg_score', 'min_score', 
        'max_score', 'score_std', 'avg_weight', 'avg_submission_delay'
    ]
    
    # Merge with risk and demographic data
    performance = performance.merge(
        risk_data[['student_id', 'risk_category']],
        on='student_id',
        how='left'
    )
    
    return performance


def prepare_demographic_fairness_data(
    fairness_results: Dict,
    risk_data: pd.DataFrame,
    demographic_cols: List[str]
) -> pd.DataFrame:
    """Prepare demographic fairness metrics for visualization."""
    
    fairness_data = []
    
    for attr in demographic_cols:
        if attr in fairness_results:
            metrics = fairness_results[attr]['fairness_metrics']
            group_metrics = fairness_results[attr]['group_metrics']
            
            # Calculate group-level statistics
            for group in group_metrics['group'].unique():
                group_data = {
                    'demographic_attribute': attr,
                    'group': group,
                    'sample_size': group_metrics[group_metrics['group'] == group]['count'].iloc[0],
                    'prediction_rate': group_metrics[group_metrics['group'] == group]['positive_rate_pred'].iloc[0],
                    'actual_rate': group_metrics[group_metrics['group'] == group]['positive_rate_true'].iloc[0],
                    'accuracy': group_metrics[group_metrics['group'] == group]['accuracy'].iloc[0],
                    'demographic_parity_diff': metrics['demographic_parity_difference'],
                    'equal_opportunity_diff': metrics['equal_opportunity_difference']
                }
                fairness_data.append(group_data)
    
    return pd.DataFrame(fairness_data)


def prepare_model_performance_data(
    metrics: Dict,
    model_name: str,
    risk_data: pd.DataFrame
) -> pd.DataFrame:
    """Prepare model performance metrics for visualization."""
    
    # Extract overall metrics
    performance_data = {
        'model_name': model_name,
        'accuracy': metrics.get('accuracy', None),
        'f1_score': metrics.get('f1_score', None),
        'auc_roc': metrics.get('auc_roc', None),
        'threshold': metrics.get('threshold', 0.5)
    }
    
    # Calculate risk distribution
    risk_dist = risk_data['risk_category'].value_counts(normalize=True).to_dict()
    performance_data.update({
        f'pct_{k.lower().replace(" ", "_")}': v 
        for k, v in risk_dist.items()
    })
    
    return pd.DataFrame([performance_data])


def export_for_tableau(
    export_data: Dict[str, pd.DataFrame],
    export_dir: str,
    format: str = 'csv'
) -> Dict[str, str]:
    """Export prepared data for Tableau visualization."""
    
    os.makedirs(export_dir, exist_ok=True)
    exported_paths = {}
    
    for name, data in export_data.items():
        filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.{format}"
        filepath = os.path.join(export_dir, filename)
        
        if format == 'csv':
            data.to_csv(filepath, index=False)
        elif format == 'excel':
            data.to_excel(filepath, index=False)
        
        exported_paths[name] = filepath
    
    return exported_paths


def prepare_fairness_visualization_data(
    fairness_results: Dict[str, Dict],
    demographic_data: pd.DataFrame,
    risk_predictions: pd.DataFrame
) -> pd.DataFrame:
    """Prepares fairness metrics and demographic data for visualization.
    
    Args:
        fairness_results: Dictionary containing fairness analysis results
        demographic_data: DataFrame with demographic information
        risk_predictions: DataFrame with model predictions
    
    Returns:
        DataFrame with combined fairness metrics and demographic information
    """
    fairness_rows = []
    
    # Process each protected attribute's results
    for attr_name, results in fairness_results.items():
        # Extract group metrics if available
        group_metrics = results.get('group_metrics', None)
        if not isinstance(group_metrics, pd.DataFrame):
            if isinstance(group_metrics, dict):
                group_df = pd.DataFrame.from_dict(group_metrics, orient='index')
                group_df['group'] = group_df.index
            else:
                continue
        else:
            group_df = group_metrics.copy()
            
        # Add demographic attribute column
        group_df['demographic_attribute'] = attr_name
        
        # Select relevant columns
        metric_cols = ['group', 'demographic_attribute', 'count', 'accuracy', 
                      'f1', 'precision', 'recall', 'positive_rate', 'auc']
        
        available_cols = [col for col in metric_cols if col in group_df.columns]
        fairness_rows.append(group_df[available_cols])
    
    # Combine group metrics
    if fairness_rows:
        fairness_df = pd.concat(fairness_rows, ignore_index=True)
        
        # Calculate relative metrics
        if 'positive_rate' in fairness_df.columns:
            for attr in fairness_df['demographic_attribute'].unique():
                mask = (fairness_df['demographic_attribute'] == attr)
                max_rate = fairness_df.loc[mask, 'positive_rate'].max()
                if max_rate > 0:
                    fairness_df.loc[mask, 'relative_positive_rate'] = (
                        fairness_df.loc[mask, 'positive_rate'] / max_rate
                    )
    else:
        # Create empty DataFrame with expected columns
        fairness_df = pd.DataFrame(columns=[
            'demographic_attribute', 'group', 'count', 'accuracy', 'f1', 
            'positive_rate', 'relative_positive_rate'
        ])
    
    return fairness_df


def generate_summary_visualizations(
    risk_data: pd.DataFrame,
    temporal_data: pd.DataFrame,
    assessment_data: pd.DataFrame,
    fairness_data: pd.DataFrame,
    export_dir: str = '../tableau_assets'
) -> Dict[str, str]:
    """Generates summary visualizations for tableau integration."""
    
    # create export directory if not exists
    os.makedirs(export_dir, exist_ok=True)
    
    export_paths = {}
    
    # set visualization style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. risk distribution by category
    plt.figure(figsize=(10, 6))
    if 'risk_category' in risk_data.columns:
        risk_counts = risk_data['risk_category'].value_counts()
        colors = ['green' if cat == 'No Risk' else 'red' for cat in risk_counts.index]
        ax = risk_counts.plot(kind='bar', color=colors)
        plt.title('Student Risk Distribution', fontsize=14)
        plt.xlabel('Risk Category', fontsize=12)
        plt.ylabel('Number of Students', fontsize=12)
        plt.xticks(rotation=0)
        
        # add percentage labels
        total = risk_counts.sum()
        for i, count in enumerate(risk_counts):
            plt.text(i, count + 5, f'{100 * count / total:.1f}%', ha='center')
        
        file_path = os.path.join(export_dir, 'risk_distribution.png')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        export_paths['risk_distribution'] = file_path
    
    # 2. risk distribution by demographic groups
    for demo_col in ['gender', 'age_band', 'imd_band']:
        if demo_col in risk_data.columns and 'risk_category' in risk_data.columns:
            plt.figure(figsize=(12, 7))
            demo_risk = pd.crosstab(
                risk_data[demo_col], 
                risk_data['risk_category'],
                normalize='index'
            )
            
            # Use specific colors for binary risk
            colors = ['green', 'red']
            ax = demo_risk.plot(kind='bar', stacked=True, color=colors)
            
            plt.title(f'Risk Distribution by {demo_col.replace("_", " ").title()}', fontsize=14)
            plt.xlabel(demo_col.replace("_", " ").title(), fontsize=12)
            plt.ylabel('Percentage of Students', fontsize=12)
            plt.legend(title='Risk Category')
            
            # add percentage annotations
            for i, row in enumerate(demo_risk.itertuples()):
                cumsum = 0
                for j, val in enumerate(row[1:]):
                    cumsum += val
                    plt.text(i, cumsum - val/2, f'{val*100:.1f}%', ha='center')
            
            file_path = os.path.join(export_dir, f'risk_by_{demo_col}.png')
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            export_paths[f'risk_by_{demo_col}'] = file_path
    
    # 3. temporal engagement patterns
    if 'day' in temporal_data.columns and 'total_interactions' in temporal_data.columns:
        plt.figure(figsize=(14, 7))
        
        # aggregate by day
        daily_engagement = temporal_data.groupby('day')['total_interactions'].mean().reset_index()
        
        # plot line chart
        plt.plot(daily_engagement['day'], daily_engagement['total_interactions'], 
                linewidth=2, color='#1f77b4')
        
        plt.title('Average Student Engagement Over Time', fontsize=14)
        plt.xlabel('Days Since Course Start', fontsize=12)
        plt.ylabel('Average Interactions per Student', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # add moving average line
        window = 5
        if len(daily_engagement) > window:
            moving_avg = daily_engagement['total_interactions'].rolling(window=window).mean()
            plt.plot(daily_engagement['day'], moving_avg, 'r--', 
                    linewidth=1.5, label=f'{window}-day Moving Average')
            plt.legend()
        
        file_path = os.path.join(export_dir, 'temporal_engagement.png')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        export_paths['temporal_engagement'] = file_path
    
    # 4. assessment submission patterns
    if 'submission_timing' in assessment_data.columns:
        plt.figure(figsize=(12, 7))
        
        submission_counts = assessment_data['submission_timing'].value_counts().sort_index()
        
        # reorder categories from early to late
        order = [
            'Very Early (>1 week)', 
            'Early (1-7 days)', 
            'Day Before', 
            'On Due Date', 
            'Late (1 day)', 
            'Very Late (>1 day)', 
            'Not Submitted'
        ]
        
        # filter to available categories
        available_order = [cat for cat in order if cat in submission_counts.index]
        submission_counts = submission_counts.reindex(available_order)
        
        # create color palette (green to red)
        colors = sns.color_palette("RdYlGn_r", len(submission_counts))
        
        ax = submission_counts.plot(kind='bar', color=colors)
        plt.title('Assessment Submission Timing', fontsize=14)
        plt.xlabel('Submission Timing', fontsize=12)
        plt.ylabel('Number of Submissions', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # add percentage labels
        total = submission_counts.sum()
        for i, count in enumerate(submission_counts):
            plt.text(i, count + 5, f'{100 * count / total:.1f}%', ha='center')
        
        file_path = os.path.join(export_dir, 'submission_timing.png')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        export_paths['submission_timing'] = file_path
    
    # 5. fairness metrics visualization
    if ('demographic_attribute' in fairness_data.columns and 
        'f1' in fairness_data.columns and
        'group' in fairness_data.columns):
        
        for attr in fairness_data['demographic_attribute'].unique():
            plt.figure(figsize=(12, 6))
            
            attr_data = fairness_data[fairness_data['demographic_attribute'] == attr]
            if len(attr_data) < 2:
                continue
                
            # sort by f1 score
            attr_data = attr_data.sort_values('f1')
            
            # plot f1 scores by group
            ax = sns.barplot(x='group', y='f1', data=attr_data, palette='Blues_d')
            
            plt.title(f'Model Fairness by {attr.replace("_", " ").title()}', fontsize=14)
            plt.xlabel(attr.replace("_", " ").title(), fontsize=12)
            plt.ylabel('F1 Score', fontsize=12)
            plt.ylim(0, 1)
            
            # add value annotations
            for i, v in enumerate(attr_data['f1']):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
            
            # add group sizes
            if 'count' in attr_data.columns:
                plt.xlabel(f'{attr.replace("_", " ").title()} (Group Sizes)')
                ax.set_xticklabels([f'{g}\n(n={c})' for g, c in 
                                  zip(attr_data['group'], attr_data['count'])])
            
            file_path = os.path.join(export_dir, f'fairness_{attr}.png')
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            export_paths[f'fairness_{attr}'] = file_path
    
    return export_paths


def create_tableau_instructions(
    data_paths: Dict[str, str],
    visualization_paths: Dict[str, str],
    output_path: str = '../tableau_assets/instructions.md'
) -> str:
    """Creates instructions for tableau dashboard creation."""
    
    # ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    instructions = """# Tableau Dashboard Creation Instructions

## Data Sources

The following data files have been prepared for Tableau visualization:

"""
    # add data sources
    for name, path in data_paths.items():
        instructions += f"- **{name}**: `{os.path.basename(path)}`\n"
    
    instructions += """
## Pre-generated Visualizations

The following visualizations have been pre-generated and can be incorporated into your Tableau dashboards:

"""
    # add visualizations
    for name, path in visualization_paths.items():
        instructions += f"- **{name}**: `{os.path.basename(path)}`\n"
    
    instructions += """
## Recommended Dashboard Structure

### 1. Student Risk Overview Dashboard

This dashboard should provide a high-level view of student risk across the institution.

Components to include:
- Risk distribution by category
- Risk distribution by demographic groups
- Risk map by geographic region (if available)
- Filters for module, presentation, and demographic attributes

### 2. Engagement Analysis Dashboard

This dashboard should focus on temporal patterns of student engagement.

Components to include:
- Engagement timeline visualization
- Activity type breakdown
- Comparison of engagement patterns by risk category
- Week-by-week engagement heatmap

### 3. Assessment Performance Dashboard

This dashboard should analyze assessment submission and performance patterns.

Components to include:
- Submission timing visualization
- Score distribution by assessment type
- Correlation between submission timing and scores
- Assessment completion rates by risk category

### 4. Model Performance and Fairness Dashboard

This dashboard should present model performance metrics and fairness analysis.

Components to include:
- Overall model performance metrics
- Performance comparison across demographic groups
- Fairness metric visualizations
- Model confidence distribution

## Implementation Steps

1. Connect to the CSV data sources
2. Create calculated fields for any additional metrics
3. Build individual worksheets for each visualization component
4. Combine worksheets into the recommended dashboards
5. Add interactive filters and drill-down capabilities
6. Apply consistent formatting and color schemes
7. Add explanatory text and tooltips

## Notes on Data Structure

- The `risk_data` dataset contains student-level risk predictions
- The `temporal_data` dataset contains time-series engagement data
- The `assessment_data` dataset contains assessment submission and performance data
- The `fairness_data` dataset contains model fairness metrics across demographic groups

## Common Calculations

### Risk Distribution Calculation
```
SUM([Number of Records]) / TOTAL(SUM([Number of Records]))
```

### Fairness Gap Calculation
```
MAX([Performance Metric]) - MIN([Performance Metric])
```
"""
    
    # write instructions file
    with open(output_path, 'w') as f:
        f.write(instructions)
    
    print(f"Created Tableau instructions at {output_path}")
    return output_path