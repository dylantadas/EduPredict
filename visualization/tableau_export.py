import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_student_risk_data(
    student_data: pd.DataFrame,
    risk_predictions: np.ndarray,
    risk_probabilities: np.ndarray,
    student_ids: np.ndarray,
    demographic_cols: List[str] = ['gender', 'age_band', 'imd_band', 'region']
) -> pd.DataFrame:
    """Prepares student risk prediction data for tableau export."""
    
    # validate inputs
    if len(risk_predictions) != len(student_ids):
        raise ValueError("Length of predictions must match length of student_ids")
    
    # create dataframe with predictions
    risk_df = pd.DataFrame({
        'id_student': student_ids,
        'risk_prediction': risk_predictions,
        'risk_probability': risk_probabilities
    })
    
    # merge with student demographic data
    cols_to_use = ['id_student', 'code_module', 'code_presentation'] + demographic_cols
    available_cols = [col for col in cols_to_use if col in student_data.columns]
    
    result_df = student_data[available_cols].merge(
        risk_df,
        on='id_student',
        how='inner'
    )
    
    # convert risk probabilities to percentage
    result_df['risk_percentage'] = (result_df['risk_probability'] * 100).round(1)
    
    # add risk category
    risk_bins = [0, 0.3, 0.6, 0.85, 1.0]
    risk_labels = ['Low', 'Moderate', 'High', 'Very High']
    
    result_df['risk_category'] = pd.cut(
        result_df['risk_probability'],
        bins=risk_bins,
        labels=risk_labels,
        include_lowest=True
    )
    
    return result_df


def prepare_temporal_engagement_data(
    vle_data: pd.DataFrame,
    risk_data: pd.DataFrame,
    time_window: int = 7
) -> pd.DataFrame:
    """Prepares temporal engagement data for time-series visualization."""
    
    # group vle data by student and time window
    vle_data['time_window'] = vle_data['date'] // time_window
    
    # aggregate interactions
    engagement_df = vle_data.groupby(
        ['id_student', 'code_module', 'code_presentation', 'time_window']
    ).agg({
        'sum_click': 'sum',
        'id_site': 'nunique'
    }).reset_index()
    
    # rename columns for clarity
    engagement_df = engagement_df.rename(columns={
        'sum_click': 'total_interactions',
        'id_site': 'unique_materials'
    })
    
    # convert time_window to actual days (start of window)
    engagement_df['day'] = engagement_df['time_window'] * time_window
    
    # merge with risk data to include predictions
    risk_cols = ['id_student', 'risk_prediction', 'risk_probability', 'risk_category']
    
    # check if all columns exist
    available_cols = [col for col in risk_cols if col in risk_data.columns]
    
    if available_cols:
        result_df = engagement_df.merge(
            risk_data[available_cols],
            on='id_student',
            how='left'
        )
    else:
        result_df = engagement_df
    
    return result_df


def prepare_assessment_performance_data(
    assessment_data: pd.DataFrame,
    student_info: pd.DataFrame,
    risk_data: pd.DataFrame
) -> pd.DataFrame:
    """Prepares assessment performance data for visualization."""
    
    # merge assessment data with student info
    assessment_df = assessment_data.merge(
        student_info[['id_student', 'code_module', 'code_presentation', 'final_result']],
        on=['id_student', 'code_module', 'code_presentation'],
        how='inner'
    )
    
    # add assessment submission status
    assessment_df['submission_status'] = np.where(
        assessment_df['date_submitted'].notna(),
        'Submitted',
        'Not Submitted'
    )
    
    # add assessment outcome
    assessment_df['assessment_outcome'] = np.where(
        assessment_df['score'] >= 40,
        'Pass',
        'Fail'
    )
    
    # calculate submission delay
    assessment_df['submission_delay'] = (
        assessment_df['date_submitted'] - assessment_df['date']
    )
    
    # add submission timing category
    def categorize_timing(delay):
        if pd.isna(delay):
            return 'Not Submitted'
        elif delay < -7:
            return 'Very Early (>1 week)'
        elif delay < -1:
            return 'Early (1-7 days)'
        elif delay < 0:
            return 'Day Before'
        elif delay == 0:
            return 'On Due Date'
        elif delay <= 1:
            return 'Late (1 day)'
        else:
            return 'Very Late (>1 day)'
    
    assessment_df['submission_timing'] = assessment_df['submission_delay'].apply(categorize_timing)
    
    # merge with risk data
    risk_cols = ['id_student', 'risk_prediction', 'risk_probability', 'risk_category']
    available_cols = [col for col in risk_cols if col in risk_data.columns]
    
    if available_cols:
        result_df = assessment_df.merge(
            risk_data[available_cols],
            on='id_student',
            how='left'
        )
    else:
        result_df = assessment_df
    
    return result_df


def prepare_demographic_fairness_data(
    fairness_results: Dict,
    risk_data: pd.DataFrame,
    demographic_cols: List[str] = ['gender', 'age_band', 'imd_band']
) -> pd.DataFrame:
    """Prepares demographic fairness metrics for visualization."""
    
    fairness_rows = []
    
    # extract fairness metrics for each demographic attribute
    for attr_name, results in fairness_results.items():
        if attr_name not in demographic_cols:
            continue
            
        # get group metrics
        if 'group_metrics' in results:
            group_metrics = results['group_metrics']
            
            # convert to dataframe if not already
            if not isinstance(group_metrics, pd.DataFrame):
                if isinstance(group_metrics, dict):
                    group_df = pd.DataFrame.from_dict(group_metrics, orient='index')
                    group_df['group'] = group_df.index
                else:
                    continue
            else:
                group_df = group_metrics.copy()
                
            # add demographic attribute column
            group_df['demographic_attribute'] = attr_name
            
            # select relevant columns
            metric_cols = ['group', 'demographic_attribute', 'count', 'accuracy', 
                          'f1', 'precision', 'recall', 'positive_rate', 'auc']
            
            available_cols = [col for col in metric_cols if col in group_df.columns]
            fairness_rows.append(group_df[available_cols])
        
        # add overall metrics
        if 'fairness_metrics' in results:
            metrics = results['fairness_metrics']
            if isinstance(metrics, dict):
                overall_df = pd.DataFrame({
                    'demographic_attribute': [attr_name],
                    'metric_type': ['overall'],
                    'demographic_parity_difference': [metrics.get('demographic_parity_difference', np.nan)],
                    'disparate_impact_ratio': [metrics.get('disparate_impact_ratio', np.nan)],
                    'equal_opportunity_difference': [metrics.get('equal_opportunity_difference', np.nan)]
                })
                fairness_rows.append(overall_df)
    
    # combine all fairness metrics
    if fairness_rows:
        fairness_df = pd.concat(fairness_rows, ignore_index=True)
        
        # calculate additional metrics for visualization
        if 'positive_rate' in fairness_df.columns and 'demographic_attribute' in fairness_df.columns:
            # calculate relative positive rate compared to max for each demographic
            for attr in fairness_df['demographic_attribute'].unique():
                attr_mask = (fairness_df['demographic_attribute'] == attr)
                max_rate = fairness_df.loc[attr_mask, 'positive_rate'].max()
                
                if max_rate > 0:
                    fairness_df.loc[attr_mask, 'relative_positive_rate'] = (
                        fairness_df.loc[attr_mask, 'positive_rate'] / max_rate
                    )
    else:
        # create empty dataframe with expected columns
        fairness_df = pd.DataFrame(columns=[
            'demographic_attribute', 'group', 'count', 'accuracy', 'f1', 
            'positive_rate', 'relative_positive_rate'
        ])
    
    return fairness_df


def prepare_model_performance_data(
    model_results: Dict,
    model_name: str,
    demographic_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Prepares model performance metrics for visualization."""
    
    # extract basic performance metrics
    performance_metrics = {
        'model_name': model_name,
        'accuracy': model_results.get('accuracy', np.nan),
        'f1_score': model_results.get('f1_score', np.nan),
        'auc_roc': model_results.get('auc_roc', np.nan),
        'threshold': model_results.get('threshold', 0.5)
    }
    
    # create base dataframe
    performance_df = pd.DataFrame([performance_metrics])
    
    # add confusion matrix data if available
    if 'confusion_matrix' in model_results:
        cm = model_results['confusion_matrix']
        if isinstance(cm, np.ndarray) and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            performance_df['true_positives'] = tp
            performance_df['false_positives'] = fp
            performance_df['true_negatives'] = tn
            performance_df['false_negatives'] = fn
            
            # calculate additional metrics
            performance_df['precision'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            performance_df['recall'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            performance_df['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    return performance_df


def export_for_tableau(
    data_dict: Dict[str, pd.DataFrame],
    export_dir: str = '../tableau_data',
    format: str = 'csv'
) -> Dict[str, str]:
    """Exports prepared dataframes for tableau visualization."""
    
    # create export directory if not exists
    os.makedirs(export_dir, exist_ok=True)
    
    export_paths = {}
    
    # export each dataframe
    for name, df in data_dict.items():
        if format.lower() == 'csv':
            file_path = os.path.join(export_dir, f"{name}.csv")
            df.to_csv(file_path, index=False)
            export_paths[name] = file_path
            
        elif format.lower() == 'excel':
            file_path = os.path.join(export_dir, f"{name}.xlsx")
            df.to_excel(file_path, index=False)
            export_paths[name] = file_path
            
        elif format.lower() == 'json':
            file_path = os.path.join(export_dir, f"{name}.json")
            df.to_json(file_path, orient='records')
            export_paths[name] = file_path
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    # create metadata file with column descriptions
    metadata = {
        'export_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'datasets': {name: {'rows': len(df), 'columns': list(df.columns)} 
                   for name, df in data_dict.items()}
    }
    
    metadata_path = os.path.join(export_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    export_paths['metadata'] = metadata_path
    
    print(f"Exported {len(data_dict)} datasets to {export_dir}")
    return export_paths


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
        risk_counts = risk_data['risk_category'].value_counts().sort_index()
        ax = risk_counts.plot(kind='bar', color=sns.color_palette("YlOrRd", len(risk_counts)))
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
    
    # 2. risk distribution by demographic groups (for each available demographic)
    for demo_col in ['gender', 'age_band', 'imd_band']:
        if demo_col in risk_data.columns and 'risk_category' in risk_data.columns:
            plt.figure(figsize=(12, 7))
            demo_risk = pd.crosstab(
                risk_data[demo_col], 
                risk_data['risk_category'],
                normalize='index'
            )
            
            ax = demo_risk.plot(kind='bar', stacked=True, colormap='YlOrRd')
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