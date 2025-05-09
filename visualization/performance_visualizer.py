import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging
from pathlib import Path
from config import FAIRNESS

logger = logging.getLogger('edupredict')

def visualize_demographic_distributions(student_info: pd.DataFrame, 
                                     save_path: Optional[str] = None) -> None:
    """
    Visualizes distributions of demographic variables.
    
    Args:
        student_info: DataFrame with demographic data
        save_path: Path to save visualization
    """
    try:
        demo_cols = ['gender', 'age_band', 'region', 'imd_band']
        n_cols = len([col for col in demo_cols if col in student_info.columns])
        
        if n_cols == 0:
            logger.warning("No demographic columns found for visualization")
            return
            
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 5*n_cols))
        if n_cols == 1:
            axes = [axes]
            
        plot_idx = 0
        for col in demo_cols:
            if col in student_info.columns:
                # Calculate percentages
                value_counts = student_info[col].value_counts()
                percentages = value_counts / len(student_info) * 100
                
                # Create bar plot
                ax = axes[plot_idx]
                sns.barplot(x=percentages.index, y=percentages.values, ax=ax)
                
                # Add percentage labels
                for i, v in enumerate(percentages.values):
                    ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
                
                ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
                ax.set_ylabel('Percentage of Students')
                
                # Rotate labels if needed
                if len(value_counts) > 5 or col == 'region':
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
                plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Demographic distributions plot saved to {save_path}")
            plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing demographic distributions: {str(e)}")

def visualize_performance_by_demographics(student_data: pd.DataFrame,
                                       demo_cols: List[str] = ['gender', 'age_band', 'imd_band'],
                                       save_path: Optional[str] = None) -> None:
    """
    Visualizes student performance across demographic groups.
    
    Args:
        student_data: DataFrame with student data
        demo_cols: Demographic columns to visualize
        save_path: Path to save visualization
    """
    try:
        available_cols = [col for col in demo_cols if col in student_data.columns]
        n_cols = len(available_cols)
        
        if n_cols == 0:
            logger.warning("No demographic columns available for performance visualization")
            return
            
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 5*n_cols))
        if n_cols == 1:
            axes = [axes]
            
        for idx, col in enumerate(available_cols):
            ax = axes[idx]
            
            # Create boxplot
            sns.boxplot(data=student_data, x=col, y='score', ax=ax)
            
            # Add mean score line
            means = student_data.groupby(col)['score'].mean()
            ax.axhline(y=student_data['score'].mean(), color='r', linestyle='--', 
                      label='Overall Mean')
            
            # Add annotations
            for i, mean_val in enumerate(means):
                ax.text(i, mean_val, f'μ={mean_val:.1f}', ha='center', va='bottom')
                
            ax.set_title(f'Performance Distribution by {col.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            
            # Rotate labels if needed
            if len(student_data[col].unique()) > 5 or col == 'region':
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance by demographics plot saved to {save_path}")
            plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing performance by demographics: {str(e)}")

def visualize_engagement_patterns(vle_data: pd.DataFrame,
                               final_results: Optional[pd.DataFrame] = None,
                               save_path: Optional[str] = None) -> None:
    """
    Visualizes temporal engagement patterns.
    
    Args:
        vle_data: DataFrame with VLE interaction data
        final_results: DataFrame with final results
        save_path: Path to save visualization
    """
    try:
        # Create daily engagement metrics
        daily_engagement = vle_data.groupby(
            pd.Grouper(key='date', freq='1D')
        ).agg({
            'sum_click': ['mean', 'count'],
            'id_student': 'nunique',
            'activity_type': 'nunique'
        }).reset_index()
        
        # Flatten column names
        daily_engagement.columns = [
            'date', 'avg_clicks', 'total_activities',
            'unique_students', 'unique_activity_types'
        ]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot average clicks
        ax1.plot(daily_engagement['date'], daily_engagement['avg_clicks'],
                label='Daily Average', color='#1f77b4')
        
        # Add rolling average
        window = 7
        rolling_avg = daily_engagement['avg_clicks'].rolling(window=window).mean()
        ax1.plot(daily_engagement['date'], rolling_avg,
                label=f'{window}-day Moving Average', color='#ff7f0e', linestyle='--')
        
        ax1.set_title('Daily Engagement Patterns')
        ax1.set_ylabel('Average Clicks per Student')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot unique active students
        ax2.plot(daily_engagement['date'], daily_engagement['unique_students'],
                label='Active Students', color='#2ca02c')
        ax2.set_ylabel('Number of Active Students')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Add final results overlay if available
        if final_results is not None:
            result_counts = final_results['final_result'].value_counts()
            ax2.text(0.02, 0.95, 
                    '\n'.join([f"{k}: {v}" for k, v in result_counts.items()]),
                    transform=ax2.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Engagement patterns plot saved to {save_path}")
            plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing engagement patterns: {str(e)}")

def plot_fairness_metrics(fairness_results: Dict[str, Dict],
                         metric_name: str = 'f1') -> None:
    """
    Plots fairness metrics across demographic groups.
    
    Args:
        fairness_results: Dictionary of fairness results
        metric_name: Metric to plot
    """
    try:
        n_attrs = len(fairness_results)
        if n_attrs == 0:
            logger.warning("No fairness results to plot")
            return
            
        fig, axes = plt.subplots(n_attrs, 1, figsize=(12, 5*n_attrs))
        if n_attrs == 1:
            axes = [axes]
            
        for idx, (attr_name, results) in enumerate(fairness_results.items()):
            ax = axes[idx]
            
            if 'group_metrics' in results:
                group_metrics = results['group_metrics']
                groups = list(group_metrics.keys())
                metric_values = [m.get(metric_name, 0) for m in group_metrics.values()]
                
                # Sort by metric value
                sorted_idx = np.argsort(metric_values)
                sorted_groups = [groups[i] for i in sorted_idx]
                sorted_values = [metric_values[i] for i in sorted_idx]
                
                # Create bar plot
                sns.barplot(x=sorted_groups, y=sorted_values, ax=ax)
                
                # Add threshold line if defined
                if metric_name in FAIRNESS['thresholds']:
                    threshold = FAIRNESS['thresholds'][metric_name]
                    ax.axhline(y=threshold, color='r', linestyle='--',
                             label=f'Threshold ({threshold:.2f})')
                    ax.legend()
                
                # Add value labels
                for i, v in enumerate(sorted_values):
                    ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
                    
                ax.set_title(f'{metric_name.upper()} by {attr_name.replace("_", " ").title()}')
                
                # Rotate labels if needed
                if len(groups) > 5 or attr_name == 'region':
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
    except Exception as e:
        logger.error(f"Error plotting fairness metrics: {str(e)}")

def compare_group_performance(fairness_results: Dict[str, Dict],
                            metric: str = 'f1',
                            save_path: Optional[str] = None) -> None:
    """
    Visualizes model performance across demographic groups.
    
    Args:
        fairness_results: Dictionary of fairness results
        metric: Metric to compare
        save_path: Path to save visualization
    """
    try:
        performance_data = []
        
        for attr, results in fairness_results.items():
            if 'group_metrics' in results:
                group_metrics = results['group_metrics']
                
                for group, metrics in group_metrics.items():
                    if metric in metrics:
                        performance_data.append({
                            'Attribute': attr,
                            'Group': group,
                            'Value': metrics[metric],
                            'Count': metrics.get('count', 0)
                        })
        
        if not performance_data:
            logger.warning(f"No {metric} data available for group comparison")
            return
            
        # Create DataFrame
        performance_df = pd.DataFrame(performance_data)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Create grouped bar plot
        sns.barplot(data=performance_df, x='Attribute', y='Value', hue='Group')
        
        plt.title(f'{metric.upper()} Comparison Across Demographic Groups')
        plt.ylabel(metric.upper())
        
        # Rotate labels if needed
        if len(performance_df['Attribute'].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Add count annotations
        for i, row in performance_df.iterrows():
            plt.text(row.name, row.Value, f'n={row.Count}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Group performance comparison plot saved to {save_path}")
            plt.close()
            
    except Exception as e:
        logger.error(f"Error comparing group performance: {str(e)}")