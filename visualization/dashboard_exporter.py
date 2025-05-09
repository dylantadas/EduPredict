import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from config import FAIRNESS, DIRS

logger = logging.getLogger('edupredict')

def prepare_tableau_data(datasets: Dict[str, pd.DataFrame], 
                        predictions: pd.DataFrame, 
                        model_metrics: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Prepares data for Tableau dashboards.
    
    Args:
        datasets: Dictionary of dataset DataFrames
        predictions: DataFrame with predictions
        model_metrics: Dictionary of model metrics
        
    Returns:
        Dictionary of prepared DataFrames for Tableau
    """
    tableau_data = {}
    
    try:
        # Prepare student risk data
        student_data = datasets.get('demographics', pd.DataFrame())
        if not student_data.empty and not predictions.empty:
            risk_data = pd.merge(
                student_data,
                predictions,
                on='id_student',
                how='inner'
            )
            tableau_data['risk_distribution'] = risk_data
            
        # Prepare temporal engagement data
        vle_data = datasets.get('vle', pd.DataFrame())
        if not vle_data.empty:
            engagement_data = vle_data.groupby(
                ['id_student', pd.Grouper(key='date', freq='1D')]
            ).agg({
                'sum_click': ['sum', 'mean', 'count'],
                'activity_type': 'nunique'
            }).reset_index()
            
            # Flatten column names
            engagement_data.columns = [
                'student_id', 'date', 'total_clicks', 
                'avg_clicks', 'activities', 'unique_activities'
            ]
            tableau_data['engagement'] = engagement_data
            
        # Prepare performance metrics
        if model_metrics:
            metrics_df = pd.DataFrame([model_metrics])
            tableau_data['model_performance'] = metrics_df
            
        return tableau_data
        
    except Exception as e:
        logger.error(f"Error preparing Tableau data: {str(e)}")
        return {}

def export_risk_distribution_data(student_data: pd.DataFrame, 
                                predictions: pd.DataFrame, 
                                output_path: str) -> str:
    """
    Exports risk distribution data for visualization.
    
    Args:
        student_data: DataFrame with student data
        predictions: DataFrame with predictions
        output_path: Path to save export
        
    Returns:
        Path to saved file
    """
    try:
        # Merge student data with predictions
        risk_data = pd.merge(
            student_data,
            predictions,
            on='id_student',
            how='inner'
        )
        
        # Calculate risk distributions by demographic groups
        demo_cols = ['gender', 'age_band', 'region', 'imd_band']
        risk_distributions = {}
        
        for col in demo_cols:
            if col in risk_data.columns:
                dist = pd.crosstab(
                    risk_data[col],
                    risk_data['risk_category'],
                    normalize='index'
                )
                risk_distributions[col] = dist
                
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        for group, dist in risk_distributions.items():
            dist.to_csv(str(output_file.parent / f'risk_dist_{group}.csv'))
            
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error exporting risk distribution data: {str(e)}")
        return ""

def export_demographic_performance_data(metrics_by_group: Dict[str, pd.DataFrame], 
                                     output_path: str) -> str:
    """
    Exports performance data by demographic group.
    
    Args:
        metrics_by_group: Dictionary of metrics by group
        output_path: Path to save export
        
    Returns:
        Path to saved file
    """
    try:
        performance_data = []
        
        for attr, metrics in metrics_by_group.items():
            if isinstance(metrics, pd.DataFrame):
                metrics['demographic_attribute'] = attr
                performance_data.append(metrics)
                
        if performance_data:
            combined_metrics = pd.concat(performance_data, ignore_index=True)
            
            # Calculate relative metrics
            for attr in combined_metrics['demographic_attribute'].unique():
                mask = (combined_metrics['demographic_attribute'] == attr)
                max_rate = combined_metrics.loc[mask, 'positive_rate'].max()
                if max_rate > 0:
                    combined_metrics.loc[mask, 'relative_rate'] = (
                        combined_metrics.loc[mask, 'positive_rate'] / max_rate
                    )
            
            # Save to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            combined_metrics.to_csv(str(output_file), index=False)
            
            return str(output_file)
            
    except Exception as e:
        logger.error(f"Error exporting demographic performance data: {str(e)}")
        return ""

def export_temporal_engagement_data(vle_data: pd.DataFrame,
                                 predictions: pd.DataFrame, 
                                 output_path: str) -> str:
    """
    Exports temporal engagement data with predictions.
    
    Args:
        vle_data: DataFrame with VLE interaction data
        predictions: DataFrame with predictions
        output_path: Path to save export
        
    Returns:
        Path to saved file
    """
    try:
        # Aggregate daily engagement
        daily_engagement = vle_data.groupby(
            ['id_student', pd.Grouper(key='date', freq='1D')]
        ).agg({
            'sum_click': ['sum', 'mean', 'count'],
            'activity_type': 'nunique'
        }).reset_index()
        
        # Flatten column names
        daily_engagement.columns = [
            'student_id', 'date', 'total_clicks', 
            'avg_clicks', 'activities', 'unique_activities'
        ]
        
        # Add risk predictions
        if not predictions.empty:
            daily_engagement = daily_engagement.merge(
                predictions[['id_student', 'risk_category']],
                left_on='student_id',
                right_on='id_student',
                how='left'
            )
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        daily_engagement.to_csv(str(output_file), index=False)
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error exporting temporal engagement data: {str(e)}")
        return ""

def export_assessment_performance_data(assessment_data: pd.DataFrame,
                                    output_path: str) -> str:
    """
    Exports assessment performance data.
    
    Args:
        assessment_data: DataFrame with assessment data
        output_path: Path to save export
        
    Returns:
        Path to saved file
    """
    try:
        # Calculate assessment metrics
        performance = assessment_data.groupby('id_student').agg({
            'score': ['mean', 'min', 'max', 'std'],
            'weight': 'mean',
            'date_submitted': lambda x: (x - assessment_data['date']).mean()
        }).reset_index()
        
        # Flatten column names
        performance.columns = [
            'student_id', 'avg_score', 'min_score', 
            'max_score', 'score_std', 'avg_weight', 
            'avg_submission_delay'
        ]
        
        # Add timing categories
        performance['submission_timing'] = pd.cut(
            performance['avg_submission_delay'].dt.days,
            bins=[-float('inf'), -7, -1, 0, 1, float('inf')],
            labels=['Very Early (>1 week)', 'Early (1-7 days)', 
                   'On Due Date', 'Late (1 day)', 'Very Late (>1 day)']
        )
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        performance.to_csv(str(output_file), index=False)
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error exporting assessment performance data: {str(e)}")
        return ""