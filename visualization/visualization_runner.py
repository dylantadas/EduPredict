import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

class VisualizationRunner:
    """Handles generation and management of visualizations."""
    
    def __init__(self, output_dir: str):
        """Initialize visualization runner with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default style
        plt.style.use('default')
        
        # Configure plot settings using seaborn
        sns.set_theme()
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def _save_plot(self, name: str) -> str:
        """Save current plot to file and return path."""
        filepath = os.path.join(self.output_dir, f"{name}.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        return filepath
    
    def run_demographic_visualizations(self, demographics_data: pd.DataFrame) -> List[str]:
        """Generate demographic distribution visualizations."""
        viz_paths = []
        
        # Distribution of students by gender
        plt.figure()
        sns.countplot(data=demographics_data, x='gender')
        plt.title('Student Distribution by Gender')
        viz_paths.append(self._save_plot('gender_distribution'))
        
        # Distribution by age band
        plt.figure()
        sns.countplot(data=demographics_data, x='age_band')
        plt.xticks(rotation=45)
        plt.title('Student Distribution by Age Band')
        viz_paths.append(self._save_plot('age_distribution'))
        
        # Distribution by IMD band
        plt.figure()
        sns.countplot(data=demographics_data, x='imd_band')
        plt.xticks(rotation=45)
        plt.title('Student Distribution by IMD Band')
        viz_paths.append(self._save_plot('imd_distribution'))
        
        # Region distribution
        plt.figure()
        demographics_data['region'].value_counts().plot(kind='bar')
        plt.title('Student Distribution by Region')
        plt.xticks(rotation=45)
        viz_paths.append(self._save_plot('region_distribution'))
        
        return viz_paths
    
    def run_engagement_visualizations(self, 
                                   vle_data: pd.DataFrame, 
                                   demographics: pd.DataFrame) -> List[str]:
        """Generate engagement pattern visualizations."""
        viz_paths = []
        
        # Activity type distribution
        plt.figure()
        vle_data['activity_type'].value_counts().plot(kind='bar')
        plt.title('Distribution of VLE Activity Types')
        plt.xticks(rotation=45)
        viz_paths.append(self._save_plot('activity_type_distribution'))
        
        # Daily engagement patterns
        daily_engagement = vle_data.groupby('date')['sum_click'].mean()
        plt.figure()
        daily_engagement.plot(kind='line')
        plt.title('Average Daily Engagement')
        plt.xlabel('Date')
        plt.ylabel('Average Clicks')
        viz_paths.append(self._save_plot('daily_engagement'))
        
        # Engagement by gender
        if 'gender' in demographics.columns:
            merged_data = vle_data.merge(
                demographics[['id_student', 'gender']], 
                on='id_student'
            )
            plt.figure()
            sns.boxplot(data=merged_data, x='gender', y='sum_click')
            plt.title('Engagement Distribution by Gender')
            viz_paths.append(self._save_plot('engagement_by_gender'))
        
        # Engagement by age band
        if 'age_band' in demographics.columns:
            merged_data = vle_data.merge(
                demographics[['id_student', 'age_band']], 
                on='id_student'
            )
            plt.figure()
            sns.boxplot(data=merged_data, x='age_band', y='sum_click')
            plt.title('Engagement Distribution by Age Band')
            plt.xticks(rotation=45)
            viz_paths.append(self._save_plot('engagement_by_age'))
        
        return viz_paths
    
    def run_performance_visualizations(self, 
                                     clean_data: Dict[str, pd.DataFrame],
                                     config: Dict[str, bool]) -> List[str]:
        """Generate performance-related visualizations."""

        viz_paths = []
        
        # Get demographic data if requested
        demographics = None
        if config.get('demographics', False) and 'demographics' in clean_data:
            demographics = clean_data['demographics']
        
        # Performance distribution
        if 'assessments' in clean_data:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=clean_data['assessments'], x='score', bins=20)
            plt.title('Distribution of Assessment Scores')
            plt.xlabel('Score')
            plt.ylabel('Count')
            viz_paths.append(self._save_plot('score_distribution'))
            
            # Performance by demographic groups if available
            if demographics is not None:
                merged_data = clean_data['assessments'].merge(
                    demographics[['id_student', 'gender', 'age_band', 'region']],
                    on='id_student'
                )
                
                # Score by gender
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=merged_data, x='gender', y='score')
                plt.title('Score Distribution by Gender')
                viz_paths.append(self._save_plot('score_by_gender'))
                
                # Score by age band
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=merged_data, x='age_band', y='score')
                plt.title('Score Distribution by Age Band')
                plt.xticks(rotation=45)
                viz_paths.append(self._save_plot('score_by_age'))
                
                # Score by region
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=merged_data, x='region', y='score')
                plt.title('Score Distribution by Region')
                plt.xticks(rotation=45)
                viz_paths.append(self._save_plot('score_by_region'))
        
        return viz_paths

    def visualize_model_performance(self, 
                                  metrics: Dict, 
                                  model_name: str) -> List[str]:
        """Generate model performance visualizations."""
        viz_paths = []
        
        # Confusion matrix heatmap
        if 'confusion_matrix' in metrics:
            plt.figure()
            sns.heatmap(metrics['confusion_matrix'], 
                       annot=True, 
                       fmt='d',
                       cmap='Blues')
            plt.title(f'{model_name} Confusion Matrix')
            viz_paths.append(self._save_plot(f'{model_name.lower()}_confusion_matrix'))
        
        # ROC curve
        if all(k in metrics for k in ['fpr', 'tpr']):
            plt.figure()
            plt.plot(metrics['fpr'], metrics['tpr'])
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC Curve (AUC = {metrics.get("auc_roc", 0):.3f})')
            viz_paths.append(self._save_plot(f'{model_name.lower()}_roc_curve'))
        
        return viz_paths
    
    def visualize_feature_importance(self, 
                                   importance_df: pd.DataFrame, 
                                   top_n: int = 20) -> str:
        """Generate feature importance visualization."""
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=importance_df.head(top_n),
            x='Importance',
            y='Feature'
        )
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        return self._save_plot('feature_importance')
