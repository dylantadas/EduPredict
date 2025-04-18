import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

class VisualizationRunner:
    """Centralized runner for all visualizations in EduPredict."""
    
    def __init__(self, 
                 output_dir: str,
                 style: str = 'whitegrid',
                 fig_size: tuple = (12, 8),
                 dpi: int = 300,
                 format: str = 'png'):
        self.output_dir = output_dir
        self.style = style
        self.fig_size = fig_size
        self.dpi = dpi
        self.format = format
        
        # Setup style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = fig_size
        
        # Create output directories
        self.viz_paths = {
            'demographics': os.path.join(output_dir, 'demographics'),
            'performance': os.path.join(output_dir, 'performance'),
            'fairness': os.path.join(output_dir, 'fairness'),
            'engagement': os.path.join(output_dir, 'engagement'),
            'model': os.path.join(output_dir, 'model')
        }
        
        for path in self.viz_paths.values():
            os.makedirs(path, exist_ok=True)

    def run_demographic_visualizations(self, 
                                    student_data: pd.DataFrame,
                                    demo_cols: Optional[List[str]] = None) -> Dict[str, str]:
        """Run all demographic visualizations."""
        if demo_cols is None:
            demo_cols = ['gender', 'age_band', 'imd_band', 'region', 'highest_education']
        
        paths = {}
        
        # Distribution plots
        dist_path = os.path.join(self.viz_paths['demographics'], 'distributions.png')
        self._plot_demographic_distributions(student_data, demo_cols, dist_path)
        paths['distributions'] = dist_path
        
        return paths

    def run_performance_visualizations(self,
                                     data: Dict[str, pd.DataFrame],
                                     metrics: Dict[str, Any]) -> Dict[str, str]:
        """Run all performance-related visualizations."""
        paths = {}
        
        # Performance by demographics
        perf_demo_path = os.path.join(self.viz_paths['performance'], 'performance_by_demographics.png')
        self._plot_performance_by_demographics(
            data['demographics'],
            ['gender', 'age_band', 'imd_band'],
            perf_demo_path
        )
        paths['performance_demographics'] = perf_demo_path
        
        # ROC curves
        roc_path = os.path.join(self.viz_paths['performance'], 'roc_curves.png')
        if 'y_test' in data and 'y_prob' in data:
            self._plot_roc_curves(
                data['y_test'],
                data['y_prob'],
                protected_attributes=data.get('protected_attributes'),
                save_path=roc_path
            )
            paths['roc'] = roc_path
        
        return paths

    def run_fairness_visualizations(self,
                                  fairness_results: Dict,
                                  demographic_data: pd.DataFrame) -> Dict[str, str]:
        """Run all fairness-related visualizations."""
        paths = {}
        
        # Overall fairness metrics
        metrics_path = os.path.join(self.viz_paths['fairness'], 'fairness_metrics.png')
        self._plot_fairness_metrics(fairness_results, metrics_path)
        paths['metrics'] = metrics_path
        
        # Group comparisons
        for attr in ['gender', 'age_band', 'imd_band']:
            if attr in demographic_data.columns:
                group_path = os.join.path(self.viz_paths['fairness'], f'group_comparison_{attr}.png')
                self._plot_group_comparison(fairness_results, attr, group_path)
                paths[f'group_{attr}'] = group_path
        
        return paths

    def run_engagement_visualizations(self,
                                    vle_data: pd.DataFrame,
                                    final_results: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """Run all engagement-related visualizations."""
        paths = {}
        
        # Temporal patterns
        temporal_path = os.path.join(self.viz_paths['engagement'], 'temporal_patterns.png')
        self._plot_engagement_patterns(vle_data, final_results, temporal_path)
        paths['temporal'] = temporal_path
        
        return paths

    def run_model_visualizations(self, 
                               model_data: Dict[str, Any]) -> Dict[str, str]:
        """Run all model-related visualizations."""
        paths = {}
        
        # Feature importance
        if 'feature_importance' in model_data:
            imp_path = os.path.join(self.viz_paths['model'], 'feature_importance.png')
            self._plot_feature_importance(model_data['feature_importance'], imp_path)
            paths['importance'] = imp_path
        
        # Training history for neural networks
        if 'history' in model_data:
            hist_path = os.path.join(self.viz_paths['model'], 'training_history.png')
            self._plot_training_history(model_data['history'], hist_path)
            paths['history'] = hist_path
        
        return paths
    
    def _plot_demographic_distributions(self, data, cols, save_path):
        """Internal method for demographic distribution plots."""
        # Implementation moved from data_analysis.eda
        # ...existing visualization code...
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def _plot_performance_by_demographics(self, data, cols, save_path):
        """Internal method for performance by demographics plots."""
        # Implementation moved from evaluation.performance_metrics
        # ...existing visualization code...
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    # Add other internal plotting methods as needed
