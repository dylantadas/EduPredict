import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from config import FEATURE_ENGINEERING

logger = logging.getLogger('edupredict')

def visualize_feature_importance(importance_df: pd.DataFrame,
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
    """
    Visualizes top feature importances.
    
    Args:
        importance_df: DataFrame with feature importances
        top_n: Number of top features to show
        save_path: Path to save visualization
    """
    try:
        # Select top N features
        top_features = importance_df.head(min(top_n, len(importance_df)))
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar plot
        sns.barplot(
            data=top_features,
            y='feature',
            x='importance',
            palette='viridis'
        )
        
        plt.title(f'Top {len(top_features)} Most Important Features')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        
        # Add value annotations
        for i, v in enumerate(top_features['importance']):
            plt.text(v, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
            plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing feature importance: {str(e)}")

def plot_correlation_heatmap(X: pd.DataFrame,
                           threshold: float = 0.85,
                           save_path: Optional[str] = None) -> None:
    """
    Plots correlation heatmap for features.
    
    Args:
        X: Feature DataFrame
        threshold: Correlation threshold to highlight
        save_path: Path to save visualization
    """
    try:
        # Select numeric columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns available for correlation analysis")
            return
            
        # Calculate correlation matrix
        corr_matrix = X[numeric_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        
        # Create visualization
        plt.figure(figsize=(15, 12))
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5
        )
        
        plt.title('Feature Correlation Matrix')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
            plt.close()
            
        # Log highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            logger.info("Highly correlated feature pairs:")
            for feat1, feat2, corr in high_corr_pairs:
                logger.info(f"{feat1} - {feat2}: {corr:.3f}")
            
    except Exception as e:
        logger.error(f"Error plotting correlation heatmap: {str(e)}")

def visualize_engagement_over_time(vle_data: pd.DataFrame,
                                student_ids: List[int],
                                metric: str = 'sum_click',
                                save_path: Optional[str] = None) -> None:
    """
    Visualizes student engagement over time.
    
    Args:
        vle_data: DataFrame with VLE interaction data
        student_ids: List of student IDs to visualize
        metric: Engagement metric to visualize
        save_path: Path to save visualization
    """
    try:
        if metric not in vle_data.columns:
            logger.warning(f"Metric {metric} not found in VLE data")
            return
            
        # Filter data for selected students
        student_data = vle_data[vle_data['id_student'].isin(student_ids)]
        
        if len(student_data) == 0:
            logger.warning("No data found for selected students")
            return
            
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Plot engagement lines for each student
        for student_id in student_ids:
            student_timeline = student_data[student_data['id_student'] == student_id]
            if len(student_timeline) > 0:
                plt.plot(
                    student_timeline['date'],
                    student_timeline[metric],
                    label=f'Student {student_id}',
                    marker='o',
                    markersize=4,
                    alpha=0.7
                )
        
        plt.title('Student Engagement Over Time')
        plt.xlabel('Date')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Engagement timeline plot saved to {save_path}")
            plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing engagement over time: {str(e)}")

def visualize_ensemble_weights(results_df: pd.DataFrame,
                             metric: str = 'f1',
                             save_path: Optional[str] = None) -> None:
    """
    Visualizes ensemble weight optimization results.
    
    Args:
        results_df: DataFrame with optimization results
        metric: Metric used for optimization
        save_path: Path to save visualization
    """
    try:
        if 'weights' not in results_df.columns or metric not in results_df.columns:
            logger.warning("Required columns not found in results DataFrame")
            return
            
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(
            range(len(results_df)),
            results_df[metric],
            alpha=0.6,
            c=results_df[metric],
            cmap='viridis'
        )
        
        # Add best result marker
        best_idx = results_df[metric].idxmax()
        plt.scatter(
            best_idx,
            results_df.loc[best_idx, metric],
            color='red',
            s=100,
            label='Best weights',
            zorder=5
        )
        
        plt.title(f'Ensemble Weight Optimization ({metric.upper()})')
        plt.xlabel('Optimization Step')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add weight annotation for best result
        best_weights = results_df.loc[best_idx, 'weights']
        weight_text = '\n'.join([f'w{i+1}: {w:.3f}' for i, w in enumerate(best_weights)])
        plt.annotate(
            f'Best weights:\n{weight_text}',
            xy=(best_idx, results_df.loc[best_idx, metric]),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Ensemble weights plot saved to {save_path}")
            plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing ensemble weights: {str(e)}")

def visualize_tuning_results(results_df: pd.DataFrame,
                           x_col: str,
                           y_col: str,
                           hue_col: Optional[str] = None,
                           title: str = 'Hyperparameter Tuning Results',
                           save_path: Optional[str] = None) -> None:
    """
    Visualizes hyperparameter tuning results.
    
    Args:
        results_df: DataFrame with tuning results
        x_col: Column for x-axis
        y_col: Column for y-axis
        hue_col: Column for color
        title: Plot title
        save_path: Path to save visualization
    """
    try:
        if x_col not in results_df.columns or y_col not in results_df.columns:
            logger.warning("Required columns not found in results DataFrame")
            return
            
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot
        if hue_col and hue_col in results_df.columns:
            sns.scatterplot(
                data=results_df,
                x=x_col,
                y=y_col,
                hue=hue_col,
                alpha=0.6
            )
        else:
            sns.scatterplot(
                data=results_df,
                x=x_col,
                y=y_col,
                alpha=0.6
            )
        
        # Add best result marker
        best_idx = results_df[y_col].idxmax()
        plt.scatter(
            results_df.loc[best_idx, x_col],
            results_df.loc[best_idx, y_col],
            color='red',
            s=100,
            label='Best Result',
            zorder=5
        )
        
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add best result annotation
        best_params = results_df.loc[best_idx]
        param_text = f'Best {y_col}: {best_params[y_col]:.3f}\n'
        param_text += f'{x_col}: {best_params[x_col]}'
        if hue_col:
            param_text += f'\n{hue_col}: {best_params[hue_col]}'
            
        plt.annotate(
            param_text,
            xy=(best_params[x_col], best_params[y_col]),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Tuning results plot saved to {save_path}")
            plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing tuning results: {str(e)}")

def plot_model_comparison_curves(models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                              curve_type: str = 'roc',
                              title: str = 'Model Comparison',
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Creates performance comparison curves for multiple models.
    
    Args:
        models_data: Dict mapping model names to (y_true, y_pred_proba) tuples
        curve_type: Type of curve to plot ('roc' or 'precision_recall')
        title: Plot title
        save_path: Path to save visualization
        
    Returns:
        Matplotlib figure object
    """
    try:
        from sklearn.metrics import (
            roc_curve, auc, 
            precision_recall_curve, average_precision_score
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Process each model's data
        for model_name, (y_true, y_pred_proba) in models_data.items():
            if curve_type.lower() == 'roc':
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                ax.plot(
                    fpr, tpr, 
                    lw=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})'
                )
                
            elif curve_type.lower() == 'precision_recall':
                # Calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                avg_precision = average_precision_score(y_true, y_pred_proba)
                
                # Plot precision-recall curve
                ax.plot(
                    recall, precision, 
                    lw=2, 
                    label=f'{model_name} (AP = {avg_precision:.3f})'
                )
            else:
                logger.warning(f"Unsupported curve type: {curve_type}")
                return None
        
        # Customize plot based on curve type
        if curve_type.lower() == 'roc':
            # Plot diagonal reference line
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            
        elif curve_type.lower() == 'precision_recall':
            # Plot baseline reference line for precision-recall curve
            baseline = sum(y_true) / len(y_true)  # Positive class prevalence
            ax.plot([0, 1], [baseline, baseline], 'k--', lw=1.5, alpha=0.7, 
                    label=f'Baseline (Prevalence = {baseline:.3f})')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
        
        # Add grid, title and legend
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison curves saved to {save_path}")
            
        return fig
            
    except Exception as e:
        logger.error(f"Error creating model comparison curves: {str(e)}")
        return None