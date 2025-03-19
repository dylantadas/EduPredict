import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EnsembleModel:
    """Ensemble model combining static and sequential predictions."""
    
    def __init__(self, 
                static_weight: float = 0.5,
                sequential_weight: float = 0.5,
                threshold: float = 0.5):
        """Initializes ensemble with prediction weights."""
        
        # validate weights sum to 1
        if abs(static_weight + sequential_weight - 1.0) > 1e-10:
            raise ValueError("Weights must sum to 1.0")
        
        self.static_weight = static_weight
        self.sequential_weight = sequential_weight
        self.threshold = threshold
        self.static_model = None
        self.sequential_model = None
        self.optimized = False
    
    def set_models(self, static_model, sequential_model):
        """Sets component models for ensemble."""
        
        self.static_model = static_model
        self.sequential_model = sequential_model
        return self
    
    def predict_proba(self, 
                     static_features: pd.DataFrame, 
                     sequential_features: Dict[str, np.ndarray],
                     student_id_map: Optional[Dict[int, int]] = None) -> np.ndarray:
        """Predicts risk probabilities using weighted ensemble."""
        
        if self.static_model is None or self.sequential_model is None:
            raise ValueError("Both static and sequential models must be set")
        
        # get predictions from static model
        static_probs = self.static_model.predict_proba(static_features)
        
        # get predictions from sequential model
        sequential_probs = self.sequential_model.predict_proba(sequential_features)
        
        # if student id map provided, align predictions
        if student_id_map is not None:
            # create mapping from student IDs to positions in sequential predictions
            aligned_static_probs = np.zeros_like(sequential_probs)
            
            for i, student_id in enumerate(static_features['id_student']):
                if student_id in student_id_map:
                    seq_idx = student_id_map[student_id]
                    aligned_static_probs[seq_idx] = static_probs[i]
            
            static_probs = aligned_static_probs
        
        # weighted combination
        ensemble_probs = (self.static_weight * static_probs + 
                         self.sequential_weight * sequential_probs)
        
        return ensemble_probs
    
    def predict(self, 
               static_features: pd.DataFrame, 
               sequential_features: Dict[str, np.ndarray],
               student_id_map: Optional[Dict[int, int]] = None) -> np.ndarray:
        """Predicts binary risk class using weighted ensemble and threshold."""
        
        probs = self.predict_proba(static_features, sequential_features, student_id_map)
        return (probs >= self.threshold).astype(int)
    
    def optimize_weights(self,
                        static_features_val: pd.DataFrame,
                        sequential_features_val: Dict[str, np.ndarray],
                        y_val: np.ndarray,
                        student_id_map: Dict[int, int],
                        metric: str = 'f1',
                        weight_grid: int = 11):
        """Optimizes ensemble weights using grid search."""
        
        if self.static_model is None or self.sequential_model is None:
            raise ValueError("Both static and sequential models must be set")
        
        # get predictions from both models
        static_probs = self.static_model.predict_proba(static_features_val)
        sequential_probs = self.sequential_model.predict_proba(sequential_features_val)
        
        # align static predictions with sequential order if needed
        aligned_static_probs = np.zeros_like(sequential_probs)
        for i, student_id in enumerate(static_features_val['id_student']):
            if student_id in student_id_map:
                seq_idx = student_id_map[student_id]
                aligned_static_probs[seq_idx] = static_probs[i]
        
        # grid search for optimal weights
        weights = np.linspace(0, 1, weight_grid)
        best_score = -1
        best_weight = 0.5
        best_threshold = 0.5
        
        results = []
        
        for w in weights:
            # create weighted ensemble
            ensemble_probs = w * aligned_static_probs + (1-w) * sequential_probs
            
            # try different thresholds
            thresholds = np.linspace(0.2, 0.8, 13)
            for threshold in thresholds:
                ensemble_pred = (ensemble_probs >= threshold).astype(int)
                
                # calculate metric
                if metric == 'f1':
                    score = f1_score(y_val, ensemble_pred)
                elif metric == 'accuracy':
                    score = accuracy_score(y_val, ensemble_pred)
                elif metric == 'auc':
                    score = roc_auc_score(y_val, ensemble_probs)
                    # for auc, threshold is irrelevant
                    break
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                results.append({
                    'static_weight': w,
                    'sequential_weight': 1-w,
                    'threshold': threshold,
                    'score': score
                })
                
                # update best parameters
                if score > best_score:
                    best_score = score
                    best_weight = w
                    best_threshold = threshold
        
        # set optimized weights
        self.static_weight = best_weight
        self.sequential_weight = 1 - best_weight
        self.threshold = best_threshold
        self.optimized = True
        
        print(f"Optimized weights: static={self.static_weight:.3f}, "
              f"sequential={self.sequential_weight:.3f}, threshold={self.threshold:.3f}")
        print(f"Best {metric} score: {best_score:.4f}")
        
        # create results dataframe
        results_df = pd.DataFrame(results)
        
        # plot weight optimization results
        plt.figure(figsize=(10, 6))
        pivot_results = results_df.pivot_table(
            index='static_weight', 
            columns='threshold', 
            values='score'
        )
        sns.heatmap(pivot_results, annot=False, cmap='viridis')
        plt.title(f'Ensemble {metric} score by weight and threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Static Weight')
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def evaluate(self,
                static_features: pd.DataFrame,
                sequential_features: Dict[str, np.ndarray],
                y_true: np.ndarray,
                student_id_map: Dict[int, int]) -> Dict[str, float]:
        """Evaluates ensemble model performance."""
        
        # get ensemble predictions
        ensemble_probs = self.predict_proba(static_features, sequential_features, student_id_map)
        ensemble_pred = (ensemble_probs >= self.threshold).astype(int)
        
        # calculate metrics
        accuracy = accuracy_score(y_true, ensemble_pred)
        f1 = f1_score(y_true, ensemble_pred)
        auc = roc_auc_score(y_true, ensemble_probs)
        cm = confusion_matrix(y_true, ensemble_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc,
            'confusion_matrix': cm,
            'threshold': self.threshold,
            'static_weight': self.static_weight,
            'sequential_weight': self.sequential_weight
        }
        
        print(f"Ensemble Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        return metrics
    
    def evaluate_demographic_fairness(self,
                                     static_features: pd.DataFrame,
                                     sequential_features: Dict[str, np.ndarray],
                                     y_true: np.ndarray,
                                     demographic_col: str,
                                     student_id_map: Dict[int, int]) -> pd.DataFrame:
        """Evaluates ensemble fairness across demographic groups."""
        
        # get ensemble predictions
        ensemble_probs = self.predict_proba(static_features, sequential_features, student_id_map)
        ensemble_pred = (ensemble_probs >= self.threshold).astype(int)
        
        # extract demographic information
        demographics = static_features[['id_student', demographic_col]].drop_duplicates()
        
        # create dataframe with predictions and demographic info
        results = pd.DataFrame({
            'id_student': static_features['id_student'].unique(),
            'true': y_true,
            'pred': ensemble_pred,
            'prob': ensemble_probs
        })
        
        # merge with demographics
        results = results.merge(demographics, on='id_student')
        
        # calculate metrics by demographic group
        metrics_by_group = {}
        
        for group in results[demographic_col].unique():
            group_data = results[results[demographic_col] == group]
            
            if len(group_data) < 10:  # skip groups with too few samples
                continue
                
            metrics_by_group[group] = {
                'count': len(group_data),
                'accuracy': accuracy_score(group_data['true'], group_data['pred']),
                'f1': f1_score(group_data['true'], group_data['pred'], zero_division=0),
                'auc': roc_auc_score(group_data['true'], group_data['prob']) 
                       if len(np.unique(group_data['true'])) > 1 else np.nan,
                'positive_rate': group_data['pred'].mean()
            }
        
        # convert to dataframe
        fairness_df = pd.DataFrame.from_dict(metrics_by_group, orient='index')
        
        # calculate demographic parity difference
        if 'positive_rate' in fairness_df.columns:
            max_rate = fairness_df['positive_rate'].max()
            min_rate = fairness_df['positive_rate'].min()
            
            print(f"Demographic Parity Difference: {max_rate - min_rate:.4f}")
            print(f"Demographic Parity Ratio: {min_rate / max_rate if max_rate > 0 else 0:.4f}")
            
            # add disparate impact ratio
            fairness_df['disparate_impact_ratio'] = fairness_df['positive_rate'] / max_rate
        
        return fairness_df
    
    def save_model(self, filepath: str):
        """Saves ensemble configuration to disk."""
        
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # save ensemble configuration
        with open(filepath, 'wb') as f:
            pickle.dump({
                'static_weight': self.static_weight,
                'sequential_weight': self.sequential_weight,
                'threshold': self.threshold,
                'optimized': self.optimized
            }, f)
        
        print(f"Ensemble configuration saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EnsembleModel':
        """Loads ensemble configuration from disk."""
        
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        ensemble = cls(
            static_weight=config['static_weight'],
            sequential_weight=config['sequential_weight'],
            threshold=config['threshold']
        )
        ensemble.optimized = config['optimized']
        
        return ensemble


def combine_model_predictions(static_model, sequential_model, 
                             static_features, sequential_features,
                             student_id_map, weights=(0.5, 0.5)):
    """Utility function to combine predictions from two models."""
    
    # validate weights
    if abs(sum(weights) - 1.0) > 1e-10:
        raise ValueError("Weights must sum to 1.0")
    
    # get predictions from static model
    static_probs = static_model.predict_proba(static_features)
    
    # get predictions from sequential model
    sequential_probs = sequential_model.predict_proba(sequential_features)
    
    # align predictions by student id
    aligned_static_probs = np.zeros_like(sequential_probs)
    for i, student_id in enumerate(static_features['id_student']):
        if student_id in student_id_map:
            seq_idx = student_id_map[student_id]
            aligned_static_probs[seq_idx] = static_probs[i]
    
    # weighted combination
    ensemble_probs = weights[0] * aligned_static_probs + weights[1] * sequential_probs
    
    return ensemble_probs