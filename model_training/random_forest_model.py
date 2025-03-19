import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class RandomForestModel:
    """Random forest model for static feature path."""
    
    def __init__(self, 
                n_estimators: int = 100, 
                max_depth: Optional[int] = None,
                min_samples_split: int = 2,
                min_samples_leaf: int = 1,
                class_weight: Optional[str] = 'balanced',
                random_state: int = 42,
                n_jobs: int = -1):
        """Initializes model with hyperparameters."""
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.feature_names = None
        self.trained = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Trains the model on provided features."""
        
        # store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # train the model
        self.model.fit(X, y)
        self.trained = True
        
        print(f"Model trained on {X.shape[0]} samples with {X.shape[1]} features")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts risk probabilities."""
        
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)[:, 1]  # return only positive class probability
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary risk class using threshold."""
        
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict:
        """Evaluates model performance."""
        
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # calculate performance metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        cm = confusion_matrix(y, y_pred)
        
        # calculate fairness metrics
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc,
            'confusion_matrix': cm,
            'threshold': threshold
        }
        
        print(f"Model Performance at threshold {threshold}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        return metrics
    
    def get_feature_importance(self, plot: bool = True) -> pd.DataFrame:
        """Retrieves feature importance scores."""
        
        if not self.trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # get feature importance
        importance = self.model.feature_importances_
        
        # create dataframe with feature importance
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # plot feature importance
        if plot:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.show()
        
        return importance_df
    
    def evaluate_demographic_fairness(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series, 
                                    demographic_col: str,
                                    threshold: float = 0.5) -> pd.DataFrame:
        """Evaluates model fairness across demographic groups."""
        
        if demographic_col not in X.columns:
            raise ValueError(f"Demographic column '{demographic_col}' not found in features")
        
        # get predictions
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # create dataframe with predictions and demographic info
        results = pd.DataFrame({
            'true': y,
            'pred': y_pred,
            'prob': y_pred_proba,
            'demographic': X[demographic_col]
        })
        
        # calculate metrics by demographic group
        metrics_by_group = {}
        
        for group in results['demographic'].unique():
            group_data = results[results['demographic'] == group]
            
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
        
        # calculate disparate impact
        if 'positive_rate' in fairness_df.columns:
            max_rate = fairness_df['positive_rate'].max()
            min_rate = fairness_df['positive_rate'].min()
            
            if max_rate > 0:
                fairness_df['disparate_impact_ratio'] = min_rate / max_rate
            
        return fairness_df
    
    def save_model(self, filepath: str) -> None:
        """Saves model to disk."""
        
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RandomForestModel':
        """Loads model from disk."""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        rf_model = cls()
        rf_model.model = model_data['model']
        rf_model.feature_names = model_data['feature_names']
        rf_model.trained = True
        
        return rf_model


def find_optimal_threshold(model, X_val, y_val, metric='f1'):
    """Finds optimal threshold for binary classification."""
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = []
    
    for threshold in thresholds:
        y_pred = (model.predict_proba(X_val) >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_val, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_val, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        scores.append((threshold, score))
    
    # find threshold with max score
    best_threshold, best_score = max(scores, key=lambda x: x[1])
    
    print(f"Best threshold: {best_threshold} with {metric} score: {best_score:.4f}")
    return best_threshold