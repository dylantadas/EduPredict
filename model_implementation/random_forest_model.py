import logging
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Import project-specific modules
from config import RF_PARAM_GRID, RF_DEFAULT_PARAMS, FAIRNESS, DIRS, PROTECTED_ATTRIBUTES
from evaluation.fairness_analysis import calculate_fairness_metrics, analyze_bias_patterns

class RandomForestModel:
    """
    Random Forest model for EduPredict with integrated fairness evaluation.
    """
    
    def __init__(
        self,
        n_estimators: int = 100, 
        max_depth: Optional[int] = None, 
        min_samples_split: int = 2, 
        min_samples_leaf: int = 1, 
        class_weight: Optional[str] = 'balanced', 
        random_state: int = 42, 
        n_jobs: int = -1, 
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes model with hyperparameters.
        
        Args:
            n_estimators: Number of trees in forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            class_weight: Class weighting scheme
            random_state: Random seed
            n_jobs: Number of parallel jobs
            logger: Logger for tracking model lifecycle
        """
        self.logger = logger or logging.getLogger('edupredict')
        
        # Initialize model with provided parameters
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        # Store hyperparameters for reference/metadata
        self.hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
        # Initialize metadata container
        self.metadata = {
            'model_type': 'RandomForest',
            'hyperparameters': self.hyperparams,
            'feature_importance': None,
            'training_history': {},
            'evaluation_metrics': {}
        }
        
        self.logger.info(f"Initialized Random Forest model with {n_estimators} estimators")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the model on provided features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        
        Returns:
            None
        """
        try:
            # Log training start
            self.logger.info(f"Training Random Forest model on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Store feature names for later use in feature importance
            self.feature_names = list(X.columns)
            
            # Train the model
            self.model.fit(X, y)
            
            # Update metadata
            self.metadata['training_history'] = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'class_distribution': y.value_counts().to_dict()
            }
            
            # Store feature importance
            self._update_feature_importance()
            
            self.logger.info("Random Forest model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts risk probabilities.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of positive class probabilities
        """
        try:
            # Get probability predictions
            proba = self.model.predict_proba(X)
            # Return only positive class probabilities (second column)
            return proba[:, 1]
            
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts binary risk class using threshold.
        
        Args:
            X: Feature DataFrame
            threshold: Classification threshold
        
        Returns:
            Array of binary predictions
        """
        try:
            # Get probability predictions
            proba = self.predict_proba(X)
            # Apply threshold to convert to binary predictions
            return (proba >= threshold).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error during binary prediction: {str(e)}")
            raise
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        threshold: float = 0.5, 
        fairness_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates model performance, including fairness metrics if provided.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            threshold: Classification threshold
            fairness_params: Dictionary with fairness evaluation parameters
        
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Get predictions
            y_prob = self.predict_proba(X)
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate standard performance metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'auc': roc_auc_score(y, y_prob),
                'threshold': threshold
            }
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            metrics['confusion_matrix'] = {
                'tn': int(tn), 
                'fp': int(fp), 
                'fn': int(fn), 
                'tp': int(tp)
            }
            
            # Add fairness evaluation if parameters provided
            if fairness_params:
                if 'protected_attributes' in fairness_params:
                    protected_cols = fairness_params['protected_attributes']
                    if all(col in X.columns for col in protected_cols):
                        # Extract only protected attributes
                        protected_features = X[protected_cols]
                        
                        # Calculate fairness metrics
                        fairness_metrics = calculate_fairness_metrics(
                            y.values, 
                            y_prob, 
                            protected_features,
                            threshold=threshold
                        )
                        
                        # Analyze bias patterns
                        bias_analysis = analyze_bias_patterns(
                            X, 
                            y_pred, 
                            y.values,
                            metadata={
                                'model_type': 'RandomForest',
                                'evaluation_date': pd.Timestamp.now().isoformat()
                            }
                        )
                        
                        # Add to metrics
                        metrics['fairness_metrics'] = fairness_metrics
                        metrics['bias_analysis'] = bias_analysis
                    else:
                        missing_cols = [col for col in protected_cols if col not in X.columns]
                        self.logger.warning(f"Protected attributes {missing_cols} not found in data")
            
            # Update model metadata
            self.metadata['evaluation_metrics'] = metrics
            
            self.logger.info(f"Model evaluation completed: accuracy={metrics['accuracy']:.4f}, "
                            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                            f"f1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            raise
    
    def evaluate_with_protected_attributes(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        protected_attributes: List[str], 
        threshold: float = 0.5, 
        fairness_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates model fairness across protected attributes.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            protected_attributes: List of protected attribute columns
            threshold: Classification threshold
            fairness_thresholds: Dictionary of thresholds for fairness metrics
        
        Returns:
            Dictionary with fairness evaluation results
        """
        try:
            # Use the evaluate function with fairness parameters
            fairness_params = {
                'protected_attributes': protected_attributes,
                'thresholds': fairness_thresholds or FAIRNESS['thresholds']
            }
            
            # Run evaluation with fairness analysis
            metrics = self.evaluate(X, y, threshold, fairness_params)
            
            # Extract fairness-specific metrics for readability
            fairness_results = {
                'standard_metrics': {k: v for k, v in metrics.items() 
                                    if k not in ['fairness_metrics', 'bias_analysis']},
                'fairness_metrics': metrics.get('fairness_metrics', {}),
                'bias_analysis': metrics.get('bias_analysis', {})
            }
            
            # Check for fairness violations
            violations = []
            for attr, metrics in fairness_results['fairness_metrics'].items():
                if 'threshold_violations' in metrics and metrics['threshold_violations']:
                    violations.append({
                        'attribute': attr,
                        'violations': metrics['threshold_violations']
                    })
            
            if violations:
                self.logger.warning(f"Fairness violations detected: {len(violations)} attributes affected")
                fairness_results['violations_summary'] = violations
            else:
                self.logger.info("No fairness violations detected")
            
            return fairness_results
            
        except Exception as e:
            self.logger.error(f"Error during fairness evaluation: {str(e)}")
            raise
    
    def get_feature_importance(
        self, 
        plot: bool = True, 
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieves feature importance scores.
        
        Args:
            plot: Whether to plot importances
            output_path: Path to save importance visualization
        
        Returns:
            DataFrame with feature importances
        """
        try:
            if not hasattr(self, 'model') or not hasattr(self.model, 'feature_importances_'):
                raise ValueError("Model has not been trained yet")
            
            # Create DataFrame of feature importances
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create DataFrame with feature names
            feature_names = getattr(self, 'feature_names', [f"feature_{i}" for i in range(len(importances))])
            importance_df = pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices]
            })
            
            # Plot feature importances if requested
            if plot:
                plt.figure(figsize=(12, 8))
                plt.title("Feature Importances")
                
                # Only show top 20 features for readability
                top_n = min(20, len(importance_df))
                sns.barplot(
                    x='importance',
                    y='feature',
                    data=importance_df.head(top_n),
                    palette='viridis'
                )
                plt.tight_layout()
                
                # Save plot if output path provided
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_path)
                    self.logger.info(f"Feature importance plot saved to {output_path}")
                else:
                    # Default path if none provided
                    output_dir = DIRS['viz_features']
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / 'rf_feature_importance.png'
                    plt.savefig(output_path)
                    self.logger.info(f"Feature importance plot saved to {output_path}")
                
                plt.close()
            
            # Update metadata
            self.metadata['feature_importance'] = importance_df.to_dict(orient='records')
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            raise
    
    def save_model(
        self, 
        filepath: str, 
        include_metadata: bool = True, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Saves model to disk with optional metadata.
        
        Args:
            filepath: Path to save model
            include_metadata: Whether to include metadata
            metadata: Dictionary of metadata to save
        
        Returns:
            None
        """
        try:
            # Resolve filepath
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data to save
            save_data = {
                'model': self.model,
                'hyperparams': self.hyperparams
            }
            
            # Include metadata if requested
            if include_metadata:
                # Combine instance metadata with provided metadata
                model_metadata = self.metadata.copy()
                if metadata:
                    model_metadata.update(metadata)
                
                # Add timestamp
                model_metadata['save_timestamp'] = pd.Timestamp.now().isoformat()
                
                # Add to save data
                save_data['metadata'] = model_metadata
            
            # Save the model and data
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Save metadata separately in JSON format for easier access
            if include_metadata:
                metadata_path = filepath.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2, default=str)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str, logger: Optional[logging.Logger] = None) -> 'RandomForestModel':
        """
        Loads model from disk.
        
        Args:
            filepath: Path to saved model
            logger: Logger for tracking model loading
        
        Returns:
            Loaded RandomForestModel instance
        """
        try:
            logger = logger or logging.getLogger('edupredict')
            logger.info(f"Loading model from {filepath}")
            
            # Load model data
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Get model and hyperparameters
            model_obj = data['model']
            hyperparams = data['hyperparams']
            
            # Create new instance with the hyperparameters
            instance = cls(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                class_weight=hyperparams.get('class_weight', 'balanced'),
                random_state=hyperparams.get('random_state', 42),
                n_jobs=hyperparams.get('n_jobs', -1),
                logger=logger
            )
            
            # Set the trained model
            instance.model = model_obj
            
            # Restore metadata if available
            if 'metadata' in data:
                instance.metadata = data['metadata']
                
                # Restore feature names if available in metadata
                if 'feature_importance' in data['metadata'] and data['metadata']['feature_importance']:
                    instance.feature_names = [item['feature'] for item in data['metadata']['feature_importance']]
            
            logger.info(f"Successfully loaded model with {hyperparams.get('n_estimators', 'unknown')} estimators")
            
            return instance
            
        except Exception as e:
            if logger:
                logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _update_feature_importance(self) -> None:
        """
        Updates feature importance in the model metadata.
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Create DataFrame with feature names
                feature_names = getattr(self, 'feature_names', [f"feature_{i}" for i in range(len(importances))])
                importance_dict = [{
                    'feature': feature_names[i],
                    'importance': float(importances[i])
                } for i in indices]
                
                self.metadata['feature_importance'] = importance_dict
        except Exception as e:
            self.logger.warning(f"Could not update feature importance: {str(e)}")


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = 'f1',
    n_jobs: int = -1,
    random_search: bool = True,
    n_iter: int = 20,
    verbose: int = 1,
    fairness_aware: bool = False,
    protected_attributes: Optional[List[str]] = None,
    fairness_thresholds: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict, Any, pd.DataFrame]:
    """
    Tunes random forest hyperparameters using grid or random search with optional fairness awareness.
    
    Args:
        X_train: Training features
        y_train: Training targets
        param_grid: Parameter grid for tuning
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        random_search: Whether to use random search
        n_iter: Number of random iterations
        verbose: Verbosity level
        fairness_aware: Whether to include fairness in optimization
        protected_attributes: List of protected attribute columns
        fairness_thresholds: Dictionary of thresholds for fairness metrics
        logger: Logger for tracking tuning process
    
    Returns:
        Tuple of best parameters, best model, and results DataFrame
    """
    try:
        logger = logger or logging.getLogger('edupredict')
        logger.info(f"Starting Random Forest hyperparameter tuning with {'random' if random_search else 'grid'} search")
        
        # Use default param grid if none provided
        param_grid = param_grid or RF_PARAM_GRID
        
        # Create base model
        rf = RandomForestClassifier(class_weight='balanced', random_state=RF_DEFAULT_PARAMS['random_state'])
        
        # Choose search method
        if random_search:
            search = RandomizedSearchCV(
                rf,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                refit=True,
                random_state=RF_DEFAULT_PARAMS['random_state']
            )
        else:
            search = GridSearchCV(
                rf,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                refit=True
            )
        
        # Perform hyperparameter search
        logger.info(f"Searching over {len(param_grid)} parameter combinations")
        search.fit(X_train, y_train)
        
        # Log results
        logger.info(f"Best parameters found: {search.best_params_}")
        logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(search.cv_results_)
        
        # If fairness-aware tuning is requested, perform extra evaluation
        if fairness_aware and protected_attributes:
            # Check if all protected attributes are present in the data
            missing_attrs = [attr for attr in protected_attributes if attr not in X_train.columns]
            if missing_attrs:
                logger.warning(f"Protected attributes {missing_attrs} not found in data. Skipping fairness tuning.")
            else:
                logger.info("Performing fairness-aware tuning")
                
                # Create RandomForestModel with best parameters for fairness evaluation
                best_model = RandomForestModel(
                    n_estimators=search.best_params_.get('n_estimators', 100),
                    max_depth=search.best_params_.get('max_depth', None),
                    min_samples_split=search.best_params_.get('min_samples_split', 2),
                    min_samples_leaf=search.best_params_.get('min_samples_leaf', 1),
                    class_weight='balanced',
                    random_state=RF_DEFAULT_PARAMS['random_state'],
                    n_jobs=n_jobs,
                    logger=logger
                )
                
                # Train the model
                best_model.fit(X_train, y_train)
                
                # Evaluate fairness
                fairness_results = best_model.evaluate_with_protected_attributes(
                    X_train, 
                    y_train,
                    protected_attributes,
                    fairness_thresholds=fairness_thresholds
                )
                
                # Save fairness-aware tuning results
                fairness_path = DIRS['reports_fairness'] / 'tuning_fairness_results.json'
                fairness_path.parent.mkdir(parents=True, exist_ok=True)
                with open(fairness_path, 'w') as f:
                    json.dump({
                        'best_params': search.best_params_,
                        'fairness_evaluation': fairness_results
                    }, f, indent=2, default=str)
                
                logger.info(f"Fairness-aware tuning results saved to {fairness_path}")
        
        return search.best_params_, search.best_estimator_, results_df
        
    except Exception as e:
        if logger:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
        raise


def find_optimal_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = 'f1',
    fairness_aware: bool = False,
    protected_attributes: Optional[List[str]] = None,
    fairness_thresholds: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Finds optimal threshold for binary classification with optional demographic-specific thresholds.
    
    Args:
        model: Trained model with predict_proba method
        X_val: Validation features
        y_val: Validation targets
        metric: Metric to optimize
        fairness_aware: Whether to optimize for fairness
        protected_attributes: List of protected attribute columns
        fairness_thresholds: Dictionary of thresholds for fairness metrics
        logger: Logger for tracking threshold optimization
    
    Returns:
        Dictionary with optimal threshold or demographic-specific thresholds
    """
    try:
        logger = logger or logging.getLogger('edupredict')
        logger.info(f"Finding optimal threshold optimizing {metric}")
        
        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)
            # Check if we need the second column (positive class proba)
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
        else:
            raise ValueError("Model does not support probability predictions")
        
        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, 81)  # Test from 0.1 to 0.9 in 0.01 increments
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            if metric == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred)
            elif metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'balanced_accuracy':
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (tpr + tnr) / 2
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            results.append({
                'threshold': threshold,
                'score': score
            })
        
        # Find best threshold
        results_df = pd.DataFrame(results)
        best_idx = results_df['score'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_score = results_df.loc[best_idx, 'score']
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} with {metric} score: {best_score:.4f}")
        
        result = {
            'threshold': float(best_threshold),
            f'{metric}_score': float(best_score)
        }
        
        # If fairness-aware, find optimal thresholds for each demographic group
        if fairness_aware and protected_attributes:
            # Check if all protected attributes are present
            missing_attrs = [attr for attr in protected_attributes if attr not in X_val.columns]
            if missing_attrs:
                logger.warning(f"Protected attributes {missing_attrs} not found in data. Skipping fairness-aware thresholds.")
            else:
                logger.info("Finding demographic-specific thresholds")
                
                demographic_thresholds = {}
                
                # Process each protected attribute
                for attr in protected_attributes:
                    group_thresholds = {}
                    
                    # Get unique groups
                    groups = X_val[attr].unique()
                    
                    # Find optimal threshold for each group
                    for group in groups:
                        group_mask = X_val[attr] == group
                        
                        # Skip if not enough samples
                        min_samples = FAIRNESS.get('min_group_size', 50)
                        if group_mask.sum() < min_samples:
                            logger.warning(f"Group {group} in {attr} has fewer than {min_samples} samples. Skipping.")
                            continue
                        
                        # Extract group-specific data
                        group_y_val = y_val[group_mask]
                        group_y_prob = y_prob[group_mask]
                        
                        # Try different thresholds for this group
                        group_results = []
                        for threshold in thresholds:
                            group_y_pred = (group_y_prob >= threshold).astype(int)
                            
                            # Calculate metric
                            if metric == 'accuracy':
                                score = accuracy_score(group_y_val, group_y_pred)
                            elif metric == 'precision':
                                score = precision_score(group_y_val, group_y_pred)
                            elif metric == 'recall':
                                score = recall_score(group_y_val, group_y_pred)
                            elif metric == 'f1':
                                score = f1_score(group_y_val, group_y_pred)
                            elif metric == 'balanced_accuracy':
                                tn, fp, fn, tp = confusion_matrix(group_y_val, group_y_pred).ravel()
                                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                                score = (tpr + tnr) / 2
                            
                            group_results.append({
                                'threshold': threshold,
                                'score': score
                            })
                        
                        # Find best threshold for this group
                        group_results_df = pd.DataFrame(group_results)
                        group_best_idx = group_results_df['score'].idxmax()
                        group_best_threshold = group_results_df.loc[group_best_idx, 'threshold']
                        group_best_score = group_results_df.loc[group_best_idx, 'score']
                        
                        logger.info(f"Optimal threshold for {attr}={group}: {group_best_threshold:.4f} "
                                   f"with {metric} score: {group_best_score:.4f}")
                        
                        group_thresholds[str(group)] = {
                            'threshold': float(group_best_threshold),
                            f'{metric}_score': float(group_best_score)
                        }
                    
                    demographic_thresholds[attr] = group_thresholds
                
                # Add demographic thresholds to result
                result['demographic_thresholds'] = demographic_thresholds
        
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"Error finding optimal threshold: {str(e)}")
        raise