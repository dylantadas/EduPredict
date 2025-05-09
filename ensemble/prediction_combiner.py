import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from config import FAIRNESS
from evaluation.performance_metrics import evaluate_fairness_metrics

class PredictionCombiner:
    """Combines predictions from multiple models with optional fairness constraints."""
    
    def __init__(
        self,
        method: str = 'weighted_average',
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        fairness_constraints: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize prediction combiner.
        
        Args:
            method: Method for combining predictions ('weighted_average' or 'stacking')
            weights: Model weights dictionary (defaults to rf:0.6, gru:0.4)
            threshold: Base classification threshold
            fairness_constraints: Dictionary containing:
                - thresholds: Dict[str, float] for group-specific classification thresholds
                - fairness metric thresholds from FAIRNESS config
            random_state: Random seed
            logger: Optional logger
        """
        self.method = method
        self.weights = weights or {'rf': 0.6, 'gru': 0.4}
        self.threshold = threshold
        
        # Initialize fairness constraints
        if fairness_constraints is None:
            self.fairness_constraints = {
                'thresholds': {},  # Group-specific classification thresholds
                **FAIRNESS['thresholds']  # Fairness metric thresholds
            }
        else:
            self.fairness_constraints = fairness_constraints
            
        self.random_state = random_state
        self.logger = logger or logging.getLogger('edupredict')
        
    def combine_predictions(
        self,
        rf_proba: np.ndarray,
        gru_proba: np.ndarray,
        temporal_context: Optional[np.ndarray] = None,
        sensitive_features: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Combine model predictions with optional temporal context.
        
        Args:
            rf_proba: Random Forest probabilities
            gru_proba: GRU probabilities
            temporal_context: Optional temporal features
            sensitive_features: Optional demographic features
            
        Returns:
            Combined probability predictions
        """
        if self.method == 'weighted_average':
            # Basic weighted average
            combined_proba = (
                self.weights['rf'] * rf_proba +
                self.weights['gru'] * gru_proba
            )
            
            # Incorporate temporal context if available
            if temporal_context is not None:
                # Normalize temporal weights
                temp_weights = temporal_context / temporal_context.sum()
                combined_proba = combined_proba * temp_weights
            
            # Apply fairness corrections if needed
            if sensitive_features is not None and self.fairness_constraints:
                # Store original predictions
                original_proba = combined_proba.copy()
                predictions = np.zeros_like(combined_proba)
                
                # Apply group-specific thresholds
                for group in sensitive_features.unique():
                    group_mask = sensitive_features == group
                    group_threshold = self.fairness_constraints.get(
                        'thresholds', {}
                    ).get(group, self.threshold)
                    
                    predictions[group_mask] = (
                        original_proba[group_mask] >= group_threshold
                    ).astype(int)
                
                return predictions
            
            return combined_proba
            
        elif self.method == 'stacking':
            raise NotImplementedError("Stacking method not implemented yet")
            
        else:
            raise ValueError(f"Unsupported combination method: {self.method}")
            
    def optimize_weights(
        self,
        rf_proba: np.ndarray,
        gru_proba: np.ndarray,
        true_labels: np.ndarray,
        temporal_context: Optional[np.ndarray] = None,
        sensitive_features: Optional[pd.Series] = None,
        metric: str = 'f1'
    ) -> Dict[str, float]:
        """Find optimal weights for combining predictions.
        
        Args:
            rf_proba: Random Forest probabilities
            gru_proba: GRU probabilities
            true_labels: True class labels
            temporal_context: Optional temporal features
            sensitive_features: Optional demographic features
            metric: Metric to optimize for
            
        Returns:
            Dictionary of optimal weights
        """
        best_score = 0.0
        best_weights = self.weights.copy()
        
        # Grid search over weights
        for rf_weight in np.arange(0.1, 1.0, 0.1):
            gru_weight = 1.0 - rf_weight
            weights = {'rf': rf_weight, 'gru': gru_weight}
            
            # Try these weights
            combined = self.combine_predictions(
                rf_proba=rf_proba,
                gru_proba=gru_proba,
                temporal_context=temporal_context,
                sensitive_features=sensitive_features
            )
            
            # Calculate performance and fairness metrics
            if sensitive_features is not None:
                # Evaluate fairness metrics
                fairness_metrics = evaluate_fairness_metrics(
                    true_labels,
                    combined,
                    sensitive_features
                )
                
                # Check if fairness constraints are satisfied
                constraints_satisfied = True
                for metric_name, threshold in self.fairness_constraints.items():
                    if metric_name != 'thresholds':  # Skip classification thresholds
                        if fairness_metrics.get(metric_name, float('inf')) > threshold:
                            constraints_satisfied = False
                            break
                
                # Only consider weights that satisfy fairness constraints
                if not constraints_satisfied:
                    continue
            
            # Calculate performance metric
            if metric == 'f1':
                score = f1_score(true_labels, (combined >= self.threshold).astype(int))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            # Update best weights if better performance achieved
            if score > best_score:
                best_score = score
                best_weights = weights
                
        self.weights = best_weights
        self.logger.info(f"Optimized weights: {best_weights}, {metric}: {best_score:.4f}")
        return best_weights
        
    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()
        
    def set_fairness_constraints(
        self,
        constraints: Dict[str, Any]
    ) -> None:
        """Set fairness constraints for prediction combination."""
        self.fairness_constraints = constraints

def combine_model_predictions(
    static_probs: np.ndarray,
    sequential_probs: np.ndarray,
    weights: Tuple[float, float] = (0.5, 0.5),
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Combines predictions from multiple models.
    
    Args:
        static_probs: Probabilities from static model
        sequential_probs: Probabilities from sequential model
        weights: Tuple of weights for each model
        logger: Logger for tracking prediction combination
        
    Returns:
        Array of combined probabilities
    """
    logger = logger or logging.getLogger('edupredict2')
    
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1")
        
    if static_probs.shape != sequential_probs.shape:
        raise ValueError("Prediction arrays must have same shape")
    
    try:
        # Apply weights and combine
        weighted_static = static_probs * weights[0]
        weighted_sequential = sequential_probs * weights[1]
        combined_probs = weighted_static + weighted_sequential
        
        logger.debug(f"Combined predictions with weights {weights}")
        return combined_probs
        
    except Exception as e:
        logger.error(f"Error combining predictions: {str(e)}")
        raise

def align_predictions_by_student(
    static_probs: np.ndarray,
    static_ids: np.ndarray,
    sequential_probs: np.ndarray,
    sequential_ids: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Aligns predictions from different paths by student ID.
    
    Args:
        static_probs: Probabilities from static model
        static_ids: Student IDs for static predictions
        sequential_probs: Probabilities from sequential model
        sequential_ids: Student IDs for sequential predictions
        logger: Logger for tracking alignment process
        
    Returns:
        Array of aligned probabilities
    """
    logger = logger or logging.getLogger('edupredict2')
    
    try:
        # Create mapping from sequential IDs to indices
        seq_id_to_idx = {id_: idx for idx, id_ in enumerate(sequential_ids)}
        
        # Initialize aligned probabilities array
        aligned_probs = np.zeros_like(static_probs)
        
        # Map static IDs to sequential probabilities
        for i, static_id in enumerate(static_ids):
            if static_id in seq_id_to_idx:
                seq_idx = seq_id_to_idx[static_id]
                aligned_probs[i] = sequential_probs[seq_idx]
            else:
                logger.warning(f"No sequential prediction found for student {static_id}")
                aligned_probs[i] = 0.5  # Default to uncertainty
        
        logger.debug(f"Aligned predictions for {len(static_ids)} students")
        return aligned_probs
        
    except Exception as e:
        logger.error(f"Error aligning predictions: {str(e)}")
        raise

def calibrate_probabilities(
    probs: np.ndarray,
    y_true: np.ndarray,
    method: str = 'isotonic',
    cv: int = 5,
    demographic_data: Optional[pd.DataFrame] = None,
    protected_attribute: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Calibrates raw probabilities to improve reliability with optional demographic-specific calibration.
    
    Args:
        probs: Raw probabilities
        y_true: True targets
        method: Calibration method ('isotonic', 'sigmoid', etc.)
        cv: Number of cross-validation folds
        demographic_data: DataFrame with demographic information
        protected_attribute: Protected attribute column for group-specific calibration
        logger: Logger for tracking calibration process
        
    Returns:
        Array of calibrated probabilities or dictionary mapping groups to calibrated probabilities
    """
    logger = logger or logging.getLogger('edupredict2')
    
    try:
        if demographic_data is not None and protected_attribute is not None:
            # Group-specific calibration
            if protected_attribute not in demographic_data.columns:
                raise ValueError(f"Protected attribute {protected_attribute} not found in demographic data")
            
            calibrated_probs = np.zeros_like(probs)
            group_calibrators = {}
            
            # Get unique groups
            groups = demographic_data[protected_attribute].unique()
            
            for group in groups:
                # Get group mask
                group_mask = demographic_data[protected_attribute] == group
                
                # Skip if no samples for group
                if not np.any(group_mask):
                    continue
                
                # Get group data
                group_probs = probs[group_mask]
                group_true = y_true[group_mask]
                
                # Calibrate for group
                if method == 'isotonic':
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                else:  # sigmoid
                    calibrator = CalibratedClassifierCV(cv=cv, method='sigmoid')
                    
                # Reshape for sklearn
                group_probs_2d = group_probs.reshape(-1, 1)
                
                # Fit calibrator
                calibrator.fit(group_probs_2d, group_true)
                
                # Apply calibration
                if method == 'isotonic':
                    calibrated_group_probs = calibrator.predict(group_probs_2d)
                else:
                    calibrated_group_probs = calibrator.predict_proba(group_probs_2d)[:, 1]
                
                # Store results
                calibrated_probs[group_mask] = calibrated_group_probs
                group_calibrators[group] = calibrator
                
                logger.debug(f"Calibrated probabilities for group {group}")
            
            return calibrated_probs
            
        else:
            # Global calibration
            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(probs.reshape(-1, 1), y_true)
                calibrated_probs = calibrator.predict(probs.reshape(-1, 1))
            else:  # sigmoid
                calibrator = CalibratedClassifierCV(cv=cv, method='sigmoid')
                calibrator.fit(probs.reshape(-1, 1), y_true)
                calibrated_probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
            
            logger.debug("Calibrated probabilities globally")
            return calibrated_probs
            
    except Exception as e:
        logger.error(f"Error calibrating probabilities: {str(e)}")
        raise

def apply_threshold(
    probs: np.ndarray,
    threshold: Union[float, Dict[str, float]] = 0.5,
    demographic_data: Optional[pd.DataFrame] = None,
    protected_attribute: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Applies threshold to probabilities with optional demographic-specific thresholds.
    
    Args:
        probs: Probabilities
        threshold: Classification threshold or dictionary mapping groups to thresholds
        demographic_data: DataFrame with demographic information
        protected_attribute: Protected attribute column for group-specific thresholds
        logger: Logger for tracking threshold application
        
    Returns:
        Array of binary predictions
    """
    logger = logger or logging.getLogger('edupredict2')
    
    try:
        if isinstance(threshold, dict) and demographic_data is not None and protected_attribute is not None:
            # Group-specific thresholding
            predictions = np.zeros_like(probs, dtype=int)
            
            for group, group_threshold in threshold.items():
                group_mask = demographic_data[protected_attribute] == group
                predictions[group_mask] = (probs[group_mask] >= group_threshold).astype(int)
                
            logger.debug(f"Applied group-specific thresholds for {protected_attribute}")
        else:
            # Global thresholding
            if isinstance(threshold, dict):
                logger.warning("Demographic data or protected attribute missing, using default threshold")
                threshold = 0.5
                
            predictions = (probs >= threshold).astype(int)
            logger.debug(f"Applied global threshold: {threshold}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error applying thresholds: {str(e)}")
        raise

def evaluate_calibration(
    probs: np.ndarray,
    y_true: np.ndarray,
    demographic_data: Optional[pd.DataFrame] = None,
    protected_attributes: Optional[List[str]] = None,
    n_bins: int = 10,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Evaluates calibration quality of probability predictions.
    
    Args:
        probs: Predicted probabilities
        y_true: True target values
        demographic_data: DataFrame with demographic information
        protected_attributes: List of protected attribute columns
        n_bins: Number of bins for calibration curve
        logger: Logger for tracking evaluation
        
    Returns:
        Dictionary with calibration metrics
    """
    logger = logger or logging.getLogger('edupredict2')
    
    try:
        results = {}
        
        # Global calibration evaluation
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins)
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        
        results['global'] = {
            'calibration_curve': {
                'true_probs': prob_true.tolist(),
                'pred_probs': prob_pred.tolist()
            },
            'calibration_error': calibration_error
        }
        
        # Group-specific calibration evaluation
        if demographic_data is not None and protected_attributes:
            group_results = {}
            
            for attr in protected_attributes:
                if attr not in demographic_data.columns:
                    logger.warning(f"Protected attribute {attr} not found in demographic data")
                    continue
                
                attr_results = {}
                groups = demographic_data[attr].unique()
                
                for group in groups:
                    group_mask = demographic_data[attr] == group
                    group_true = y_true[group_mask]
                    group_probs = probs[group_mask]
                    
                    if len(group_true) > n_bins:  # Need enough samples for reliable curve
                        g_prob_true, g_prob_pred = calibration_curve(
                            group_true, group_probs, n_bins=n_bins
                        )
                        g_calibration_error = np.mean(np.abs(g_prob_true - g_prob_pred))
                        
                        attr_results[group] = {
                            'calibration_curve': {
                                'true_probs': g_prob_true.tolist(),
                                'pred_probs': g_prob_pred.tolist()
                            },
                            'calibration_error': g_calibration_error
                        }
                    else:
                        logger.warning(f"Not enough samples for calibration curve for {attr}={group}")
                
                group_results[attr] = attr_results
            
            results['group_specific'] = group_results
        
        logger.info("Completed calibration evaluation")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating calibration: {str(e)}")
        raise

def optimize_ensemble_weights(
    rf_probs: np.ndarray,
    gru_probs: np.ndarray,
    y_true: np.ndarray,
    weight_steps: int = 21,
    threshold_steps: int = 21,
    metric: str = 'f1',
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Finds optimal ensemble weights and threshold.
    
    Args:
        rf_probs: Probabilities from Random Forest model
        gru_probs: Probabilities from GRU model
        y_true: True target values
        weight_steps: Number of weight values to try
        threshold_steps: Number of threshold values to try
        metric: Metric to optimize ('f1', 'accuracy', or 'roc_auc')
        logger: Logger for tracking optimization
        
    Returns:
        Tuple of (best parameters dict, results dataframe)
    """
    logger = logger or logging.getLogger('edupredict2')
    
    # Create grid of weights and thresholds
    rf_weights = np.linspace(0, 1, weight_steps)
    thresholds = np.linspace(0, 1, threshold_steps)
    
    results = []
    
    for rf_weight in rf_weights:
        gru_weight = 1 - rf_weight
        
        # Combine predictions
        ensemble_probs = rf_weight * rf_probs + gru_weight * gru_probs
        
        for threshold in thresholds:
            # Make binary predictions
            ensemble_preds = (ensemble_probs >= threshold).astype(int)
            
            # Calculate metric
            if metric == 'f1':
                score = f1_score(y_true, ensemble_preds)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, ensemble_preds)
            elif metric == 'roc_auc':
                score = roc_auc_score(y_true, ensemble_probs)
                # For AUC, threshold is irrelevant
                break
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            results.append({
                'rf_weight': rf_weight,
                'gru_weight': gru_weight,
                'threshold': threshold,
                metric: score
            })
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Find best parameters
    best_idx = results_df[metric].idxmax()
    best_params = {
        'rf_weight': results_df.loc[best_idx, 'rf_weight'],
        'gru_weight': results_df.loc[best_idx, 'gru_weight'],
        'threshold': results_df.loc[best_idx, 'threshold'],
        'best_score': results_df.loc[best_idx, metric]
    }
    
    logger.info(
        f"Best ensemble parameters: RF weight = {best_params['rf_weight']:.3f}, "
        f"GRU weight = {best_params['gru_weight']:.3f}, threshold = {best_params['threshold']:.3f}"
    )
    logger.info(f"Best {metric} score: {best_params['best_score']:.4f}")
    
    return best_params, results_df