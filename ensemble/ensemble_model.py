import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import pickle
import logging
from pathlib import Path

from evaluation.performance_metrics import calculate_model_metrics
from model_implementation.random_forest_model import RandomForestModel
from model_implementation.gru_model import GRUModel
from utils.logging_utils import setup_logger
from sklearn.metrics import f1_score
from config import FAIRNESS

logger = logging.getLogger('edupredict')

class EnsembleModel:
    """
    Ensemble model that combines predictions from Random Forest and GRU models.
    """
    
    def __init__(
        self,
        rf_model=None,
        gru_model=None,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        fairness_constraints: Optional[Dict[str, Any]] = None
    ):
        """Initialize ensemble model.
        
        Args:
            rf_model: Random Forest model
            gru_model: GRU model
            weights: Dictionary of weights for each model (defaults to rf:0.6, gru:0.4)
            threshold: Base classification threshold
            fairness_constraints: Optional fairness constraints for demographic groups
                Format: {
                    'thresholds': Dict[str, float],  # Classification thresholds per group
                    'demographic_parity_difference': float,
                    'disparate_impact_ratio': float,
                    'equal_opportunity_difference': float,
                    'average_odds_difference': float
                }
        """
        self.rf_model = rf_model
        self.gru_model = gru_model
        self.weights = weights or {'rf': 0.5, 'gru': 0.5}
        self.threshold = threshold
        
        # Initialize fairness constraints from config if not provided
        if fairness_constraints is None:
            self.fairness_constraints = {
                'thresholds': {},  # Will be optimized per group if needed
                **FAIRNESS['thresholds']  # Add default fairness metric thresholds
            }
        else:
            self.fairness_constraints = fairness_constraints

    def predict_proba(
        self,
        static_features: pd.DataFrame,
        sequential_features: Dict[str, Any],
        temporal_context: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Generate probability predictions using both models.
        
        Args:
            static_features: Static features for RF model
            sequential_features: Sequential features for GRU model
            temporal_context: Optional temporal context features
            
        Returns:
            Array of probability predictions
        """
        # Get RF predictions
        rf_proba = self.rf_model.predict_proba(static_features)

        # Get GRU predictions
        gru_proba = self.gru_model.predict_proba(
            sequential_features['sequence_data'],
            sequence_lengths=sequential_features.get('sequence_lengths')
        )

        # Combine predictions using weights
        ensemble_proba = (
            self.weights['rf'] * rf_proba[:, 1] + 
            self.weights['gru'] * gru_proba[:, 1]
        )

        return np.column_stack((1 - ensemble_proba, ensemble_proba))

    def predict(
        self,
        static_features: pd.DataFrame,
        sequential_features: Dict[str, Any],
        temporal_context: Optional[pd.DataFrame] = None,
        threshold: float = 0.5,
        demographic_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Generate class predictions using both models.
        
        Args:
            static_features: Static features for RF model
            sequential_features: Sequential features for GRU model
            temporal_context: Optional temporal context features
            threshold: Classification threshold
            demographic_features: Optional demographic features for fairness
            
        Returns:
            Array of class predictions
        """
        proba = self.predict_proba(
            static_features,
            sequential_features,
            temporal_context
        )[:, 1]

        if demographic_features is not None and self.fairness_constraints:
            # Apply demographic-specific thresholds
            predictions = np.zeros_like(proba)
            for group in demographic_features['gender'].unique():
                group_mask = demographic_features['gender'] == group
                group_threshold = self.fairness_constraints.get(
                    'thresholds', {}
                ).get(group, threshold)
                predictions[group_mask] = (
                    proba[group_mask] >= group_threshold
                ).astype(int)
            return predictions
        
        return (proba >= threshold).astype(int)

    def fit(
        self,
        static_features: pd.DataFrame,
        sequential_features: Dict[str, Any],
        targets: np.ndarray,
        demographic_features: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            static_features: Static features for RF model
            sequential_features: Sequential features for GRU model
            targets: True labels
            demographic_features: Optional demographic features
        """
        # Get predictions from both models
        rf_proba = self.rf_model.predict_proba(static_features)[:, 1]
        gru_proba = self.gru_model.predict_proba(
            sequential_features['sequence_data'],
            sequence_lengths=sequential_features.get('sequence_lengths')
        )[:, 1]

        # Grid search for optimal weights
        best_score = 0
        best_weights = self.weights

        for rf_weight in np.arange(0.1, 1.0, 0.1):
            gru_weight = 1 - rf_weight
            ensemble_proba = rf_weight * rf_proba + gru_weight * gru_proba
            
            # Calculate metric (e.g., F1 score)
            score = f1_score(targets, ensemble_proba >= 0.5)
            
            if score > best_score:
                best_score = score
                best_weights = {'rf': rf_weight, 'gru': gru_weight}

        self.weights = best_weights

    @property
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance from RF model."""
        if hasattr(self.rf_model, 'feature_importance'):
            return self.rf_model.feature_importance
        return {}

    def evaluate(
        self,
        static_features: pd.DataFrame,
        sequential_features: Dict[str, Any],
        y_true: np.ndarray,
        student_id_map: Optional[Dict[int, int]] = None
    ) -> Dict[str, float]:
        """Evaluate ensemble model performance."""
        predictions = self.predict(static_features, sequential_features)
        probabilities = self.predict_proba(static_features, sequential_features)
        
        return calculate_model_metrics(y_true, predictions, probabilities)
        
    def save(self, path: str) -> None:
        """Save ensemble model configuration."""
        model_data = {
            'weights': self.weights,
            'fairness_constraints': self.fairness_constraints,
            'rf_model_path': 'rf_model.pkl',
            'gru_model_path': 'gru_model.h5'
        }
        
        save_dir = Path(path).parent
        
        # Save component models
        if self.rf_model:
            self.rf_model.save(str(save_dir / model_data['rf_model_path']))
        if self.gru_model:
            self.gru_model.save(str(save_dir / model_data['gru_model_path']))
            
        # Save ensemble configuration
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Saved ensemble model to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """Load ensemble model from saved configuration."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        ensemble = cls(
            weights=model_data['weights'],
            fairness_constraints=model_data['fairness_constraints']
        )
        
        save_dir = Path(path).parent
        
        # Load component models
        rf_path = save_dir / model_data['rf_model_path']
        gru_path = save_dir / model_data['gru_model_path']
        
        if rf_path.exists():
            ensemble.rf_model = RandomForestModel.load(str(rf_path))
        if gru_path.exists():
            ensemble.gru_model = GRUModel.load(str(gru_path))
            
        logger.info(f"Loaded ensemble model from {path}")
        return ensemble