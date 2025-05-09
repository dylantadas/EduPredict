import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from config import FEATURE_ENGINEERING, FAIRNESS, DIRS, VERSION_CONTROL, RANDOM_SEED
from dataclasses import dataclass
from datetime import datetime
from utils.monitoring_utils import monitor_memory_usage, track_progress

logger = logging.getLogger('edupredict')

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8, np.uint16,
            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

@dataclass
class FeatureMetadata:
    name: str
    description: str
    feature_type: str  # demographic, sequential, temporal
    data_type: str
    statistics: Dict
    creation_timestamp: str
    parameters_used: Dict
    validation_checks: Dict
    demographic_impact: Optional[Dict] = None
    importance_score: Optional[float] = None

class FeatureSelector:
    """
    Feature selection with multiple methods and fairness considerations.
    """
    
    def __init__(
        self,
        method: str = 'importance',
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        random_state: int = 42
    ):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('importance', 'correlation', 'mutual_info', or 'f_score')
            n_features: Number of features to select
            threshold: Threshold for feature importance/correlation
            random_state: Random state for reproducibility
        """
        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.random_state = random_state
        
        self.feature_importances_: Optional[Dict[str, float]] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
        
    @monitor_memory_usage
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        protected_attributes: Optional[List[str]] = None
    ) -> 'FeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            protected_attributes: List of protected attribute columns
        """
        try:
            if protected_attributes is None:
                protected_attributes = []
                
            # Initialize feature scores dictionary
            self.feature_scores_ = {}
            
            if self.method == 'importance':
                # Use Random Forest for feature importance
                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                rf.fit(X, y)
                
                # Store feature importances
                self.feature_importances_ = dict(zip(X.columns, rf.feature_importances_))
                self.feature_scores_ = self.feature_importances_
                
            elif self.method == 'correlation':
                # Calculate correlation with target for numeric features
                correlations = {}
                numeric_features = X.select_dtypes(include=[np.number]).columns
                
                for col in numeric_features:
                    corr = abs(X[col].corr(y))
                    if not pd.isna(corr):
                        correlations[col] = corr
                        
                self.feature_scores_ = correlations
                
            elif self.method == 'mutual_info':
                # Calculate mutual information scores
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
                self.feature_scores_ = dict(zip(X.columns, mi_scores))
                
            elif self.method == 'f_score':
                # Calculate F-scores
                f_scores, _ = f_classif(X, y)
                self.feature_scores_ = dict(zip(X.columns, f_scores))
                
            else:
                raise ValueError(f"Unknown selection method: {self.method}")
                
            # Select features based on scores
            scores_series = pd.Series(self.feature_scores_)
            
            if self.n_features is not None:
                # Select top N features
                selected = scores_series.nlargest(self.n_features).index.tolist()
            elif self.threshold is not None:
                # Select features above threshold
                selected = scores_series[scores_series >= self.threshold].index.tolist()
            else:
                # Use default threshold of mean score
                mean_score = scores_series.mean()
                selected = scores_series[scores_series >= mean_score].index.tolist()
                
            # Always include protected attributes
            self.selected_features_ = list(set(selected + protected_attributes))
            
            # Log selection results
            logger.info(f"Selected {len(self.selected_features_)} features")
            
            return self
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            raise
            
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to include only selected features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")
            
        return X[self.selected_features_]
        
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        protected_attributes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit the selector and transform the data in one step.
        """
        return self.fit(X, y, protected_attributes).transform(X)
        
    def get_feature_scores(self) -> pd.DataFrame:
        """
        Get feature scores with detailed statistics.
        """
        if self.feature_scores_ is None:
            raise ValueError("Selector has not been fitted yet")
            
        scores_df = pd.DataFrame({
            'feature': list(self.feature_scores_.keys()),
            'score': list(self.feature_scores_.values())
        })
        
        scores_df['selected'] = scores_df['feature'].isin(self.selected_features_)
        scores_df = scores_df.sort_values('score', ascending=False)
        
        return scores_df
        
    def export_feature_metadata(self, output_dir: Union[str, Path]) -> None:
        """
        Export feature selection metadata and scores.
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                'method': self.method,
                'n_features': self.n_features,
                'threshold': self.threshold,
                'selected_features': self.selected_features_,
                'feature_scores': self.feature_scores_,
                'timestamp': datetime.now().isoformat()
            }
            
            output_path = output_dir / f'feature_selection_{self.method}.json'
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
                
            logger.info(f"Exported feature selection metadata to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting feature metadata: {str(e)}")
            raise

def analyze_feature_correlations(
    features: pd.DataFrame,
    threshold: float = 0.8
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    Analyze correlations between features to identify redundancy.
    
    Args:
        features: Feature DataFrame
        threshold: Correlation threshold for identifying high correlations
        
    Returns:
        Correlation matrix and list of highly correlated feature pairs
    """
    try:
        # Calculate correlation matrix
        corr_matrix = features.corr()
        
        # Find highly correlated feature pairs
        high_correlations = []
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find feature pairs with correlation above threshold
        pairs = np.where(np.abs(upper) > threshold)
        for i, j in zip(*pairs):
            feat1 = corr_matrix.index[i]
            feat2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            high_correlations.append((feat1, feat2, corr))
            
        if high_correlations:
            logger.warning(
                f"Found {len(high_correlations)} highly correlated feature pairs "
                f"(correlation > {threshold})"
            )
            
        return corr_matrix, high_correlations
        
    except Exception as e:
        logger.error(f"Error analyzing feature correlations: {str(e)}")
        raise

def analyze_feature_importance(
    features: pd.DataFrame,  
    target_col: str,
    categorical_cols: Optional[List[str]] = None,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Analyze feature importance using random forest classifier.
    
    Args:
        features: DataFrame containing features
        target_col: Target column name
        categorical_cols: List of categorical columns to encode
        random_state: Random state for reproducibility
        logger: Logger instance
        
    Returns:
        Dictionary containing feature importance analysis results
    """
    if logger is None:
        logger = logging.getLogger('edupredict')
        
    try:
        # Prepare features
        X = features.drop(columns=[target_col])
        y = features[target_col]
        
        # Handle categorical columns
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols)
            
        # Train random forest for importance analysis
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importance scores
        importance_scores = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate statistics
        total_importance = importance_scores['importance'].sum()
        cumulative_importance = importance_scores['importance'].cumsum() / total_importance
        
        results = {
            'feature_importance': importance_scores.to_dict('records'),
            'statistics': {
                'n_features': len(X.columns),
                'mean_importance': float(importance_scores['importance'].mean()),
                'std_importance': float(importance_scores['importance'].std()),
                'top_10_importance_sum': float(importance_scores['importance'].head(10).sum())
            },
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(X),
                'random_state': random_state
            }
        }
        
        logger.info(f"Completed feature importance analysis for {len(X.columns)} features")
        return results
        
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {str(e)}")
        raise