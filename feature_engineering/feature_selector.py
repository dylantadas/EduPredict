import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from config import FEATURE_ENGINEERING, FAIRNESS, DIRS, VERSION_CONTROL
from dataclasses import dataclass

logger = logging.getLogger('edupredict')

@dataclass
class FeatureMetadata:
    name: str
    description: str
    feature_type: str  # demographic, sequential, temporal
    data_type: str
    statistics: Dict
    demographic_impact: Optional[Dict] = None
    importance_score: Optional[float] = None
    creation_timestamp: str
    parameters_used: Dict
    validation_checks: Dict

class FeatureSelector:
    def __init__(self, config: Dict):
        self.config = config
        self.metadata_store = {}
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(DIRS['features'])  # Use configured directory
        self.metadata_dir = Path(DIRS['intermediate']) / 'features' / 'metadata'
        
    def track_feature_metadata(self, 
                             feature_df: pd.DataFrame,
                             feature_type: str,
                             parameters: Dict) -> None:
        """Track metadata for generated features."""
        for column in feature_df.columns:
            stats = feature_df[column].describe().to_dict()
            
            metadata = FeatureMetadata(
                name=column,
                description=self._generate_feature_description(column, feature_type),
                feature_type=feature_type,
                data_type=str(feature_df[column].dtype),
                statistics=stats,
                parameters_used=parameters,
                creation_timestamp=pd.Timestamp.now().isoformat(),
                validation_checks=self._validate_feature(feature_df[column])
            )
            
            self.metadata_store[column] = metadata
            
    def _validate_feature(self, feature_series: pd.Series) -> Dict:
        """Perform validation checks on feature."""
        return {
            "missing_percentage": (feature_series.isna().sum() / len(feature_series)) * 100,
            "unique_values": feature_series.nunique(),
            "is_constant": feature_series.nunique() == 1,
            "has_infinity": np.isinf(feature_series).any() if pd.api.types.is_numeric_dtype(feature_series) else False
        }
    
    def export_metadata(self) -> None:
        """Export feature metadata to JSON file."""
        try:
            # Ensure directories exist
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_dict = {
                name: vars(metadata) 
                for name, metadata in self.metadata_store.items()
            }
            
            output_path = self.metadata_dir / "feature_metadata.json"
            with open(output_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            self.logger.info(f"Exported feature metadata to {output_path}")
            
            # Version control if enabled
            if VERSION_CONTROL.get('enable_data_versioning', False):
                self._version_metadata(metadata_dict)
                
        except Exception as e:
            self.logger.error(f"Error exporting metadata: {str(e)}")
            raise
            
    def _version_metadata(self, metadata: Dict) -> None:
        """Version control for feature metadata."""
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            version_path = self.metadata_dir / f"feature_metadata_{timestamp}.json"
            
            with open(version_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"Saved versioned metadata to {version_path}")
            
        except Exception as e:
            self.logger.error(f"Error versioning metadata: {str(e)}")
            raise

    def _generate_feature_description(self, 
                                    feature_name: str, 
                                    feature_type: str) -> str:
        """Generate human-readable feature description."""
        # Add logic to generate descriptions based on feature naming conventions
        pass

class FeatureMetadataLegacy:
    def __init__(self, feature_type: str):
        self.metadata = {
            'feature_type': feature_type,
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'features': {},
            'demographic_impact': {},
            'performance_metrics': {},
            'warnings': []
        }
        
    def add_feature(self, 
                   name: str, 
                   description: str, 
                   data_type: str,
                   statistics: Dict[str, float],
                   demographic_impact: Optional[Dict] = None):
        """Add metadata for a single feature."""
        self.metadata['features'][name] = {
            'description': description,
            'data_type': data_type,
            'statistics': statistics
        }
        if demographic_impact:
            self.metadata['demographic_impact'][name] = demographic_impact
            
    def add_performance_metric(self, 
                             metric_name: str, 
                             value: float):
        """Add performance metrics for feature set."""
        self.metadata['performance_metrics'][metric_name] = value
        
    def add_warning(self, warning_message: str):
        """Add warning message to metadata."""
        self.metadata['warnings'].append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'message': warning_message
        })
        
    def save_metadata(self, feature_type: str):
        """Save metadata to JSON file."""
        output_dir = Path(DIRS['intermediate']) / 'features' / 'metadata'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f'{feature_type}_metadata.json'
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Saved feature metadata to: {output_path}")

def create_feature_statistics(feature_series: pd.Series) -> Dict[str, float]:
    """
    Calculate standard statistics for a feature.
    
    Args:
        feature_series: Series containing feature values
    
    Returns:
        Dictionary with calculated statistics
    """
    return {
        'mean': float(feature_series.mean()),
        'std': float(feature_series.std()),
        'min': float(feature_series.min()),
        'max': float(feature_series.max()),
        'missing_ratio': float(feature_series.isnull().mean()),
        'unique_count': int(feature_series.nunique())
    }

def select_features_by_importance(
    features: pd.DataFrame,
    model: Any,
    threshold: float = FEATURE_ENGINEERING['importance_threshold']
) -> pd.DataFrame:
    """
    Selects features based on importance scores.
    
    Args:
        features: DataFrame containing features
        model: Trained model with feature importance attribute
        threshold: Minimum importance score to retain feature
        
    Returns:
        DataFrame with selected features
    """
    try:
        # Get numeric features
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        
        # Remove identifier columns and protected attributes
        exclude_cols = (
            ['id_student', 'code_module', 'code_presentation'] +
            FAIRNESS['protected_attributes']
        )
        feature_cols = [
            col for col in numeric_features.columns 
            if col not in exclude_cols
        ]
        
        if not feature_cols:
            logger.warning("No numeric features available for importance calculation")
            return features
            
        # Get feature importance scores
        importance_scores = getattr(model, 'feature_importances_', None)
        if importance_scores is None:
            logger.warning(
                "Model does not have feature_importances_ attribute. "
                "Using coefficients if available."
            )
            importance_scores = abs(getattr(model, 'coef_', None))
            
        if importance_scores is None:
            logger.error(
                "Model does not provide feature importance scores"
            )
            return features
            
        # Create importance dictionary
        importance_dict = dict(zip(feature_cols, importance_scores))
        
        # Select features above threshold
        selected_features = [
            col for col, score in importance_dict.items()
            if score >= threshold
        ]
        
        # Always include identifier columns and protected attributes
        selected_features.extend(exclude_cols)
        
        logger.info(
            f"Selected {len(selected_features)} features "
            f"(threshold: {threshold})"
        )
        
        return features[selected_features]
        
    except Exception as e:
        logger.error(f"Error selecting features by importance: {str(e)}")
        raise

def remove_correlated_features(
    features: pd.DataFrame,
    threshold: float = FEATURE_ENGINEERING['correlation_threshold']
) -> pd.DataFrame:
    """
    Removes highly correlated features.
    
    Args:
        features: DataFrame containing features
        threshold: Correlation threshold for feature removal
        
    Returns:
        DataFrame with features after removing highly correlated ones
    """
    try:
        # Get numeric features
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        
        # Remove identifier columns and protected attributes
        exclude_cols = (
            ['id_student', 'code_module', 'code_presentation'] +
            FAIRNESS['protected_attributes']
        )
        feature_cols = [
            col for col in numeric_features.columns 
            if col not in exclude_cols
        ]
        
        if len(feature_cols) < 2:
            logger.warning("Not enough numeric features for correlation analysis")
            return features
            
        # Calculate correlation matrix
        corr_matrix = numeric_features[feature_cols].corr()
        
        # Find highly correlated pairs
        high_corr_features = set()
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    feat1, feat2 = feature_cols[i], feature_cols[j]
                    # Keep the feature with higher variance
                    if features[feat1].std() < features[feat2].std():
                        high_corr_features.add(feat1)
                    else:
                        high_corr_features.add(feat2)
        
        # Remove highly correlated features
        features_to_keep = [
            col for col in features.columns 
            if col not in high_corr_features
        ]
        
        logger.info(
            f"Removed {len(high_corr_features)} highly correlated features "
            f"(threshold: {threshold})"
        )
        
        return features[features_to_keep]
        
    except Exception as e:
        logger.error(f"Error removing correlated features: {str(e)}")
        raise

def calculate_feature_statistics(features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates statistics for feature distributions.
    
    Args:
        features: DataFrame containing features
        
    Returns:
        DataFrame with feature statistics
    """
    try:
        # Get numeric features
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        
        # Remove identifier columns
        exclude_cols = ['id_student', 'code_module', 'code_presentation']
        feature_cols = [
            col for col in numeric_features.columns 
            if col not in exclude_cols
        ]
        
        if not feature_cols:
            logger.warning("No numeric features available for statistics")
            return pd.DataFrame()
        
        # Calculate statistics
        stats = pd.DataFrame({
            'feature': feature_cols,
            'mean': [features[col].mean() for col in feature_cols],
            'std': [features[col].std() for col in feature_cols],
            'min': [features[col].min() for col in feature_cols],
            'max': [features[col].max() for col in feature_cols],
            'missing': [features[col].isnull().sum() for col in feature_cols],
            'unique': [features[col].nunique() for col in feature_cols]
        })
        
        # Calculate skewness and kurtosis
        stats['skewness'] = [features[col].skew() for col in feature_cols]
        stats['kurtosis'] = [features[col].kurtosis() for col in feature_cols]
        
        # Add correlation with protected attributes if available
        for protected_attr in FAIRNESS['protected_attributes']:
            if protected_attr in features.columns:
                # Calculate correlation if attribute is numeric or encoded
                attr_col = f"{protected_attr}_encoded" if protected_attr in features.columns else protected_attr
                if attr_col in features.columns and pd.api.types.is_numeric_dtype(features[attr_col]):
                    stats[f'corr_{protected_attr}'] = [
                        spearmanr(features[col], features[attr_col])[0]
                        if not (features[col].nunique() <= 1 or features[attr_col].nunique() <= 1)
                        else np.nan
                        for col in feature_cols
                    ]
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating feature statistics: {str(e)}")
        raise

def analyze_demographic_impact(
    features: pd.DataFrame,
    protected_attributes: List[str] = FAIRNESS['protected_attributes']
) -> Dict[str, pd.DataFrame]:
    """
    Analyzes feature distributions across demographic groups.
    
    Args:
        features: DataFrame containing features
        protected_attributes: List of protected attributes to analyze
    
    Returns:
        Dictionary with demographic impact statistics
    """
    try:
        impact_stats = {}
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        
        for attr in protected_attributes:
            if attr not in features.columns:
                logger.warning(f"Protected attribute {attr} not found in features")
                continue
                
            # Calculate statistics per group
            group_stats = []
            for feature_col in numeric_features.columns:
                if feature_col == attr:
                    continue
                    
                group_means = features.groupby(attr)[feature_col].mean()
                group_stds = features.groupby(attr)[feature_col].std()
                
                # Calculate disparity metrics
                max_mean = group_means.max()
                min_mean = group_means.min()
                mean_disparity = (max_mean - min_mean) / max_mean if max_mean != 0 else 0
                
                group_stats.append({
                    'feature': feature_col,
                    'max_group_mean': max_mean,
                    'min_group_mean': min_mean,
                    'mean_disparity': mean_disparity,
                    'group_means': group_means.to_dict(),
                    'group_stds': group_stds.to_dict()
                })
            
            impact_stats[attr] = pd.DataFrame(group_stats)
            
            # Log concerning disparities
            high_disparity_features = impact_stats[attr][
                impact_stats[attr]['mean_disparity'] > FAIRNESS['threshold']
            ]
            if not high_disparity_features.empty:
                logger.warning(
                    f"Features with high demographic disparity for {attr}:\n"
                    f"{high_disparity_features[['feature', 'mean_disparity']]}"
                )
        
        return impact_stats
        
    except Exception as e:
        logger.error(f"Error analyzing demographic impact: {str(e)}")
        raise

def validate_feature_parameters(params: Dict) -> bool:
    """
    Validates feature engineering parameters against config.
    
    Args:
        params: Optional parameters for feature engineering

    Returns:
        bool: True if parameters are valid
    """
    try:
        # Check correlation threshold
        if not 0 <= params.get('correlation_threshold', FEATURE_ENGINEERING['correlation_threshold']) <= 1:
            logger.error("Correlation threshold must be between 0 and 1")
            return False
            
        # Check importance threshold
        if params.get('importance_threshold', FEATURE_ENGINEERING['importance_threshold']) < 0:
            logger.error("Importance threshold must be non-negative")
            return False
            
        # Check window sizes
        window_sizes = params.get('window_sizes', FEATURE_ENGINEERING['window_sizes'])
        if not all(isinstance(w, int) and w > 0 for w in window_sizes):
            logger.error("Window sizes must be positive integers")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating parameters: {str(e)}")
        return False

def export_feature_metadata(
    features: pd.DataFrame,
    output_path: str,
    params: Optional[Dict] = None
) -> str:
    """
Exports feature metadata for downstream reference.
    
    Args:
        features: DataFrame containing features
        output_path: Path to save metadata file
        params: Optional parameters for feature engineering
        
    Returns:
        Path to saved metadata file
"""
    try:
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate basic statistics
        feature_stats = calculate_feature_statistics(features)
        
        # Analyze demographic impact
        demographic_impact = analyze_demographic_impact(features)
        
        # Create enhanced metadata
        metadata = {
            'feature_count': len(features.columns),
            'numeric_features': len(features.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_features': len(features.select_dtypes(include=['object', 'category']).columns),
            'row_count': len(features),
            'memory_usage': features.memory_usage(deep=True).sum(),
            'protected_attributes': [
                attr for attr in FAIRNESS['protected_attributes']
                if attr in features.columns
            ],
            'feature_statistics': feature_stats.to_dict(orient='records'),
            'demographic_impact': {
                attr: impact_df.to_dict(orient='records')
                for attr, impact_df in demographic_impact.items()
            },
            'parameters': params or FEATURE_ENGINEERING,
            'timestamp': pd.Timestamp.now().isoformat(),
            'version': VERSION_CONTROL.get('version_format', 'v1.0.0')
        }
        
        # Export metadata
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported enhanced feature metadata to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error exporting feature metadata: {str(e)}")
        raise

def analyze_feature_importance(
    features: pd.DataFrame,
    target: pd.Series,
    feature_type: str,
    save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Analyzes and visualizes feature importance.
    
    Args:
        features: DataFrame containing features
        target: Series containing target variable
        feature_type: Type of features (e.g., demographic, sequential)
        save_path: Optional path to save visualization
        
    Returns:
        Tuple containing DataFrame with importance scores and path to visualization
    """
    try:
        # Calculate feature importance using mutual information
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        importance_scores = mutual_info_classif(numeric_features, target)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': numeric_features.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Create visualization directory if it doesn't exist
        viz_dir = Path(DIRS['visualizations']) / 'features'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title(f'Top 15 {feature_type} Feature Importance Scores')
        plt.tight_layout()
        
        # Save visualization
        viz_path = viz_dir / f'{feature_type}_importance.png'
        plt.savefig(viz_path)
        plt.close()
        
        # Log results
        logger.info(f"Generated feature importance visualization: {viz_path}")
        
        # Check for low importance features
        low_importance = importance_df[
            importance_df['importance'] < FEATURE_ENGINEERING['importance_threshold']
        ]
        if not low_importance.empty:
            logger.warning(
                f"Found {len(low_importance)} features with low importance "
                f"(< {FEATURE_ENGINEERING['importance_threshold']})"
            )
            
        return importance_df, str(viz_path)
        
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {str(e)}")
        raise

def analyze_feature_correlations(
    features: pd.DataFrame,
    feature_type: str
) -> pd.DataFrame:
    """
    Analyzes and visualizes feature correlations.
    
    Args:
        features: DataFrame containing features
        feature_type: Type of features (e.g., demographic, sequential)
        
    Returns:
        DataFrame with highly correlated feature pairs
    """
    try:
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        
        # Calculate correlation matrix
        corr_matrix = numeric_features.corr()
        
        # Create visualization directory
        viz_dir = Path(DIRS['visualizations']) / 'features'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            cmap='RdBu',
            center=0,
            annot=True,
            fmt='.2f',
            square=True
        )
        plt.title(f'{feature_type} Feature Correlations')
        plt.tight_layout()
        
        # Save visualization
        viz_path = viz_dir / f'{feature_type}_correlations.png'
        plt.savefig(viz_path)
        plt.close()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > FEATURE_ENGINEERING['correlation_threshold']:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
                    
        if high_corr_pairs:
            logger.warning(
                f"Found {len(high_corr_pairs)} highly correlated feature pairs "
                f"(> {FEATURE_ENGINEERING['correlation_threshold']})"
            )
            
        return pd.DataFrame(high_corr_pairs)
        
    except Exception as e:
        logger.error(f"Error analyzing feature correlations: {str(e)}")
        raise

def generate_feature_report(
    importance_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    feature_type: str
) -> None:
    """
    Generates a comprehensive feature report.
    
    Args:
        importance_df: DataFrame with feature importance scores
        correlation_df: DataFrame with feature correlations
        feature_type: Type of features (e.g., demographic, sequential)
    """
    try:
        report_dir = Path(DIRS['reports']) / 'features'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'feature_type': feature_type,
            'total_features': len(importance_df),
            'top_features': importance_df.head(10).to_dict(orient='records'),
            'high_correlations': correlation_df.to_dict(orient='records'),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        report_path = report_dir / f'{feature_type}_report.json'
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Generated feature report: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating feature report: {str(e)}")
        raise