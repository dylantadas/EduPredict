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
from config import FEATURE_ENGINEERING, FAIRNESS, DIRS, VERSION_CONTROL, RANDOM_SEED
from dataclasses import dataclass

logger = logging.getLogger('edupredict')

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

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
    target: pd.Series,
    threshold: float = FEATURE_ENGINEERING['importance_threshold']
) -> pd.DataFrame:
    """
    Selects features based on importance scores using RandomForest.
    
    Args:
        features: DataFrame containing features
        target: Target variable Series
        threshold: Minimum importance score to retain feature
        
    Returns:
        DataFrame with selected features
    """
    try:
        # Get numeric features
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        
        # Remove identifier columns and protected attributes while preserving them
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
            
        # Handle NaN values with fair imputation
        for col in feature_cols:
            if numeric_features[col].isnull().any():
                # Group by protected attributes for fair imputation
                for attr in FAIRNESS['protected_attributes']:
                    if attr in features.columns:
                        # Calculate group means
                        group_means = numeric_features.groupby(features[attr])[col].transform('mean')
                        # Fill NaN values with their group means
                        numeric_features.loc[numeric_features[col].isnull(), col] = group_means[numeric_features[col].isnull()]
                
                # Fill any remaining NaN values with overall mean
                if numeric_features[col].isnull().any():
                    numeric_features[col].fillna(numeric_features[col].mean(), inplace=True)
                    
                logger.info(f"Imputed NaN values in {col} using fairness-aware approach")
        
        # Initialize and train RandomForest model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED
        )
        rf_model.fit(numeric_features[feature_cols], target)
        
        # Get feature importance scores
        importance_scores = rf_model.feature_importances_
        importance_dict = dict(zip(feature_cols, importance_scores))
        
        # Select features above threshold while preserving protected attributes
        selected_features = [
            col for col, score in importance_dict.items()
            if score >= threshold
        ]
        
        # Always include identifier columns and protected attributes
        selected_features.extend(exclude_cols)
        
        # Log selected features with importance scores
        for col in selected_features:
            if col in importance_dict:
                logger.info(f"Selected feature {col} with importance: {importance_dict[col]:.4f}")
        
        logger.info(
            f"Selected {len(selected_features)} features "
            f"(threshold: {threshold})"
        )
        
        # Return features with selected columns
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
    Calculate comprehensive feature statistics with timeline awareness.
    Properly interprets temporal features and their relationships.
    """
    try:
        stats = pd.DataFrame()
        
        # Identify temporal columns
        temporal_cols = [col for col in features.columns if any(x in col.lower() for x in ['date', 'time', 'day', 'window'])]
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        
        # Basic statistics
        stats['mean'] = features[numeric_cols].mean()
        stats['std'] = features[numeric_cols].std()
        stats['missing_ratio'] = features[numeric_cols].isnull().mean()
        
        # Timeline-specific statistics for temporal features
        if temporal_cols:
            for col in temporal_cols:
                if col in numeric_cols:
                    stats.loc[col, 'is_temporal'] = True
                    stats.loc[col, 'min_date'] = features[col].min()
                    stats.loc[col, 'max_date'] = features[col].max()
                    stats.loc[col, 'timeline_span'] = features[col].max() - features[col].min()
                    stats.loc[col, 'pre_module_ratio'] = (features[col] < 0).mean() if 'date' in col else None
        
        # Check for potential data interpretation issues
        if any(stats['missing_ratio'] > 0.1):
            logger.warning(f"High missing ratio detected in features: {stats[stats['missing_ratio'] > 0.1].index.tolist()}")
        
        # Correlation with temporal features
        for tcol in temporal_cols:
            if tcol in numeric_cols:
                correlations = features[numeric_cols].corrwith(features[tcol])
                high_corr = correlations[abs(correlations) > 0.7].index.tolist()
                if high_corr:
                    logger.info(f"Features highly correlated with {tcol}: {high_corr}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating feature statistics: {str(e)}")
        raise

def analyze_feature_correlations(features: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes feature correlations with enhanced timeline understanding.
    Ensures temporal features are properly interpreted in correlation analysis.
    """
    try:
        # Identify different feature types
        temporal_cols = [col for col in features.columns if any(x in col.lower() for x in ['date', 'time', 'day', 'window'])]
        activity_cols = [col for col in features.columns if any(x in col.lower() for x in ['click', 'activity', 'session'])]
        performance_cols = [col for col in features.columns if any(x in col.lower() for x in ['score', 'grade', 'result'])]
        
        numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
        
        # Calculate correlations by feature type
        correlations = pd.DataFrame()
        
        # Timeline correlations
        if temporal_cols:
            time_corr = features[temporal_cols].corr()
            logger.info("Analyzing temporal feature relationships...")
            for col1 in temporal_cols:
                for col2 in temporal_cols:
                    if col1 < col2 and abs(time_corr.loc[col1, col2]) > 0.7:
                        logger.warning(f"Strong temporal correlation between {col1} and {col2}: {time_corr.loc[col1, col2]:.3f}")
        
        # Activity-timeline correlations
        if temporal_cols and activity_cols:
            activity_time_corr = features[temporal_cols + activity_cols].corr()
            logger.info("Analyzing activity-timeline relationships...")
            significant_correlations = []
            for tcol in temporal_cols:
                for acol in activity_cols:
                    corr = activity_time_corr.loc[tcol, acol]
                    if abs(corr) > 0.5:
                        significant_correlations.append((tcol, acol, corr))
            
            if significant_correlations:
                logger.info("Significant activity-timeline correlations found:")
                for t, a, c in significant_correlations:
                    logger.info(f"{t} - {a}: {c:.3f}")
        
        # Performance-timeline correlations
        if temporal_cols and performance_cols:
            perf_time_corr = features[temporal_cols + performance_cols].corr()
            logger.info("Analyzing performance-timeline relationships...")
            for tcol in temporal_cols:
                for pcol in performance_cols:
                    corr = perf_time_corr.loc[tcol, pcol]
                    if abs(corr) > 0.3:  # Lower threshold for performance correlations
                        logger.info(f"Performance-timeline correlation - {tcol} vs {pcol}: {corr:.3f}")
        
        # Overall correlations
        correlations = features[numeric_cols].corr()
        
        return correlations
        
    except Exception as e:
        logger.error(f"Error analyzing feature correlations: {str(e)}")
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
            'memory_usage': float(features.memory_usage(deep=True).sum()),  # Convert to native Python type
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
        
        # Export metadata using custom encoder
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyJSONEncoder)
        
        logger.info(f"Exported enhanced feature metadata to {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error exporting feature metadata: {str(e)}")
        raise

def analyze_feature_importance(
    features: pd.DataFrame,
    target: pd.Series,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    """
    Analyzes feature importance with fairness considerations, handling NaN values properly.
    """
    logger = logger or logging.getLogger('edupredict')

    try:
        # Handle NaN values first
        features_clean = features.copy()
        
        # Get numeric columns for importance analysis
        numeric_cols = features_clean.select_dtypes(include=['int64', 'float64']).columns
        numeric_features = features_clean[numeric_cols]
        
        # Handle remaining NaN values using fairness-aware imputation
        for col in numeric_cols:
            if numeric_features[col].isnull().any():
                # Use column median by default
                col_median = numeric_features[col].median()
                if pd.isnull(col_median):  # If median is NaN, use 0
                    col_median = 0
                numeric_features[col] = numeric_features[col].fillna(col_median)
                logger.info(f"Filled NaN values in {col} with median {col_median}")

        # Calculate importance scores
        importance_scores = mutual_info_classif(numeric_features, target)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': importance_scores
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Mutual Information Score')
        plt.ylabel('Feature')
        
        # Log top features
        logger.info("\nTop 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        return importance_df, plt.gcf()

    except Exception as e:
        logger.error(f"Error analyzing feature importance: {str(e)}")
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