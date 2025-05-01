import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
import psutil
import time
import matplotlib.pyplot as plt

from data_processing.data_loader import load_raw_datasets
from data_processing.data_cleaner import (
    clean_demographic_data,
    clean_vle_data, 
    clean_assessment_data
)

from data_processing.data_splitter import create_stratified_splits
from data_processing.data_monitor import detect_data_quality_issues

from feature_engineering.demographic_features import create_demographic_features
from feature_engineering.sequential_features import (
    create_sequential_features,
    combine_sequential_features
)

from feature_engineering.temporal_features import create_temporal_features
from feature_engineering.feature_selector import (
    analyze_feature_importance,
    analyze_feature_correlations,
    remove_correlated_features,
    select_features_by_importance,
    analyze_demographic_impact,
    export_feature_metadata
)

from utils.logging_utils import setup_logger, log_memory_usage, log_progress
from utils.monitoring_utils import track_execution_time, monitor_memory_usage
from utils.validation_utils import (
    validate_directories,
    validate_data_consistency,
    validate_feature_engineering_inputs
)

from config import (
    DIRS, DATA_PROCESSING, FEATURE_ENGINEERING, 
    FAIRNESS, BIAS_MITIGATION, PROTECTED_ATTRIBUTES,
    LOG_DIR, LOG_LEVEL
)

def validate_configuration() -> bool:
    """Validates all configuration parameters for consistency."""
    try:
        # Validate data processing parameters
        if DATA_PROCESSING['chunk_size'] <= 0:
            logger.error("Chunk size must be positive")
            return False
            
        # Validate feature engineering parameters
        if not all(size > 0 for size in FEATURE_ENGINEERING['window_sizes']):
            logger.error("Window sizes must be positive")
            return False
            
        if not 0 <= FEATURE_ENGINEERING['correlation_threshold'] <= 1:
            logger.error("Correlation threshold must be between 0 and 1")
            return False
            
        if FEATURE_ENGINEERING['importance_threshold'] < 0:
            logger.error("Importance threshold must be non-negative")
            return False
            
        # Validate fairness parameters
        if not 0 <= FAIRNESS['threshold'] <= 1:
            logger.error("Fairness threshold must be between 0 and 1")
            return False
            
        for metric, threshold in FAIRNESS['thresholds'].items():
            if metric == 'disparate_impact_ratio':
                if not 0 < threshold <= 1:
                    logger.error(f"Invalid threshold for {metric}: must be between 0 and 1")
                    return False
            else:
                if not 0 <= threshold <= 1:
                    logger.error(f"Invalid threshold for {metric}: must be between 0 and 1")
                    return False
                    
        if FAIRNESS['min_group_size'] < 1:
            logger.error("Minimum group size must be positive")
            return False
            
        # Validate protected attributes
        for attr in FAIRNESS['protected_attributes']:
            if attr not in PROTECTED_ATTRIBUTES:
                logger.error(f"Protected attribute {attr} not defined in PROTECTED_ATTRIBUTES")
                return False
                
        # Validate bias mitigation parameters
        if BIAS_MITIGATION['method'] not in ['reweight', 'oversample', 'undersample', 'none']:
            logger.error("Invalid bias mitigation method")
            return False
            
        if BIAS_MITIGATION['balance_strategy'] not in ['group_balanced', 'stratified']:
            logger.error("Invalid balance strategy")
            return False
            
        if BIAS_MITIGATION['max_ratio'] <= 1:
            logger.error("Maximum group ratio must be greater than 1")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation error: {str(e)}")
        return False

# Setup logging with custom handler
logger = setup_logger(LOG_DIR, LOG_LEVEL)

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments to control workflow execution.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """

    parser = argparse.ArgumentParser(description='EduPredict Pipeline')
    parser.add_argument('--data-dir', 
                       type=str, 
                       default='./data/OULAD',
                       help='Directory containing raw data files')
    parser.add_argument('--output-dir', 
                       type=str, 
                       default='./output',
                       help='Directory for output files')
    parser.add_argument('--chunk-size', 
                       type=int, 
                       default=DATA_PROCESSING['chunk_size'], 
                       help='Chunk size for processing large datasets')
    parser.add_argument('--window-sizes', 
                       type=int, 
                       nargs='+', 
                       default=FEATURE_ENGINEERING['window_sizes'],
                       help='Window sizes for temporal feature engineering')
    return parser.parse_args()

def setup_environment(args: argparse.Namespace) -> Tuple[Dict[str, str], logging.Logger]:
    """
    Sets up environment, directories, and logging.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
    
    Returns:
        Tuple[Dict[str, str], logging.Logger]: Directory paths and logger instance
    """

    # Create all required directories
    for dir_path in DIRS.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Validate directory structure
    if not validate_directories(DIRS):
        raise RuntimeError("Directory validation failed")
        
    return DIRS, logger

@track_execution_time
def run_data_processing_workflow(
    args: argparse.Namespace, 
    dirs: Dict[str, str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Orchestrates data processing workflow.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
        dirs (Dict[str, str]): Directory paths
        logger (logging.Logger): Logger instance
        
    Returns:
        Dict[str, Any]: Processed data and metadata
    """

    try:
        # Load raw datasets with memory tracking
        logger.info("Loading raw datasets...")
        initial_mem = monitor_memory_usage("Before data loading")
        datasets = load_raw_datasets(
            args.data_dir,
            chunk_size=args.chunk_size
        )
        monitor_memory_usage("After data loading")
        
        # Validate data consistency
        logger.info("Validating data consistency...")
        if not validate_data_consistency(datasets):
            raise ValueError("Data consistency validation failed")
        
        # Check data quality with focus on protected attributes
        logger.info("Checking data quality...")
        quality_report = detect_data_quality_issues(
            datasets['student_info'],
            protected_cols=FAIRNESS['protected_attributes'],
            output_dir=Path(dirs['reports_fairness'])
        )
        
        if quality_report['recommendations']:
            logger.warning("Data quality recommendations:")
            for rec in quality_report['recommendations']:
                logger.warning(f"- {rec}")
        
        # Clean data with progress tracking
        logger.info("Cleaning datasets...")
        clean_data = {}
        total_datasets = 3
        
        # Clean demographic data
        log_progress(logger, "Demographic cleaning", 1, total_datasets)
        clean_data['demographics'] = clean_demographic_data(
            datasets['student_info'],
            logger=logger
        )
        
        # Clean VLE data
        log_progress(logger, "VLE cleaning", 2, total_datasets)
        clean_data['vle'] = clean_vle_data(
            datasets['vle_interactions'],
            datasets['vle_materials'],
            logger=logger
        )
        
        # Clean assessment data
        log_progress(logger, "Assessment cleaning", 3, total_datasets)
        clean_data['assessments'] = clean_assessment_data(
            datasets['assessments'],
            datasets['student_assessments'],
            logger=logger
        )
        
        # Create stratified splits with demographic balance validation
        logger.info("Creating data splits...")
        splits = create_stratified_splits(
            clean_data['demographics'],
            target_col='final_result',
            strat_cols=FAIRNESS['protected_attributes'],
            test_size=0.2,
            validation_size=0.2
        )
        
        return {
            'clean_data': clean_data,
            'splits': splits,
            'quality_report': quality_report
        }
        
    except Exception as e:
        logger.error(f"Error in data processing workflow: {str(e)}")
        raise

@track_execution_time
def run_feature_engineering_workflow(
    data_splits: Dict[str, pd.DataFrame],
    output_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Orchestrates feature engineering workflow with fairness monitoring.

    Args:
        data_splits (Dict[str, pd.DataFrame]): Data splits for training, validation, and testing
        output_dir (Path): Directory for output files
        logger (Optional[logging.Logger]): Logger instance

    Returns:
        Dict[str, Any]: Processed features and metadata
    """
    try:
        # Monitor initial demographic distributions
        logger.info("Analyzing initial demographic distributions...")
        for protected_attr in PROTECTED_ATTRIBUTES.keys():
            if protected_attr in data_splits['train'].columns:
                dist = data_splits['train'][protected_attr].value_counts(normalize=True)
                logger.info(f"Initial {protected_attr} distribution:\n{dist}")

        # Create demographic features with fairness monitoring
        logger.info("Creating demographic features...")
        demographic_features = create_demographic_features(
            data_splits['train'],
            params={
                'encoding_method': 'both',
                'create_interaction_terms': True,
                'min_group_size': FAIRNESS['min_group_size']
            }
        )

        # Monitor demographic feature creation impact
        logger.info("Validating demographic feature fairness...")
        for protected_attr in PROTECTED_ATTRIBUTES.keys():
            if protected_attr in demographic_features.columns:
                before_dist = data_splits['train'][protected_attr].value_counts(normalize=True)
                after_dist = demographic_features[protected_attr].value_counts(normalize=True)
                max_diff = max(abs(before_dist - after_dist))
                if max_diff > FAIRNESS['threshold']:
                    logger.warning(
                        f"Significant change in {protected_attr} distribution "
                        f"after feature creation: {max_diff:.3f}"
                    )

        # Create temporal features with demographic monitoring
        logger.info("Creating temporal features...")
        temporal_features = create_temporal_features(
            data_splits['train'],
            demographic_features,
            FEATURE_ENGINEERING['window_sizes']
        )

        # Create sequential features
        logger.info("Creating sequential features...")
        sequential_features = create_sequential_features(
            data_splits['train'],
            demographic_features,
            logger=logger
        )

        # Ensure all feature DataFrames have compatible indices
        common_index = data_splits['train'].index
        demographic_features = demographic_features.reindex(common_index)
        temporal_features = temporal_features.reindex(common_index)
        sequential_features = sequential_features.reindex(common_index)

        # Apply fairness-aware feature scaling
        logger.info("Applying fairness-aware feature scaling...")
        for df in [demographic_features, temporal_features, sequential_features]:
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                for protected_attr in PROTECTED_ATTRIBUTES.keys():
                    if protected_attr in data_splits['train'].columns:
                        # Calculate group-specific statistics
                        group_stats = df.groupby(data_splits['train'][protected_attr])[col].agg(['mean', 'std'])
                        
                        # Scale features within each demographic group
                        for group in group_stats.index:
                            mask = data_splits['train'][protected_attr] == group
                            group_mean = group_stats.loc[group, 'mean']
                            # Clip std to prevent division by zero
                            group_std = np.clip(group_stats.loc[group, 'std'], 1e-6, None)
                            df.loc[mask, col] = (df.loc[mask, col] - group_mean) / group_std

        # Combine features with fairness monitoring
        combined_features = pd.concat(
            [demographic_features, temporal_features, sequential_features],
            axis=1
        )

        # Handle missing values with fairness awareness
        logger.info("Handling missing values with fairness awareness...")
        for col in combined_features.select_dtypes(include=['int64', 'float64']).columns:
            if combined_features[col].isnull().any():
                for protected_attr in PROTECTED_ATTRIBUTES.keys():
                    if protected_attr in data_splits['train'].columns:
                        # Calculate group-specific medians
                        group_medians = combined_features.groupby(
                            data_splits['train'][protected_attr]
                        )[col].transform('median')
                        
                        # Fill missing values with group-specific medians
                        combined_features[col] = combined_features[col].fillna(group_medians)
                
                # Fill any remaining nulls with global median
                if combined_features[col].isnull().any():
                    global_median = combined_features[col].median()
                    if pd.isnull(global_median):
                        global_median = 0
                    combined_features[col] = combined_features[col].fillna(global_median)

        # Log feature shapes and check for demographic balance
        logger.info("Validating final feature distributions...")
        logger.info(f"Demographic features shape: {demographic_features.shape}")
        logger.info(f"Temporal features shape: {temporal_features.shape}")
        logger.info(f"Sequential features shape: {sequential_features.shape}")
        logger.info(f"Combined features shape: {combined_features.shape}")

        # Analyze feature importance with fairness considerations
        logger.info("Analyzing feature importance...")
        feature_importance, importance_viz = analyze_feature_importance(
            features=combined_features,
            target=data_splits['train']['final_result'].reindex(combined_features.index),
            logger=logger
        )

        # Check feature importance fairness
        logger.info("Validating feature importance fairness...")
        top_features = feature_importance.head(10)['feature'].tolist()
        for feature in top_features:
            if feature in combined_features.columns:
                for protected_attr in PROTECTED_ATTRIBUTES.keys():
                    if protected_attr in data_splits['train'].columns:
                        group_stats = combined_features.groupby(
                            data_splits['train'][protected_attr]
                        )[feature].agg(['mean', 'std'])
                        
                        # Calculate disparity
                        max_mean = group_stats['mean'].max()
                        min_mean = group_stats['mean'].min()
                        if max_mean != 0:
                            disparity = (max_mean - min_mean) / max_mean
                            if disparity > FAIRNESS['threshold']:
                                logger.warning(
                                    f"High demographic disparity detected in important feature "
                                    f"{feature} for {protected_attr}: {disparity:.3f}"
                                )

        # Save visualization
        if importance_viz:
            viz_path = output_dir / 'visualizations' / 'feature_importance.png'
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            importance_viz.savefig(str(viz_path))
            plt.close(importance_viz)

        return {
            'demographic_features': demographic_features,
            'temporal_features': temporal_features,
            'sequential_features': sequential_features,
            'combined_features': combined_features,
            'feature_importance': feature_importance,
            'fairness_metrics': {
                'feature_disparities': top_features
            }
        }

    except Exception as e:
        logger.error(f"Error in feature engineering workflow: {str(e)}")
        raise

def main() -> None:
    """Entry point for the application."""
    try:
        start_time = time.time()
        
        # Parse arguments and setup environment
        args = parse_arguments()
        dirs, logger = setup_environment(args)
        
        # Log system information
        logger.info(f"Starting EduPredict pipeline with chunk size: {args.chunk_size}")
        initial_memory = monitor_memory_usage("Pipeline start")
        
        # Validate configuration
        if not validate_configuration():
            logger.error("Configuration validation failed")
            return
            
        # Run data processing workflow with timing
        logger.info("Starting data processing workflow...")
        data_results = run_data_processing_workflow(args, dirs, logger)
        
        # Run feature engineering workflow with timing
        logger.info("Starting feature engineering workflow...")
        feature_results = run_feature_engineering_workflow(
            data_results['splits'],
            Path(args.output_dir),
            logger
        )
        
        # Log completion statistics
        execution_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
        monitor_memory_usage("Pipeline completion")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()