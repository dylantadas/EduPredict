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
    clean_assessment_data,
    clean_registration_data
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
    
    # Mode selection (required)
    parser.add_argument("--mode", type=str, required=True, choices=[
        "full", "processing", "feature", "rf", "gru", "ensemble", "evaluate", "visualize"
    ], help="Execution mode determining which pipeline components to run")
    
    # Data loading shortcuts
    parser.add_argument("--load_processed", action="store_true",
        help="Load pre-processed data from the default location (skips data processing)")
    parser.add_argument("--load_features", action="store_true",
        help="Load pre-engineered features from the default location (skips feature engineering)")
    
    # Original arguments
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
            datasets,
            logger=logger
        )
        
        if quality_report['recommendations']:
            logger.warning("Data quality recommendations:")
            for rec in quality_report['recommendations']:
                logger.warning(f"- {rec}")
        
        # Clean data with progress tracking
        logger.info("Cleaning datasets...")
        clean_data = {}
        total_datasets = 4
        
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
        
        # Clean registration data
        log_progress(logger, "Registration cleaning", 4, total_datasets)
        clean_data['registration'] = clean_registration_data(
            datasets['student_registration'],
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

def apply_fairness_aware_scaling(
    feature_df: pd.DataFrame,
    reference_data: pd.DataFrame,
    protected_attributes: Dict,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Applies fairness-aware feature scaling within demographic groups.
    
    Args:
        feature_df: DataFrame of features to scale
        reference_data: DataFrame containing protected attributes
        protected_attributes: Dictionary of protected attributes
        logger: Optional logger instance
        
    Returns:
        Scaled DataFrame
    """
    scaled_df = feature_df.copy()
    
    for col in scaled_df.select_dtypes(include=['int64', 'float64']).columns:
        for protected_attr in protected_attributes.keys():
            if protected_attr in reference_data.columns:
                # Calculate group-specific statistics
                group_stats = scaled_df.groupby(reference_data[protected_attr])[col].agg(['mean', 'std'])
                
                # Scale features within each demographic group
                for group in group_stats.index:
                    mask = reference_data[protected_attr] == group
                    group_mean = group_stats.loc[group, 'mean']
                    # Clip std to prevent division by zero
                    group_std = np.clip(group_stats.loc[group, 'std'], 1e-6, None)
                    scaled_df.loc[mask, col] = (scaled_df.loc[mask, col] - group_mean) / group_std
                    
    return scaled_df

@track_execution_time
def run_feature_engineering_workflow(
    data_splits: Dict[str, pd.DataFrame],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    data_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orchestrates feature engineering workflow with fairness monitoring.

    Args:
        data_splits (Dict[str, pd.DataFrame]): Data splits for training, validation, and testing
        output_dir (Path): Directory for output files
        logger (Optional[logging.Logger]): Logger instance
        data_results (Optional[Dict[str, Any]]): Processed data and metadata

    Returns:
        Dict[str, Any]: Processed features and metadata
    """
    try:
        # Add debug logging for input data validation
        logger.debug("Data splits keys: %s", list(data_splits.keys()))
        logger.debug("Train data columns: %s", list(data_splits['train'].columns))
        logger.debug("Train data types:\n%s", data_splits['train'].dtypes)

        # Monitor initial demographic distributions
        logger.info("Analyzing initial demographic distributions...")
        for protected_attr in PROTECTED_ATTRIBUTES.keys():
            if protected_attr in data_splits['train'].columns:
                dist = data_splits['train'][protected_attr].value_counts(normalize=True)
                logger.debug(f"Distribution for {protected_attr}:\n{dist}")

        # Debug log the parameters being passed to create_demographic_features
        logger.debug("Feature engineering parameters:")
        logger.debug("Encoding method: %s", 'both')
        logger.debug("Create interaction terms: %s", True)
        logger.debug("Min group size: %s", FAIRNESS['min_group_size'])
        logger.debug("Protected attributes: %s", list(PROTECTED_ATTRIBUTES.keys()))

        # Create demographic features with fairness monitoring
        logger.info("Creating demographic features...")
        try:
            feature_params = {
                'encoding_method': 'both',
                'create_interaction_terms': True,
                'min_group_size': int(FAIRNESS['min_group_size'])  # Ensure this is an int
            }
            demographic_features = create_demographic_features(
                data_splits['train'],
                params=feature_params
            )
            logger.debug("Demographic features created successfully")
            logger.debug("Demographic features shape: %s", demographic_features.shape)
            logger.debug("Demographic features columns: %s", list(demographic_features.columns))
        except Exception as e:
            logger.error("Error in create_demographic_features: %s", str(e), exc_info=True)
            logger.debug("Function type of create_demographic_features: %s", type(create_demographic_features))
            raise

        # Force garbage collection after demographic features
        import gc
        gc.collect()

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

        # Create temporal features with demographic monitoring - MEMORY OPTIMIZATION
        logger.info("Creating temporal features...")
        
        # Use a reduced set of columns for vle data to reduce memory footprint
        vle_data_subset = data_results['clean_data']['vle'][
            ['id_student', 'code_module', 'code_presentation', 'date', 'sum_click', 'id_site', 'activity_type']
        ].copy()
        
        # Convert to efficient dtypes
        vle_data_subset['date'] = vle_data_subset['date'].astype(np.float32)
        vle_data_subset['sum_click'] = vle_data_subset['sum_click'].astype(np.float32)
        vle_data_subset['id_site'] = vle_data_subset['id_site'].astype(np.int32)
        
        # Process window sizes one at a time to reduce memory pressure
        temporal_features_dict = {}
        for window_size in FEATURE_ENGINEERING['window_sizes']:
            logger.info(f"Processing window size: {window_size}")
            
            # Create the single window feature
            single_window_dict = create_temporal_features(
                vle_data_subset,  # Use optimized subset
                demographic_features,
                [window_size]  # Process just one size at a time
            )
            
            # Add to main dictionary
            if single_window_dict:
                temporal_features_dict.update(single_window_dict)
                
            # Force garbage collection after each window
            gc.collect()
            
            # Save intermediate results to disk to reduce memory pressure
            window_key = f'window_{window_size}'
            if window_key in temporal_features_dict:
                try:
                    # Create directory if it doesn't exist
                    intermediate_dir = Path(DIRS['intermediate']) / 'window_features'
                    intermediate_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save to disk and remove from memory
                    tmp_path = intermediate_dir / f"{window_key}.parquet"
                    temporal_features_dict[window_key].to_parquet(str(tmp_path))
                    logger.info(f"Saved {window_key} to disk at {tmp_path}")
                    
                    # Replace DataFrame with path to reduce memory
                    temporal_features_dict[window_key] = str(tmp_path)
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Could not save {window_key} to disk: {str(e)}")

        # Convert the dictionary of temporal features into a single DataFrame
        if temporal_features_dict:
            # Load window features back from disk if needed
            window_dfs = []
            for window_name, window_data in temporal_features_dict.items():
                try:
                    # Check if this is a path string or a DataFrame
                    if isinstance(window_data, str):
                        window_df = pd.read_parquet(window_data)
                        logger.info(f"Loaded {window_name} from {window_data}")
                    else:
                        window_df = window_data
                    
                    # Add window size as a prefix to all column names to avoid conflicts
                    prefixed_df = window_df.copy()
                    prefixed_df.columns = [f"{window_name}_{col}" if col not in ['id_student', 'code_module', 'code_presentation'] 
                                          else col for col in prefixed_df.columns]
                    window_dfs.append(prefixed_df)
                except Exception as e:
                    logger.warning(f"Error processing window {window_name}: {str(e)}")
            
            # Merge all window DataFrames on student, module and presentation
            if window_dfs:
                # Use progressive merging to reduce memory usage
                temporal_features = window_dfs[0]
                for i, df in enumerate(window_dfs[1:]):
                    logger.info(f"Merging window DataFrame {i+1}/{len(window_dfs)-1}")
                    temporal_features = pd.merge(
                        temporal_features, 
                        df,
                        on=['id_student', 'code_module', 'code_presentation'],
                        how='outer'
                    )
                    # Force garbage collection after each merge
                    gc.collect()
            else:
                # Create an empty DataFrame if no temporal features were generated
                logger.warning("No temporal features were generated")
                temporal_features = pd.DataFrame()
        else:
            # Create an empty DataFrame if temporal_features_dict is empty
            logger.warning("Temporal features dictionary is empty")
            temporal_features = pd.DataFrame()

        # Create sequential features with reduced memory settings
        logger.info("Creating sequential features...")
        sequential_features = create_sequential_features(
            data_results['clean_data']['vle'],
            chunk_size=min(5000, DATA_PROCESSING['chunk_size'])  # Use smaller chunks
        )
        
        # Force garbage collection
        gc.collect()

        # Ensure all feature DataFrames have compatible indices
        common_index = data_splits['train'].index
        demographic_features = demographic_features.reindex(common_index)
        temporal_features = temporal_features.reindex(common_index)
        sequential_features = sequential_features.reindex(common_index)

        # Apply fairness-aware feature scaling using the new function
        logger.info("Applying fairness-aware feature scaling...")
        demographic_features = apply_fairness_aware_scaling(
            demographic_features, 
            data_splits['train'],
            PROTECTED_ATTRIBUTES,
            logger
        )
        temporal_features = apply_fairness_aware_scaling(
            temporal_features,
            data_splits['train'],
            PROTECTED_ATTRIBUTES,
            logger
        )
        sequential_features = apply_fairness_aware_scaling(
            sequential_features,
            data_splits['train'],
            PROTECTED_ATTRIBUTES,
            logger
        )

        # Combine features progressively to manage memory
        logger.info("Combining features with memory optimization...")
        combined_features = demographic_features.copy()
        # First, merge with temporal features
        combined_features = pd.merge(
            combined_features,
            temporal_features,
            left_index=True,
            right_index=True,
            how='left'
        )
        # Force garbage collection
        gc.collect()
        # Then merge with sequential features
        combined_features = pd.merge(
            combined_features,
            sequential_features,
            left_index=True,
            right_index=True,
            how='left'
        )
        # Force final garbage collection
        gc.collect()

        # Handle missing values with fairness awareness
        logger.info("Handling missing values with fairness awareness...")
        for col in combined_features.select_dtypes(include=['int64', 'float64']).columns:
            if combined_features[col].isnull().any():
                for protected_attr in PROTECTED_ATTRIBUTES.keys():
                    if protected_attr in data_splits['train'].columns:
                        # Calculate group-specific medians for missing value imputation
                        group_medians = combined_features.groupby(
                            data_splits['train'][protected_attr]
                        )[col].transform('median')
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
        logger.info(f"Starting EduPredict pipeline in {args.mode} mode with chunk size: {args.chunk_size}")
        initial_memory = monitor_memory_usage("Pipeline start")
        
        # Validate configuration
        if not validate_configuration():
            logger.error("Configuration validation failed")
            return
        
        data_results = None
        feature_results = None
        
        # Process data if needed (for all modes except visualize)
        if not args.load_processed and args.mode not in ["visualize"]:
            logger.info("Starting data processing workflow...")
            data_results = run_data_processing_workflow(args, dirs, logger)
            
            # Save processed data for future runs
            logger.info("Saving processed data for future use...")
            processed_data_dir = Path(dirs['intermediate']) / 'processed_data'
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to save as parquet, fallback to CSV if parquet isn't available
            try:
                for name, data in data_results['clean_data'].items():
                    data.to_parquet(str(processed_data_dir / f"{name}.parquet"), index=False)
                    logger.info(f"Saved {name} data to {processed_data_dir / f'{name}.parquet'}")
                    
                # Save splits
                for split_name, split_data in data_results['splits'].items():
                    split_data.to_parquet(str(processed_data_dir / f"split_{split_name}.parquet"), index=True)
                    logger.info(f"Saved {split_name} split to {processed_data_dir / f'split_{split_name}.parquet'}")
            except ImportError as e:
                logger.warning(f"Parquet support missing: {str(e)}. Falling back to CSV format.")
                for name, data in data_results['clean_data'].items():
                    data.to_csv(str(processed_data_dir / f"{name}.csv"), index=False)
                    logger.info(f"Saved {name} data to {processed_data_dir / f'{name}.csv'}")
                    
                # Save splits
                for split_name, split_data in data_results['splits'].items():
                    split_data.to_csv(str(processed_data_dir / f"split_{split_name}.csv"), index=True)
                    logger.info(f"Saved {split_name} split to {processed_data_dir / f'split_{split_name}.csv'}")
        else:
            # Load pre-processed data (needed for all modes except processing and visualize)
            if args.mode not in ["processing", "visualize"]:
                logger.info("Loading pre-processed data...")
                processed_data_dir = Path(dirs['intermediate']) / 'processed_data'
                
                # Check if processed data exists
                if not processed_data_dir.exists():
                    logger.error("Processed data directory does not exist. Run without --load_processed first.")
                    return
                
                # Load cleaned data
                clean_data = {}
                for name in ['demographics', 'vle', 'assessments', 'registration']:
                    parquet_path = processed_data_dir / f"{name}.parquet"
                    csv_path = processed_data_dir / f"{name}.csv"
                    
                    if parquet_path.exists():
                        try:
                            clean_data[name] = pd.read_parquet(str(parquet_path))
                            logger.info(f"Loaded {name} data from parquet: {clean_data[name].shape}")
                        except ImportError:
                            if csv_path.exists():
                                clean_data[name] = pd.read_csv(str(csv_path))
                                logger.info(f"Loaded {name} data from csv: {clean_data[name].shape}")
                            else:
                                logger.warning(f"Could not find {name} csv data file. Some features may be unavailable.")
                    elif csv_path.exists():
                        clean_data[name] = pd.read_csv(str(csv_path))
                        logger.info(f"Loaded {name} data from csv: {clean_data[name].shape}")
                    else:
                        logger.warning(f"Could not find {name} data file. Some features may be unavailable.")
                
                # Load splits
                splits = {}
                for split_name in ['train', 'validation', 'test']:
                    parquet_path = processed_data_dir / f"split_{split_name}.parquet"
                    csv_path = processed_data_dir / f"split_{split_name}.csv"
                    
                    if parquet_path.exists():
                        try:
                            splits[split_name] = pd.read_parquet(str(parquet_path))
                            logger.info(f"Loaded {split_name} split from parquet: {splits[split_name].shape}")
                        except ImportError:
                            if csv_path.exists():
                                splits[split_name] = pd.read_csv(str(csv_path), index_col=0)
                                logger.info(f"Loaded {split_name} split from csv: {splits[split_name].shape}")
                            else:
                                logger.error(f"Could not find {split_name} split csv file. Cannot continue.")
                                return
                    elif csv_path.exists():
                        splits[split_name] = pd.read_csv(str(csv_path), index_col=0)
                        logger.info(f"Loaded {split_name} split from csv: {splits[split_name].shape}")
                    else:
                        logger.error(f"Could not find {split_name} split file. Cannot continue.")
                        return
                
                data_results = {
                    'clean_data': clean_data,
                    'splits': splits
                }

        # Feature engineering based on mode
        if not args.load_features and args.mode not in ["processing", "visualize"]:
            logger.info("Starting feature engineering workflow...")
            feature_results = run_feature_engineering_workflow(
                data_results['splits'],
                Path(args.output_dir),
                logger,
                data_results
            )
            
            # Save features for future runs
            logger.info("Saving engineered features for future use...")
            features_dir = Path(dirs['intermediate']) / 'features'
            features_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to save as parquet, fallback to CSV if parquet isn't available
            try:
                for name, features in feature_results.items():
                    if isinstance(features, pd.DataFrame):
                        features.to_parquet(str(features_dir / f"{name}.parquet"))
                        logger.info(f"Saved {name} to {features_dir / f'{name}.parquet'}")
            except ImportError as e:
                logger.warning(f"Parquet support missing: {str(e)}. Falling back to CSV format.")
                for name, features in feature_results.items():
                    if isinstance(features, pd.DataFrame):
                        features.to_csv(str(features_dir / f"{name}.csv"))
                        logger.info(f"Saved {name} to {features_dir / f'{name}.csv'}")
        elif args.mode not in ["processing", "visualize", "feature"]:
            # Load pre-engineered features
            logger.info("Loading pre-engineered features...")
            features_dir = Path(dirs['intermediate']) / 'features'
            
            if not features_dir.exists():
                logger.error("Features directory does not exist. Run without --load_features first.")
                return
            
            feature_results = {}
            for name in ['demographic_features', 'temporal_features', 'sequential_features', 'combined_features']:
                parquet_path = features_dir / f"{name}.parquet"
                csv_path = features_dir / f"{name}.csv"
                
                if parquet_path.exists():
                    try:
                        feature_results[name] = pd.read_parquet(str(parquet_path))
                        logger.info(f"Loaded {name} from parquet: {feature_results[name].shape}")
                    except ImportError:
                        if csv_path.exists():
                            feature_results[name] = pd.read_csv(str(csv_path), index_col=0)
                            logger.info(f"Loaded {name} from csv: {feature_results[name].shape}")
                        else:
                            logger.warning(f"Could not find {name} csv file. Some models may not work correctly.")
                elif csv_path.exists():
                    feature_results[name] = pd.read_csv(str(csv_path), index_col=0)
                    logger.info(f"Loaded {name} from csv: {feature_results[name].shape}")
                else:
                    logger.warning(f"Could not find {name} file. Some models may not work correctly.")
        
        # If mode is "feature", exit here as we've completed feature engineering
        if args.mode == "feature":
            logger.info("Feature engineering mode complete. Exiting...")
            execution_time = time.time() - start_time
            logger.info(f"Feature engineering completed successfully in {execution_time:.2f} seconds")
            monitor_memory_usage("Feature engineering completion")
            return
        
        # Run the appropriate model workflow based on mode
        if args.mode == "rf" or args.mode == "full":
            logger.info("Starting Random Forest model training...")
            # Call RandomForest training function here
            # rf_results = run_random_forest_workflow(feature_results, data_results['splits'], Path(args.output_dir), logger)
            logger.info("Random Forest training complete")
            
        if args.mode == "gru" or args.mode == "full":
            logger.info("Starting GRU model training...")
            # Call GRU training function here
            # gru_results = run_gru_model_workflow(feature_results, data_results['splits'], Path(args.output_dir), logger)
            logger.info("GRU training complete")
            
        if args.mode == "ensemble" or args.mode == "full":
            logger.info("Starting ensemble model training...")
            # Call ensemble model training function here
            # ensemble_results = run_ensemble_model_workflow(rf_results, gru_results, data_results['splits'], Path(args.output_dir), logger)
            logger.info("Ensemble training complete")
            
        if args.mode == "evaluate" or args.mode == "full":
            logger.info("Starting model evaluation...")
            # Call model evaluation function here
            # evaluation_results = run_evaluation_workflow(...) 
            logger.info("Model evaluation complete")
            
        if args.mode == "visualize":
            logger.info("Starting visualization generation...")
            # Call visualization function here
            # visualization_results = run_visualization_workflow(Path(args.output_dir), logger)
            logger.info("Visualization generation complete")
        
        # Log completion statistics
        execution_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
        monitor_memory_usage("Pipeline completion")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()