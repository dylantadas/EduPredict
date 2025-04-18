#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EduPredict: Early Warning Academic Performance Prediction System

This is the main script for the EduPredict system, implementing a dual-path
ensemble model that combines static demographic features and sequential
engagement features to predict student outcomes.
"""

import os
import sys
import argparse
import time
import json
import logging
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm.auto import tqdm
import warnings
import joblib
import psutil

# add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EduPredict'))
sys.path.insert(0, project_root)

# import configuration
from config import (
    DATA_PATH, OUTPUT_DIR, MODEL_DIR, RANDOM_SEED,
    TEST_SIZE, VALIDATION_SIZE, WINDOW_SIZES,
    CORRELATION_THRESHOLD, IMPORTANCE_THRESHOLD,
    RF_PARAM_GRID, RF_DEFAULT_PARAMS,
    GRU_PARAM_GRID, GRU_DEFAULT_PARAMS,
    FAIRNESS_THRESHOLDS, PROTECTED_ATTRIBUTES,
    DEMOGRAPHIC_COLS
)

# suppress warnings
warnings.filterwarnings('ignore')

# set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.max_columns', None)

# memory management functions
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def optimize_memory():
    """Optimize memory usage by clearing caches and running garbage collection."""
    gc.collect()
    tf.keras.backend.clear_session()
    
def log_memory_usage(logger, operation):
    """Log memory usage for an operation."""
    memory_used = get_memory_usage()
    logger.info(f"Memory usage after {operation}: {memory_used:.2f} MB")

# configure logging with tqdm-compatible output
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edupredict.log'),
        TqdmLoggingHandler()
    ]
)
logger = logging.getLogger('edupredict')

# progress tracking class
class ProgressTracker:
    def __init__(self, total_steps, desc="Processing"):
        self.pbar = tqdm(total=total_steps, desc=desc)
        self.start_time = time.time()
        
    def update(self, step_name):
        self.pbar.update(1)
        self.pbar.set_description(f"Processing {step_name}")
        logger.info(f"Completed step: {step_name}")
        log_memory_usage(logger, step_name)
        
    def close(self):
        elapsed_time = time.time() - self.start_time
        self.pbar.close()
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")

# add project root to python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# import from modules
from data_processing.data_processing import (
    load_raw_datasets,  # Needs dataset_paths parameter
    clean_demographic_data,
    clean_vle_data,
    clean_assessment_data,
    validate_data_consistency  # Missing implementation
)

from data_processing.feature_engineering import (
    create_demographic_features,
    create_temporal_features,  # Needs window_sizes parameter
    create_assessment_features,
    create_sequential_features,
    prepare_dual_path_features,  # Needs all feature types as parameters
    create_stratified_splits,  # Needs stratification_cols parameter
    prepare_target_variable  # Missing implementation
)

from data_analysis.eda import (
    perform_automated_eda,
    analyze_student_performance,
    analyze_engagement_patterns,
    document_eda_findings,
    visualize_demographic_distributions,
    visualize_performance_by_demographics,
    visualize_engagement_patterns
)

from model_training.random_forest_model import (
    RandomForestModel,  # Constructor needs alignment with tuning params
    find_optimal_threshold  # Missing implementation
)

from model_training.gru_model import (
    GRUModel,
    SequencePreprocessor,
    prepare_gru_training_data
)

from model_training.hyperparameter_tuning import (
    tune_random_forest,
    tune_gru_hyperparameters,
    optimize_ensemble_weights,
    visualize_tuning_results,
    visualize_ensemble_weights,
    save_tuning_results
)

from ensemble.ensemble import (
    EnsembleModel,
    combine_model_predictions
)

from evaluation.performance_metrics import (
    analyze_feature_importance,
    analyze_feature_correlations,
    calculate_model_metrics,
    calculate_fairness_metrics,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_fairness_metrics
)

from evaluation.fairness_analysis import (
    evaluate_model_fairness,  # Needs clearer metrics parameters
    generate_fairness_report,  # Missing implementation
    analyze_subgroup_fairness,  # Imported but not used
    mitigate_bias_with_thresholds  # Missing implementation
)

from visualization.tableau_export import (
    prepare_student_risk_data,
    prepare_temporal_engagement_data,
    prepare_assessment_performance_data,
    prepare_demographic_fairness_data,
    prepare_model_performance_data,
    export_for_tableau,
    generate_summary_visualizations,
    create_tableau_instructions
)

from visualization.visualization_runner import (
    VisualizationRunner  # Methods need alignment with usage
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EduPredict: Early Warning Academic Performance Prediction')
    
    parser.add_argument('--data_path', type=str, default='./data/OULAD',
                        help='Path to the data directory')
    
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'data_only', 'rf_only', 'gru_only', 'ensemble_only', 'evaluation_only', 'visualization_only'],
                        help='Processing mode')
    
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Directory to save/load models')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    parser.add_argument('--export_tableau', action='store_true',
                        help='Export data for Tableau')
    
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for GRU model training')
    
    parser.add_argument('--fairness_threshold', type=float, default=0.05,
                        help='Maximum allowed demographic parity difference')
    
    parser.add_argument('--importance_threshold', type=float, default=0.01,
                        help='Feature importance threshold for selection')
                        
    parser.add_argument('--correlation_threshold', type=float, default=0.85,
                        help='Correlation threshold for feature selection')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories."""
    # ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    # create subdirectories
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'feature_data'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tableau_data'), exist_ok=True)
    
    # return paths dictionary
    return {
        'output_dir': args.output_dir,
        'model_dir': args.model_dir,
        'viz_dir': os.path.join(args.output_dir, 'visualizations'),
        'report_dir': os.path.join(args.output_dir, 'reports'),
        'feature_dir': os.path.join(args.output_dir, 'feature_data'),
        'tableau_dir': os.path.join(args.output_dir, 'tableau_data')
    }


def load_and_process_data(data_path, output_paths, visualize=False):
    """Load and process raw data."""
    logger.info("Starting data loading and processing")
    
    # Load and validate data using data processing module
    datasets = load_raw_datasets(data_path)
    validate_data_consistency(datasets)
    
    # Clean data using data processing module
    clean_data = {
        'demographics': clean_demographic_data(datasets['student_info']),
        'vle': clean_vle_data(datasets['vle_interactions'], datasets['vle_materials']),
        'assessments': clean_assessment_data(datasets['assessments'], datasets['student_assessments'])
    }
    
    # Log shapes
    for name, data in clean_data.items():
        logger.info(f"Clean {name} shape: {data.shape}")
    
    # Save cleaned data
    for name, data in clean_data.items():
        data.to_csv(os.path.join(output_paths['feature_dir'], f'clean_{name}.csv'), index=False)
    
    # run exploratory data analysis
    if visualize:
        viz_runner = VisualizationRunner(output_paths['viz_dir'])
        viz_paths = viz_runner.run_demographic_visualizations(clean_data['demographics'])
        logger.info(f"Generated demographic visualizations: {viz_paths}")
        
        engagement_paths = viz_runner.run_engagement_visualizations(
            clean_data['vle'],
            clean_data['demographics']
        )
        logger.info(f"Generated engagement visualizations: {engagement_paths}")
    
    return clean_data


def engineer_features(clean_data, output_paths):
    """Generate features for both static and sequential paths."""
    logger.info("Starting feature engineering")
    
    # create demographic features
    demographic_features = create_demographic_features(clean_data['demographics'])
    logger.info(f"Demographic features shape: {demographic_features.shape}")
    
    # create temporal features with multiple window sizes
    window_sizes = [7, 14, 30]  # weekly, bi-weekly, monthly
    temporal_features = create_temporal_features(clean_data['vle'], window_sizes)
    
    # log temporal feature sizes
    for window_size, features in temporal_features.items():
        logger.info(f"{window_size} features shape: {features.shape}")
    
    # create assessment features
    assessment_features = create_assessment_features(clean_data['assessments'])
    logger.info(f"Assessment features shape: {assessment_features.shape}")
    
    # create sequential features for gru path
    sequential_features = create_sequential_features(clean_data['vle'])
    logger.info(f"Sequential features shape: {sequential_features.shape}")
    
    # prepare dual path features
    dual_path_features = prepare_dual_path_features(
        demographic_features, 
        temporal_features,
        assessment_features,
        sequential_features
    )
    
    # log dual path feature shapes
    for path_name, features in dual_path_features.items():
        logger.info(f"{path_name} shape: {features.shape}")
    
    # save feature data
    demographic_features.to_csv(os.path.join(output_paths['feature_dir'], 'demographic_features.csv'), index=False)
    sequential_features.to_csv(os.path.join(output_paths['feature_dir'], 'sequential_features.csv'), index=False)
    
    for window_size, features in temporal_features.items():
        features.to_csv(os.path.join(output_paths['feature_dir'], f'{window_size}_features.csv'), index=False)
    
    # save dual path features
    for path_name, features in dual_path_features.items():
        features.to_csv(os.path.join(output_paths['feature_dir'], f'{path_name}_features.csv'), index=False)
    
    # return engineered features
    return {
        'demographic_features': demographic_features,
        'temporal_features': temporal_features,
        'assessment_features': assessment_features,
        'sequential_features': sequential_features,
        'dual_path_features': dual_path_features
    }


def train_and_evaluate_models(feature_data, output_paths, args):
    """Orchestrates model training and evaluation across both paths."""
    model_results = {}
    
    # Train Random Forest if specified
    if args.mode in ['full', 'rf_only', 'ensemble_only']:
        model_results['rf'] = train_random_forest(feature_data, output_paths, args)
    
    # Train GRU if specified
    if args.mode in ['full', 'gru_only', 'ensemble_only']:
        model_results['gru'] = train_gru_model(
            feature_data,
            model_results.get('rf', {}).get('split_data'),
            output_paths,
            args
        )
    
    # Train Ensemble if specified
    if args.mode in ['full', 'ensemble_only'] and all(k in model_results for k in ['rf', 'gru']):
        model_results['ensemble'] = train_ensemble_model(
            model_results['rf'],
            model_results['gru'],
            model_results['rf']['split_data'],
            output_paths,
            args
        )
    
    return model_results


def evaluate_and_report(model_results, feature_data, clean_data, output_paths, args):
    """Handles evaluation, fairness analysis, and reporting."""
    
    # Perform fairness analysis
    fairness_results = evaluate_model_fairness(
        model_results,
        clean_data['demographics'],
        output_paths,
        args
    )
    
    # Generate visualizations if requested
    if args.visualize:
        viz_runner = VisualizationRunner(output_paths['viz_dir'])
        
        # Model performance visualizations
        for model_name, results in model_results.items():
            if 'metrics' in results:
                viz_runner.visualize_model_performance(
                    results['metrics'],
                    model_name.upper()
                )
        
        # Feature importance for RF model
        if 'rf' in model_results:
            rf_model = model_results['rf']['model']
            importance_df = rf_model.get_feature_importance(plot=False)
            viz_runner.visualize_feature_importance(importance_df)
    
    # Export data for Tableau if requested
    if args.export_tableau:
        export_data = prepare_visualizations(
            model_results,
            clean_data,
            fairness_results,
            output_paths,
            args
        )
    
    return fairness_results


def main():
    """Main function to orchestrate all processing steps."""
    # parse command line arguments
    args = parse_arguments()
    
    # setup directories
    output_paths = setup_directories(args)
    
    logger.info(f"Starting EduPredict in {args.mode} mode")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Initialize progress tracker
        progress = ProgressTracker(total_steps=6, desc="Processing")
        
        # Step 1: Load and process data
        clean_data = load_and_process_data(args.data_path, output_paths, args.visualize)
        progress.update("Data Loading")
        
        # Step 2: Feature engineering
        if args.mode != 'visualization_only':
            feature_data = engineer_features(clean_data, output_paths)
            progress.update("Feature Engineering")
        else:
            # Load pre-processed feature data
            feature_data = load_preprocessed_features(output_paths['feature_dir'])
            progress.update("Loading Features")
        
        # Step 3: Train models
        if args.mode not in ['data_only', 'visualization_only']:
            model_results = train_and_evaluate_models(feature_data, output_paths, args)
            progress.update("Model Training")
        else:
            model_results = None
            progress.update("Skipping Training")
        
        # Step 4: Evaluation and fairness analysis
        if model_results:
            fairness_results = evaluate_and_report(
                model_results,
                feature_data,
                clean_data,
                output_paths,
                args
            )
            progress.update("Evaluation")
        else:
            fairness_results = None
            progress.update("Skipping Evaluation")
        
        # Step 5: Export visualization data
        if args.export_tableau or args.visualize:
            if model_results:
                prepare_visualizations(
                    model_results,
                    clean_data,
                    fairness_results,
                    output_paths,
                    args
                )
            progress.update("Visualization Export")
        
        progress.close()
        logger.info(f"EduPredict completed successfully in {args.mode} mode")
        
    except Exception as e:
        logger.error(f"Error in EduPredict: {str(e)}", exc_info=True)
        raise
    
    return 0


if __name__ == "__main__":
    sys.exit(main())