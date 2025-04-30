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
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import tensorflow as tf # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tqdm.auto import tqdm # type: ignore
import warnings
import psutil

# add project root to python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# import configuration
from config import (
    DATA_PATH, OUTPUT_DIR, MODEL_DIR, RANDOM_SEED,
    TEST_SIZE, VALIDATION_SIZE, WINDOW_SIZES,
    CORRELATION_THRESHOLD, IMPORTANCE_THRESHOLD,
    RF_PARAM_GRID, RF_DEFAULT_PARAMS,
    GRU_PARAM_GRID, GRU_DEFAULT_PARAMS,
    FAIRNESS_THRESHOLDS, PROTECTED_ATTRIBUTES,
    DEMOGRAPHIC_COLS, BIAS_MITIGATION
)

# suppress warnings
# warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# import from modules
from data_processing.data_processing import (
    load_raw_datasets,
    clean_demographic_data,
    clean_vle_data,
    clean_assessment_data,
    validate_data_consistency
)

from data_processing.feature_engineering import (
    create_demographic_features,
    create_temporal_features,
    create_assessment_features,
    create_sequential_features,
    prepare_dual_path_features,
    create_stratified_splits,
    prepare_target_variable
)

'''
from data_analysis.eda import (
    perform_automated_eda,
    analyze_student_performance,
    analyze_engagement_patterns,
    document_eda_findings,
)
'''

from model_training.random_forest_model import (
    RandomForestModel,
)

from model_training.gru_model import (
    GRUModel,
    SequencePreprocessor,
    prepare_gru_training_data
)

from model_training.hyperparameter_tuning import (
    tune_random_forest,
    tune_gru_hyperparameters,
    optimize_ensemble_weights #,
    #visualize_tuning_results,
    #visualize_ensemble_weights,
    #save_tuning_results
)

from ensemble.ensemble import (
    EnsembleModel #,
    # combine_model_predictions
)

from evaluation.performance_metrics import (
    analyze_feature_importance,
    analyze_feature_correlations #,
    #calculate_model_metrics,
    #calculate_fairness_metrics,
    #plot_roc_curves,
    #plot_precision_recall_curves,
    #plot_fairness_metrics
)

from evaluation.fairness_analysis import (
    evaluate_model_fairness,
    resample_training_data,
    #generate_fairness_report,
    #mitigate_bias_with_thresholds
)

from visualization.tableau_export import (
    prepare_student_risk_data,
    prepare_temporal_engagement_data,
    prepare_assessment_performance_data,
    prepare_demographic_fairness_data,
    prepare_model_performance_data,
    export_for_tableau,
    create_tableau_instructions #,
    #generate_summary_visualizations

)

from visualization.visualization_runner import (
    VisualizationRunner
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
    
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Skip data loading and preprocessing stages')
    
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
    """Load and process raw data with chunked processing and optional EDA."""
    logger.info("Starting data loading and processing")
    
    # Load and validate data using data processing module with chunking
    datasets = load_raw_datasets(data_path)
    validate_data_consistency(datasets)
    
    # Clean data using data processing module with chunking
    clean_data = {
        'demographics': clean_demographic_data(datasets['student_info']),
        'vle': clean_vle_data(datasets['vle_interactions'], datasets['vle_materials'], chunk_size=100000),
        'assessments': clean_assessment_data(datasets['assessments'], datasets['student_assessments'])
    }
    
    # Log shapes
    for name, data in clean_data.items():
        logger.info(f"Clean {name} shape: {data.shape}")
    
    # Save cleaned data
    for name, data in clean_data.items():
        data.to_csv(os.path.join(output_paths['feature_dir'], f'clean_{name}.csv'), index=False)
    
    # Run exploratory data analysis if requested
    if visualize:
        try:
            # First, run automated EDA and save findings
            from data_analysis.eda import perform_automated_eda
            eda_findings = perform_automated_eda(
                clean_data,
                output_paths['viz_dir'],
                output_paths['report_dir']
            )
            
            # Then generate interactive visualizations
            viz_runner = VisualizationRunner(output_paths['viz_dir'])
            
            # Generate demographic visualizations
            demo_paths = viz_runner.run_demographic_visualizations(clean_data['demographics'])
            logger.info(f"Generated demographic visualizations: {demo_paths}")
            
            # Generate engagement visualizations with chunked data
            engage_paths = viz_runner.run_engagement_visualizations(
                clean_data['vle'],
                clean_data['demographics']
            )
            logger.info(f"Generated engagement visualizations: {engage_paths}")
            
            # Log EDA completion
            logger.info("Completed exploratory data analysis")
            
        except Exception as e:
            logger.warning(f"Error during EDA: {str(e)}. Continuing with processing...")
    
    return clean_data


def engineer_features(clean_data, output_paths):
    """Generate features for both static and sequential paths."""
    logger.info("Starting feature engineering")
    
    # Create demographic features first since they're smaller
    demographic_features = create_demographic_features(clean_data['demographics'])
    logger.info(f"Demographic features shape: {demographic_features.shape}")
    
    # Create temporal features with optimized window processing
    temporal_features = create_temporal_features(clean_data['vle'], WINDOW_SIZES)
    for window_size, features in temporal_features.items():
        logger.info(f"Window {window_size} features shape: {features.shape}")
    
    # Create assessment features
    assessment_features = create_assessment_features(clean_data['assessments'])
    logger.info(f"Assessment features shape: {assessment_features.shape}")
    
    # Process sequential features in smaller chunks with better memory management
    chunk_size = 25000  # Smaller chunks
    sequential_features_list = []
    sequential_generator = create_sequential_features(clean_data['vle'], chunk_size=chunk_size)
    
    try:
        total_rows = 0
        for i, batch in enumerate(sequential_generator):
            if len(batch.shape) == 3:
                batch = batch.reshape(batch.shape[1], batch.shape[2])
            
            # Convert to DataFrame and append
            df_batch = pd.DataFrame(batch)
            sequential_features_list.append(df_batch)
            total_rows += len(df_batch)
            
            # More frequent concatenation and cleanup
            if len(sequential_features_list) >= 5:  # Reduced from 10 to 5
                sequential_features = pd.concat(sequential_features_list, ignore_index=True)
                sequential_features_list = [sequential_features]
                optimize_memory()  # Force cleanup more frequently
            
            # More frequent progress updates
            if (i + 1) % 5 == 0:  # Reduced from 10 to 5
                logger.info(f"Processed {total_rows:,} sequential feature rows...")
            
            # Periodically concat and clear list to manage memory
            if len(sequential_features_list) >= 10:
                sequential_features = pd.concat(sequential_features_list, ignore_index=True)
                sequential_features_list = [sequential_features]
                gc.collect()  # Force garbage collection
        
        # Final concatenation
        sequential_features = pd.concat(sequential_features_list, ignore_index=True)
        logger.info(f"Sequential features shape: {sequential_features.shape}")
        
    except Exception as e:
        logger.error(f"Error processing sequential features: {str(e)}")
        sequential_features = None
        raise
    
    if sequential_features is None:
        logger.warning("Sequential features is None or empty")
    
    # prepare target variable
    logger.info("Preparing target variable")
    target = prepare_target_variable(clean_data['demographics'])
    logger.info(f"Target variable shape: {target.shape}")

    # temporal features are aligned with demographic data
    temporal_window = temporal_features[f'window_{WINDOW_SIZES[0]}']

    # unique student-module combinations
    student_module = demographic_features[['id_student', 'code_module', 'code_presentation']]
    
    # Aggregate temporal features by student-module
    temporal_agg = temporal_window.groupby(
        ['id_student', 'code_module', 'code_presentation']
    ).agg({
        col: ['mean', 'std', 'max'] for col in temporal_window.select_dtypes(include=['int64', 'float64']).columns
        if col not in ['id_student', 'code_module', 'code_presentation']
    }).reset_index()

    # index columns are preserved after groupby
    temporal_agg.columns = [
        f"{col[0]}" if col[1] == '' else f"{col[0]}_{col[1]}"
        for col in temporal_agg.columns
    ]
    
    # Merge with demographic features
    X = pd.merge(
        demographic_features,
        temporal_agg,
        on=['id_student', 'code_module', 'code_presentation'],
        how='left',
        validate='1:1'
    )
    y = target
    
    # calculate feature correlations and importance
    logger.info("Analyzing feature correlations and importance")
    correlations = analyze_feature_correlations(
        X, 
        threshold=CORRELATION_THRESHOLD,
        output_dir=output_paths['viz_dir'],
        plot=True
    )
    importance_df = analyze_feature_importance(X, y, plot=False)

    # prepare dual path features
    logger.info("Preparing dual path features")
    dual_path_features = prepare_dual_path_features(
        demographic_features, 
        temporal_features,
        assessment_features,
        sequential_features
    )
    
    # log dual path feature shapes
    for path_name, features in dual_path_features.items():
        logger.info(f"{path_name} shape: {features.shape}")
    
    # save feature data with correct paths
    demographic_features.to_csv(os.path.join(output_paths['feature_dir'], 'demographic_features.csv'), index=False)
    sequential_features.to_csv(os.path.join(output_paths['feature_dir'], 'sequential_features.csv'), index=False)
    
    # Save temporal features
    for window_size, features in temporal_features.items():
        features.to_csv(os.path.join(output_paths['feature_dir'], f'{window_size}_features.csv'), index=False)
    
    # Save dual path features with correct paths
    dual_path_features['static_path'].to_csv(
        os.path.join(output_paths['feature_dir'], 'static_path_features.csv'), 
        index=False
    )
    dual_path_features['sequential_path'].to_csv(
        os.path.join(output_paths['feature_dir'], 'sequential_path_features.csv'), 
        index=False
    )
    
    # Save target variable with proper index alignment
    target_df = pd.DataFrame({
        'id_student': clean_data['demographics']['id_student'],
        'target': target
    })
    target_df.to_csv(
        os.path.join(output_paths['feature_dir'], 'target.csv'), 
        index=False  # Changed from index=True
    )
    
    logger.info(f"Saved all features to {output_paths['feature_dir']}")
    
    return {
        'demographic_features': demographic_features,
        'temporal_features': temporal_features,
        'assessment_features': assessment_features,
        'sequential_features': sequential_features,
        'dual_path_features': dual_path_features,
        'target': target
    }

def train_random_forest(feature_data, output_paths, args):
    """Trains and evaluates the random forest model."""
    logger.info("Training Random Forest model...")
    
    # Get static features from dual path features
    X = feature_data['dual_path_features']['static_path']
    y = feature_data['target']
    
    # Select only numeric and encoded features
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    encoded_cols = [col for col in X.columns if col.endswith('_encoded') or col.startswith(('gender_', 'age_band_', 'imd_band_'))]
    feature_cols = list(set(numeric_cols) | set(encoded_cols))
    
    logger.info(f"Using {len(feature_cols)} features for training")
    logger.info(f"Numeric features: {len(numeric_cols)}, Encoded features: {len(encoded_cols)}")
    
    # Create demographically-aware stratified splits
    split_data = create_stratified_splits(
        dual_path_features=feature_data['dual_path_features'],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    
    # Extract train/test sets using selected features
    X_train = split_data['static_train'][feature_cols]
    X_test = split_data['static_test'][feature_cols]
    y_train = y[X_train.index]
    y_test = y[X_test.index]

    # bias mitigation with original categorical columns for stratification
    if BIAS_MITIGATION['method'] != 'none':
        protected_attrs = {
            attr: split_data['static_train'][attr_config['name']] 
            for attr, attr_config in PROTECTED_ATTRIBUTES.items()
        }
        logger.info("Resampling training data for bias mitigation")
        X_train, y_train, sample_weights = resample_training_data(
            X_train,
            y_train, 
            protected_attrs,
            method=BIAS_MITIGATION['method'],
            random_state=RANDOM_SEED
        )
    else:
        sample_weights = None
        protected_attrs = None

    # Tune hyperparameters if in full mode
    if args.mode == 'full':
        logger.info("Tuning hyperparameters for Random Forest model")
        best_params, _ = tune_random_forest(
            X_train, 
            y_train,
            param_grid=RF_PARAM_GRID,
            n_splits=5,  # Changed from cv=5
            scoring='f1',
            n_jobs=-1,
            random_search=True,
            n_iter=20,
            verbose=args.verbose
        )
    else:
        best_params = RF_DEFAULT_PARAMS
    
    # Initialize and train model with best parameters
    logger.info(f"Best parameters for Random Forest: {best_params}")
    rf_model = RandomForestModel(**best_params)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = rf_model.evaluate(X_test, y_test)
    
    # Save model
    model_path = os.path.join(output_paths['model_dir'], 'random_forest.pkl')
    rf_model.save_model(model_path)
    logger.info(f"Saved Random Forest model to {model_path}")
    
    return {
        'model': rf_model,
        'metrics': metrics,
        'split_data': {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    }

def train_gru_model(feature_data, split_data, output_paths, args):
    """Trains and evaluates the GRU model."""
    logger.info("Training GRU model...")
    
    # Prepare sequential features and target variable
    logger.info("Preparing GRU training data")
    gru_data = prepare_gru_training_data(
        feature_data['sequential_features'],
        feature_data['demographic_features'],
        split_data['X_train'].index,
        split_data['X_test'].index
    )
    
    # Verify data alignment before splitting
    n_samples = len(gru_data['X_train']['categorical'])
    logger.info(f"Number of training samples: {n_samples}")
    logger.info(f"Target shape: {gru_data['y_train'].shape}")
    
    if n_samples != len(gru_data['y_train']):
        raise ValueError(
            f"Mismatched lengths: X samples={n_samples}, y samples={len(gru_data['y_train'])}"
        )
    
    # Split training data into train/validation using VALIDATION_SIZE
    logger.info("Creating train/validation split for GRU model")
    indices = np.arange(n_samples)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=gru_data['y_train']
    )
    
    # Create training and validation sets ensuring all components are aligned
    X_train = {
        'categorical': gru_data['X_train']['categorical'][train_indices],
        'numerical': gru_data['X_train']['numerical'][train_indices],
        'student_index_map': {
            student: idx for idx, student in enumerate(
                np.array(list(gru_data['X_train']['student_index_map'].keys()))[train_indices]
            )
        }
    }
    
    X_val = {
        'categorical': gru_data['X_train']['categorical'][val_indices],
        'numerical': gru_data['X_train']['numerical'][val_indices],
        'student_index_map': {
            student: idx for idx, student in enumerate(
                np.array(list(gru_data['X_train']['student_index_map'].keys()))[val_indices]
            )
        }
    }
    
    y_train = gru_data['y_train'][train_indices]
    y_val = gru_data['y_train'][val_indices]
    
    # Bias mitigation if enabled
    if BIAS_MITIGATION['method'] != 'none':
        protected_attrs = {
            attr: feature_data['demographic_features'][attr_config['name']]
            for attr, attr_config in PROTECTED_ATTRIBUTES.items()
        }
        # Resample training data
        logger.info("Resampling GRU training data for bias mitigation")
        for key in X_train.keys():
            if key != 'student_index_map':  # Skip metadata
                reshaped_data = X_train[key].reshape(len(X_train[key]), -1)
                resampled_data, resampled_y, sample_weights = resample_training_data(  # Store sample_weights
                    reshaped_data,
                    y_train,
                    protected_attrs,
                    method=BIAS_MITIGATION['method'],
                    random_state=RANDOM_SEED
                )
                # Use sample weights if available
                if sample_weights is not None:
                    class_weights = dict(enumerate(sample_weights))
                X_train[key] = resampled_data.reshape(X_train[key].shape)
                if 'y_resampled' not in locals():
                    y_resampled = resampled_y
        y_train = y_resampled
    
    # Tune hyperparameters if in full mode
    logger.info("Tuning hyperparameters for GRU model")
    if args.mode == 'full':
        best_params, _ = tune_gru_hyperparameters(
            X_train, y_train,
            X_val, y_val,
            protected_attributes=protected_attrs,
            param_grid=GRU_PARAM_GRID,
            bias_mitigation=BIAS_MITIGATION,
            epochs=20,
            batch_size=32,
            verbose=args.verbose
        )
    else:
        best_params = GRU_DEFAULT_PARAMS
    
    # Initialize and train model with best parameters
    gru_model = GRUModel(**best_params)
    gru_model.fit(
        gru_data['X_train'],
        gru_data['y_train'],
        X_val=gru_data['X_test'],
        y_val=gru_data['y_test']
    )
    
    # Evaluate model
    metrics = gru_model.evaluate(gru_data['X_test'], gru_data['y_test'])
    
    # Save model and preprocessor
    model_path = os.path.join(output_paths['model_dir'], 'gru_model.keras')
    gru_model.save_model(model_path, preprocessor=gru_data['preprocessor'])
    logger.info(f"Saved GRU model to {model_path}")

    return {
        'model': gru_model,
        'metrics': metrics,
        'preprocessor': gru_data['preprocessor']
    }

def train_ensemble_model(rf_results, gru_results, split_data, output_paths, args):
    """Trains and evaluates the ensemble model."""
    logger.info("Training Ensemble model...")
    
    # Initialize ensemble model
    ensemble = EnsembleModel()
    ensemble.set_models(rf_results['model'], gru_results['model'])
    
    # Get predictions from both models for optimization
    rf_probs = rf_results['model'].predict_proba(split_data['X_test'])
    gru_probs = gru_results['model'].predict_proba(gru_results['preprocessor'].transform_sequences(
        split_data['X_test']
    ))
    
    # Optimize ensemble weights
    logger.info("Optimizing ensemble weights")
    best_weights, _ = optimize_ensemble_weights(
        rf_probs=rf_probs,
        gru_probs=gru_probs,
        y_true=split_data['y_test'],
        metric='f1'
    )
    
    # Set optimized weights
    ensemble = EnsembleModel(
        static_weight=best_weights['rf_weight'],
        sequential_weight=best_weights['gru_weight'],
        threshold=best_weights['threshold']
    )
    ensemble.set_models(rf_results['model'], gru_results['model'])
    
    # Evaluate ensemble
    logger.info("Evaluating Ensemble model")
    metrics = ensemble.evaluate(
        split_data['X_test'],
        gru_results['preprocessor'].transform_sequences(split_data['X_test']),
        split_data['y_test'],
        gru_results['preprocessor'].student_index_map
    )
    
    # Save ensemble model
    model_path = os.path.join(output_paths['model_dir'], 'ensemble_model.pkl')
    ensemble.save_model(model_path)
    logger.info(f"Saved Ensemble model to {model_path}")
    
    return {
        'model': ensemble,
        'metrics': metrics,
        'weights': best_weights
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

def prepare_visualizations(model_results, clean_data, fairness_results, output_paths, args):
    """Prepares and generates all visualizations using VisualizationRunner."""
    logger.info("Preparing visualizations...")
    
    # Initialize visualization runner
    viz_runner = VisualizationRunner(output_paths['viz_dir'])
    visualization_paths = []
    
    try:
        # Generate demographic visualizations
        if 'demographics' in clean_data:
            demo_paths = viz_runner.run_demographic_visualizations(clean_data['demographics'])
            visualization_paths.extend(demo_paths)
            logger.info(f"Generated {len(demo_paths)} demographic visualizations")
        
        # Generate engagement visualizations
        if all(k in clean_data for k in ['vle', 'demographics']):
            engage_paths = viz_runner.run_engagement_visualizations(
                clean_data['vle'],
                clean_data['demographics']
            )
            visualization_paths.extend(engage_paths)
            logger.info(f"Generated {len(engage_paths)} engagement visualizations")
        
        # Generate model performance visualizations
        for model_name, results in model_results.items():
            if 'metrics' in results:
                perf_paths = viz_runner.visualize_model_performance(
                    results['metrics'],
                    model_name.upper()
                )
                visualization_paths.extend(perf_paths)
                logger.info(f"Generated performance visualizations for {model_name}")
        
        # Generate feature importance visualization for RF model
        if 'rf' in model_results and 'model' in model_results['rf']:
            rf_model = model_results['rf']['model']
            importance_df = rf_model.get_feature_importance(plot=False)
            importance_path = viz_runner.visualize_feature_importance(importance_df)
            visualization_paths.append(importance_path)
            logger.info("Generated feature importance visualization")
        
        # Export data for Tableau if requested
        if args.export_tableau:
            tableau_data = {
                'student_risk': prepare_student_risk_data(model_results, clean_data),
                'temporal_engagement': prepare_temporal_engagement_data(clean_data['vle']),
                'assessment_performance': prepare_assessment_performance_data(clean_data['assessments']),
                'demographic_fairness': prepare_demographic_fairness_data(fairness_results),
                'model_performance': prepare_model_performance_data(model_results)
            }
            
            export_paths = export_for_tableau(
                tableau_data,
                output_paths['tableau_dir']
            )
            visualization_paths.extend(export_paths)
            
            # Create Tableau instructions
            instructions_path = create_tableau_instructions(
                export_paths,
                output_paths['tableau_dir']
            )
            visualization_paths.append(instructions_path)
            
            logger.info(f"Exported {len(export_paths)} datasets for Tableau visualization")
        
        return visualization_paths
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)
        raise

def evaluate_and_report(model_results, feature_data, clean_data, output_paths, args):
    """Handles evaluation, fairness analysis, and reporting."""
    
    # Use demographic features from feature_data if clean_data is None
    demographic_data = (clean_data['demographics'] if clean_data is not None 
                       else feature_data['demographic_features'])
    
    # Perform fairness analysis
    fairness_results = evaluate_model_fairness(
        model_results,
        demographic_data,
        output_paths,
        args
    )
    
    # Generate visualizations if requested
    if args.visualize or args.export_tableau:
        # If clean_data is None, create minimal version from feature_data
        viz_data = clean_data if clean_data is not None else {
            'demographics': feature_data['demographic_features'],
            'vle': feature_data['sequential_features']
        }
        
        visualization_paths = prepare_visualizations(
            model_results,
            viz_data,
            fairness_results,
            output_paths,
            args
        )
        logger.info(f"Generated {len(visualization_paths)} visualizations")
    
    return fairness_results

def load_preprocessed_features(feature_dir):
    """Loads preprocessed features from disk for visualization-only mode."""
    logger.info("Loading preprocessed features from disk...")
    
    feature_data = {}
    required_files = [
        ('demographic_features.csv', 'demographic_features'),
        ('sequential_features.csv', 'sequential_features'),
        ('static_path_features.csv', 'static_path'),
        ('target.csv', 'target')
    ]
    
    try:
        # Verify all required files exist
        missing_files = []
        for filename, _ in required_files:
            if not os.path.exists(os.path.join(feature_dir, filename)):
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
        
        # Load demographic features
        feature_data['demographic_features'] = pd.read_csv(
            os.path.join(feature_dir, 'demographic_features.csv')
        )
        logger.info(f"Loaded demographic features: {feature_data['demographic_features'].shape}")
        
        # Load sequential features
        feature_data['sequential_features'] = pd.read_csv(
            os.path.join(feature_dir, 'sequential_features.csv')
        )
        logger.info(f"Loaded sequential features: {feature_data['sequential_features'].shape}")
        
        # Load temporal features for different windows
        temporal_features = {}
        for window_size in [7, 14, 30]:
            window_file = os.path.join(feature_dir, f'window_{window_size}_features.csv')
            if os.path.exists(window_file):
                temporal_features[f'window_{window_size}'] = pd.read_csv(window_file)
                logger.info(f"Loaded window {window_size} features: {temporal_features[f'window_{window_size}'].shape}")
        feature_data['temporal_features'] = temporal_features
        
        # Load dual path features
        dual_path_features = {}
        
        # Load static path features
        static_file = os.path.join(feature_dir, 'static_path_features.csv')
        dual_path_features['static_path'] = pd.read_csv(static_file)
        logger.info(f"Loaded static path features: {dual_path_features['static_path'].shape}")
            
        # Load sequential path features
        sequential_file = os.path.join(feature_dir, 'sequential_path_features.csv')
        if os.path.exists(sequential_file):
            dual_path_features['sequential_path'] = pd.read_csv(sequential_file)
            logger.info(f"Loaded sequential path features: {dual_path_features['sequential_path'].shape}")
        
        feature_data['dual_path_features'] = dual_path_features
        
        # Load target variable
        target_file = os.path.join(feature_dir, 'target.csv')
        target_df = pd.read_csv(target_file)
        if 'target' in target_df.columns:
            feature_data['target'] = target_df['target'].values
            logger.info(f"Loaded target variable: {feature_data['target'].shape}")
        else:
            raise KeyError("Target column not found in target.csv")
        
        logger.info("Successfully loaded all preprocessed features")
        logger.info(f"Available features: {list(feature_data.keys())}")
        
        return feature_data
        
    except Exception as e:
        logger.error(f"Error loading preprocessed features: {str(e)}")
        raise RuntimeError(f"Failed to load preprocessed features: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error loading preprocessed features: {str(e)}", exc_info=True)
        raise RuntimeError("Failed to load preprocessed features. Please run feature engineering first.")

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
        # Initialize progress tracker with dynamic steps
        total_steps = 6
        progress = ProgressTracker(total_steps=total_steps, desc="Processing")

        if args.skip_preprocessing:
            # Check if preprocessed features exist
            feature_dir = output_paths['feature_dir']
            if not os.path.exists(feature_dir) or not os.listdir(feature_dir):
                logger.error("No preprocessed features found. Please run full pipeline first.")
                return 1
                
            logger.info("Loading saved features...")
            clean_data = None
            feature_data = load_preprocessed_features(output_paths['feature_dir'])
            progress.update("Loading Saved Features")

        # Step 1-2: Data Loading and Feature Engineering
        if not args.skip_preprocessing:
            # Load and process data
            clean_data = load_and_process_data(args.data_path, output_paths, args.visualize)
            progress.update("Data Loading")
            
            # Feature engineering
            feature_data = engineer_features(clean_data, output_paths)
            progress.update("Feature Engineering")
        else:
            logger.info("Skipping data preprocessing, loading saved features...")
            clean_data = None
            feature_data = load_preprocessed_features(output_paths['feature_dir'])
            progress.update("Loading Saved Features")
        
        # Step 3: Train models
        if args.mode in ['rf_only', 'gru_only']:
            if args.mode == 'rf_only':
                model_results = {'rf': train_random_forest(feature_data, output_paths, args)}
            else:
                model_results = {'gru': train_gru_model(
                    feature_data,
                    {'X_train': feature_data['dual_path_features']['static_path']},  # Minimal split_data
                    output_paths,
                    args
                )}
            progress.update("Model Training")
        elif args.mode not in ['data_only', 'visualization_only']:
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