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
    RandomForestModel,
    find_optimal_threshold
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
    calculate_group_metrics,
    calculate_fairness_metrics,
    evaluate_model_fairness,
    visualize_fairness_metrics,
    compare_group_performance,
    generate_fairness_report,
    analyze_subgroup_fairness,
    mitigate_bias_with_thresholds
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

from visualization.visualization_runner import VisualizationRunner


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


def train_random_forest(feature_data, output_paths, args):
    """Train and optimize random forest model for static path."""
    logger.info("Training Random Forest model")
    
    # create splits
    split_data = create_stratified_splits(feature_data['dual_path_features'], test_size=0.2, random_state=42)
    
    # prepare data for static path modeling
    X_train_static = split_data['static_train'].drop(['final_result', 'id_student', 'code_module', 'code_presentation'], 
                                                  axis=1, errors='ignore')
    X_test_static = split_data['static_test'].drop(['final_result', 'id_student', 'code_module', 'code_presentation'], 
                                                axis=1, errors='ignore')
    
    # prepare target variables
    y_train = prepare_target_variable(split_data['static_train'])
    y_test = prepare_target_variable(split_data['static_test'])
    
    logger.info(f"X_train shape: {X_train_static.shape}")
    logger.info(f"X_test shape: {X_test_static.shape}")
    logger.info(f"Target distribution: {y_train.value_counts().to_dict()}")
    
    # analyze feature importance using baseline model
    feature_importance = analyze_feature_importance(X_train_static, y_train)
    
    # identify highly correlated features
    correlated_features = analyze_feature_correlations(X_train_static, threshold=args.correlation_threshold)
    
    if len(correlated_features) > 0:
        logger.info(f"Found {len(correlated_features)} highly correlated feature pairs")
        
        # remove one feature from each correlated pair
        to_drop = set()
        for _, row in correlated_features.iterrows():
            # keep the one with higher importance if possible
            if row['Feature1'] in feature_importance['Feature'].values and row['Feature2'] in feature_importance['Feature'].values:
                f1_importance = feature_importance[feature_importance['Feature'] == row['Feature1']]['Importance'].iloc[0]
                f2_importance = feature_importance[feature_importance['Feature'] == row['Feature2']]['Importance'].iloc[0]
                
                if f1_importance >= f2_importance:
                    to_drop.add(row['Feature2'])
                else:
                    to_drop.add(row['Feature1'])
            else:
                # otherwise drop the second one arbitrarily
                to_drop.add(row['Feature2'])
        
        logger.info(f"Removing {len(to_drop)} features due to multicollinearity")
        X_train_static = X_train_static.drop(to_drop, axis=1, errors='ignore')
        X_test_static = X_test_static.drop(to_drop, axis=1, errors='ignore')
    
    # feature selection based on importance
    important_features = feature_importance[feature_importance['Importance'] > args.importance_threshold]['Feature'].tolist()
    logger.info(f"Selected {len(important_features)} features with importance > {args.importance_threshold}")
    
    # filter to important features
    X_train_selected = X_train_static[important_features]
    X_test_selected = X_test_static[important_features]
    
    logger.info(f"X_train_selected shape: {X_train_selected.shape}")
    logger.info(f"X_test_selected shape: {X_test_selected.shape}")
    
    # save selected features list
    pd.Series(important_features).to_csv(os.path.join(output_paths['feature_dir'], 'important_features.csv'), index=False)
    
    # define parameter grid for random forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # tune random forest hyperparameters
    best_params, best_model = tune_random_forest(
        X_train_selected, 
        y_train,
        param_grid=param_grid,
        scoring='f1',
        random_search=True,
        n_iter=20,
        verbose=1 if args.verbose else 0
    )
    
    # save tuning results
    save_tuning_results(
        pd.DataFrame([best_params]), 
        best_params, 
        os.path.join(output_paths['model_dir'], 'rf_tuning_results.pkl'),
        model_name='random_forest'
    )
    
    # create and train optimized model
    rf_model = RandomForestModel(**best_params, random_state=42)
    rf_model.fit(X_train_selected, y_train)
    
    # find optimal threshold
    opt_threshold = find_optimal_threshold(rf_model, X_test_selected, y_test)
    logger.info(f"Optimal threshold: {opt_threshold:.4f}")
    
    # evaluate model with optimal threshold
    rf_metrics = rf_model.evaluate(X_test_selected, y_test, threshold=opt_threshold)
    
    # save evaluation metrics
    with open(os.path.join(output_paths['report_dir'], 'rf_metrics.json'), 'w') as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in rf_metrics.items()}, f, indent=2)
    
    # save model
    rf_model.save_model(os.path.join(output_paths['model_dir'], 'random_forest_model.pkl'))
    logger.info("Random Forest model training complete")
    
    return {
        'model': rf_model,
        'best_params': best_params,
        'metrics': rf_metrics,
        'threshold': opt_threshold,
        'X_train': X_train_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_test': y_test,
        'split_data': split_data,
        'important_features': important_features
    }


def train_gru_model(feature_data, split_data, output_paths, args):
    """Train and optimize GRU model for sequential path."""
    logger.info("Training GRU model")
    
    # set TensorFlow to use GPU if available and requested
    if args.gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Using GPU: {len(gpus)} device(s) available")
            except RuntimeError as e:
                logger.error(f"Error setting up GPU: {e}")
        else:
            logger.warning("No GPU available. Using CPU for training.")
    else:
        # disable GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("Using CPU for training (GPU disabled)")
    
    # prepare data for GRU model
    gru_data = prepare_gru_training_data(
        feature_data['sequential_features'],
        split_data['static_train'],
        split_data['train_ids'],
        split_data['test_ids']
    )
    
    X_train_gru = gru_data['X_train']
    y_train_gru = gru_data['y_train']
    X_test_gru = gru_data['X_test']
    y_test_gru = gru_data['y_test']
    preprocessor = gru_data['preprocessor']
    
    logger.info(f"GRU training data prepared: {len(y_train_gru)} train samples, {len(y_test_gru)} test samples")
    
    # check if we should use hyperparameter tuning or default values
    if args.mode in ['full', 'gru_only']:
        # define parameter grid for GRU
        param_grid = {
            'gru_units': [32, 64, 128],
            'dense_units': [[32], [64], [32, 16]],
            'dropout_rate': [0.2, 0.3, 0.5],
            'learning_rate': [0.001, 0.0005]
        }
        
        # tune GRU hyperparameters
        try:
            best_params, tuning_results = tune_gru_hyperparameters(
                X_train_gru, 
                y_train_gru,
                X_test_gru,
                y_test_gru,
                param_grid=param_grid,
                epochs=15,
                batch_size=32,
                early_stopping_patience=3,
                verbose=1 if args.verbose else 0
            )
            
            # save tuning results
            tuning_results.to_csv(os.path.join(output_paths['report_dir'], 'gru_tuning_results.csv'), index=False)
            
            with open(os.path.join(output_paths['report_dir'], 'gru_best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=2)
                
            logger.info(f"GRU hyperparameter tuning complete. Best params: {best_params}")
        except Exception as e:
            logger.error(f"Error during GRU hyperparameter tuning: {str(e)}")
            logger.info("Using default GRU parameters")
            
            # default parameters if tuning fails
            best_params = {
                'gru_units': 64,
                'dense_units': [32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            }
    else:
        # use default parameters in other modes
        best_params = {
            'gru_units': 64,
            'dense_units': [32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }
    
    # create and train optimized GRU model
    categorical_dim = X_train_gru['categorical'].shape[2] if 'categorical' in X_train_gru and X_train_gru['categorical'] is not None else None
    numerical_dim = X_train_gru['numerical'].shape[2] if 'numerical' in X_train_gru and X_train_gru['numerical'] is not None else None
    
    gru_model = GRUModel(
        gru_units=best_params['gru_units'],
        dense_units=best_params['dense_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        max_seq_length=preprocessor.max_seq_length,
        categorical_dim=categorical_dim,
        numerical_dim=numerical_dim
    )
    
    # build model
    gru_model.build_model()
    
    # calculate class weights to handle imbalance
    class_counts = np.bincount(y_train_gru)
    if len(class_counts) > 1 and class_counts[0] != class_counts[1]:
        class_weight = {
            0: len(y_train_gru) / (2 * class_counts[0]),
            1: len(y_train_gru) / (2 * class_counts[1])
        }
        logger.info(f"Using class weights: {class_weight}")
    else:
        class_weight = None
    
    # set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_paths['model_dir'], 'gru_checkpoint.keras'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # train model
    try:
        gru_model.fit(
            X_train_gru,
            y_train_gru,
            X_val=X_test_gru,
            y_val=y_test_gru,
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            class_weights=class_weight
        )
        
        # plot training history
        if args.visualize:
            gru_model.plot_training_history()
            plt.savefig(os.path.join(output_paths['viz_dir'], 'gru_training_history.png'))
            
        # evaluate model
        gru_metrics = gru_model.evaluate(X_test_gru, y_test_gru)
        
        # save evaluation metrics
        with open(os.path.join(output_paths['report_dir'], 'gru_metrics.json'), 'w') as f:
            json.dump(gru_metrics, f, indent=2)
        
        # save model and preprocessor
        gru_model.save_model(
            os.path.join(output_paths['model_dir'], 'gru_model.keras'),
            preprocessor
        )
        
        logger.info("GRU model training complete")
        
    except Exception as e:
        logger.error(f"Error during GRU model training: {str(e)}")
        raise
    
    return {
        'model': gru_model,
        'preprocessor': preprocessor,
        'best_params': best_params,
        'metrics': gru_metrics,
        'X_train': X_train_gru,
        'X_test': X_test_gru,
        'y_train': y_train_gru,
        'y_test': y_test_gru
    }


def train_ensemble_model(rf_data, gru_data, split_data, output_paths, args):
    """Train and optimize ensemble model combining RF and GRU."""
    logger.info("Training Ensemble model")
    
    # extract models
    rf_model = rf_data['model']
    gru_model = gru_data['model']
    
    # extract test data
    X_test_rf = rf_data['X_test']
    X_test_gru = gru_data['X_test']
    y_test = rf_data['y_test']
    
    # get student ID mapping from test set
    test_student_ids = split_data['static_test']['id_student'].values
    student_id_map = {sid: idx for idx, sid in enumerate(X_test_gru['students'])}
    
    # create ensemble model
    ensemble = EnsembleModel()
    ensemble.set_models(rf_model, gru_model)
    
    # optimize ensemble weights
    logger.info("Optimizing ensemble weights")
    ensemble.optimize_weights(
        X_test_rf,
        X_test_gru,
        y_test.values,
        student_id_map,
        metric='f1',
        weight_grid=21
    )
    
    # visualize ensemble optimization if requested
    if args.visualize:
        # get predictions from both models
        rf_probs = rf_model.predict_proba(X_test_rf)
        gru_probs = gru_model.predict_proba(X_test_gru)
        
        # create results dataframe for visualization
        results_df = []
        
        for rf_weight in np.linspace(0, 1, 21):
            for threshold in np.linspace(0, 1, 21):
                gru_weight = 1 - rf_weight
                ensemble_probs = rf_weight * rf_probs + gru_weight * gru_probs
                ensemble_preds = (ensemble_probs >= threshold).astype(int)
                
                from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
                
                results_df.append({
                    'rf_weight': rf_weight,
                    'gru_weight': gru_weight,
                    'threshold': threshold,
                    'f1': f1_score(y_test, ensemble_preds),
                    'accuracy': accuracy_score(y_test, ensemble_preds),
                    'auc': roc_auc_score(y_test, ensemble_probs)
                })
        
        results_df = pd.DataFrame(results_df)
        
        # visualize ensemble weights
        visualize_ensemble_weights(results_df, metric='f1', 
                                 save_path=os.path.join(output_paths['viz_dir'], 'ensemble_optimization.png'))
    
    # evaluate ensemble model
    ensemble_metrics = ensemble.evaluate(
        X_test_rf,
        X_test_gru,
        y_test.values,
        student_id_map
    )
    
    # save ensemble metrics
    with open(os.path.join(output_paths['report_dir'], 'ensemble_metrics.json'), 'w') as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ensemble_metrics.items()}, f, indent=2)
    
    # save ensemble model
    ensemble.save_model(os.path.join(output_paths['model_dir'], 'ensemble_model.pkl'))
    
    logger.info("Ensemble model training complete")
    
    return {
        'model': ensemble,
        'metrics': ensemble_metrics,
        'student_id_map': student_id_map
    }


def evaluate_model_fairness(model_data, demographic_data, output_paths, args):
    """Evaluates model fairness across demographic groups."""
    logger.info("Evaluating model fairness")
    
    # extract test data and model predictions
    if 'ensemble' in model_data:
        model = model_data['ensemble']['model']
        X_test_rf = model_data['rf']['X_test']
        X_test_gru = model_data['gru']['X_test']
        y_test = model_data['rf']['y_test'].values
        student_id_map = model_data['ensemble']['student_id_map']
        
        # get ensemble predictions
        y_pred = model.predict(X_test_rf, X_test_gru, student_id_map)
        y_prob = model.predict_proba(X_test_rf, X_test_gru, student_id_map)
    else:
        # use random forest model predictions
        model = model_data['rf']['model']
        X_test = model_data['rf']['X_test']
        y_test = model_data['rf']['y_test'].values
        threshold = model_data['rf']['threshold']
        
        # get predictions
        y_pred = model.predict(X_test, threshold=threshold)
        y_prob = model.predict_proba(X_test)
    
    # prepare protected attributes
    protected_attributes = {}
    for attr in ['gender', 'age_band', 'imd_band']:
        if attr in demographic_data.columns:
            protected_attributes[attr] = demographic_data[attr].values

    # set fairness thresholds
    fairness_thresholds = {
        'demographic_parity_difference': args.fairness_threshold,
        'disparate_impact_ratio': 0.8,
        'equal_opportunity_difference': args.fairness_threshold
    }

    # evaluate fairness
    fairness_results = evaluate_model_fairness(
        y_test,
        y_pred,
        y_prob,
        protected_attributes,
        fairness_thresholds
    )

    # generate fairness report
    fairness_report = generate_fairness_report(
        fairness_results,
        fairness_thresholds,
        save_path=os.path.join(output_paths['report_dir'], 'fairness_report.md')
    )

    # visualize results
    if args.visualize:
        viz_runner = VisualizationRunner(output_paths['viz_dir'])
        fairness_viz_paths = viz_runner.run_fairness_visualizations(
            fairness_results,
            demographic_data
        )
        logger.info(f"Generated fairness visualizations: {fairness_viz_paths}")
        
        # compare group performance for each protected attribute
        for attr in protected_attributes:
            compare_group_performance(
                fairness_results,
                metric='f1',
                save_path=os.path.join(output_paths['viz_dir'], f'fairness_{attr}_f1.png')
            )

    return {
        'fairness_results': fairness_results,
        'fairness_report': fairness_report,
        'mitigated_thresholds': None  # Will be updated if bias mitigation is needed
    }


def prepare_visualizations(model_data, clean_data, fairness_data, output_paths, args):
    """Prepare visualizations and export data for Tableau."""
    logger.info("Preparing visualizations and Tableau exports")
    
    if 'ensemble' in model_data:
        # use ensemble model predictions
        model = model_data['ensemble']['model']
        X_test_rf = model_data['rf']['X_test']
        X_test_gru = model_data['gru']['X_test']
        student_id_map = model_data['ensemble']['student_id_map']
        
        # get student IDs from the test set
        test_student_ids = model_data['rf']['split_data']['static_test']['id_student'].values
        
        # get ensemble predictions
        y_pred = model.predict(X_test_rf, X_test_gru, student_id_map)
        y_prob = model.predict_proba(X_test_rf, X_test_gru, student_id_map)
    else:
        # use random forest model predictions
        model = model_data['rf']['model']
        X_test = model_data['rf']['X_test']
        
        # get predictions using optimized thresholds from fairness analysis
        y_prob = model.predict_proba(X_test)
        
        # use fairness-aware thresholds if protected attributes are available
        if 'protected_attributes' in model_data:
            thresholds = mitigate_bias_with_thresholds(
                model_data['rf']['y_test'],
                y_prob,
                model_data['protected_attributes'],
                metric='demographic_parity',
                tolerance=args.fairness_threshold
            )
            # Apply group-specific thresholds
            y_pred = np.zeros_like(y_prob)
            for group, threshold in thresholds.items():
                group_mask = (model_data['protected_attributes'] == group)
                y_pred[group_mask] = (y_prob[group_mask] >= threshold).astype(int)
        else:
            # Use single threshold if no protected attributes
            y_pred = (y_prob >= model_data['rf']['threshold']).astype(int)
        
        # get student IDs from the test set
        test_student_ids = model_data['rf']['split_data']['static_test']['id_student'].values
    
    # prepare student demographic data
    student_info = clean_data['demographics']
    
    # prepare risk data for Tableau
    risk_data = prepare_student_risk_data(
        student_info,
        y_pred,
        y_prob,
        test_student_ids,
        demographic_cols=['gender', 'age_band', 'imd_band', 'region', 'highest_education']
    )
    
    # prepare temporal engagement data
    temporal_data = prepare_temporal_engagement_data(
        clean_data['vle'],
        risk_data,
        time_window=7
    )
    
    # prepare assessment performance data
    assessment_data = prepare_assessment_performance_data(
        clean_data['assessments'],
        student_info,
        risk_data
    )
    
    # prepare demographic fairness data
    fairness_viz_data = prepare_demographic_fairness_data(
        fairness_data['fairness_results'],
        risk_data,
        demographic_cols=['gender', 'age_band', 'imd_band']
    )
    
    # prepare model performance data
    if 'ensemble' in model_data:
        model_perf_data = prepare_model_performance_data(
            model_data['ensemble']['metrics'],
            'Ensemble Model',
            risk_data
        )
    else:
        model_perf_data = prepare_model_performance_data(
            model_data['rf']['metrics'],
            'Random Forest Model',
            risk_data
        )
    
    # export data for Tableau
    export_data = {
        'risk_data': risk_data,
        'temporal_data': temporal_data,
        'assessment_data': assessment_data,
        'fairness_data': fairness_viz_data,
        'model_performance': model_perf_data
    }
    
    # export to CSV files
    tableau_paths = export_for_tableau(
        export_data,
        export_dir=os.path.join(output_paths['tableau_dir'], 'data'),
        format='csv'
    )
    
    # generate summary visualizations
    viz_paths = generate_summary_visualizations(
        risk_data,
        temporal_data,
        assessment_data,
        fairness_viz_data,
        export_dir=os.path.join(output_paths['tableau_dir'], 'visualizations')
    )
    
    # create tableau instructions
    instructions_path = create_tableau_instructions(
        tableau_paths,
        viz_paths,
        output_path=os.path.join(output_paths['tableau_dir'], 'instructions.md')
    )
    
    logger.info(f"Tableau data and visualizations exported to {output_paths['tableau_dir']}")
    
    return {
        'tableau_paths': tableau_paths,
        'visualization_paths': viz_paths,
        'instructions_path': instructions_path,
        'export_data': export_data
    }


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
        # step 1: load and process data (required for all modes)
        clean_data = load_and_process_data(args.data_path, output_paths, args.visualize)
        
        # step 2: feature engineering (required for all non-visualization modes)
        if args.mode != 'visualization_only':
            feature_data = engineer_features(clean_data, output_paths)
        else:
            # load pre-processed feature data if available
            feature_data = None
            # TODO: add loading of pre-processed data in visualization-only mode
        
        # initialize model_data dictionary
        model_data = {}
        
        # step 3: train models based on mode
        if args.mode in ['full', 'rf_only', 'ensemble_only']:
            # train random forest model
            model_data['rf'] = train_random_forest(feature_data, output_paths, args)
        
        if args.mode in ['full', 'gru_only', 'ensemble_only']:
            # train GRU model
            model_data['gru'] = train_gru_model(
                feature_data, 
                model_data.get('rf', {}).get('split_data', None), 
                output_paths, 
                args
            )
        
        if args.mode in ['full', 'ensemble_only']:
            # train ensemble model
            model_data['ensemble'] = train_ensemble_model(
                model_data['rf'], 
                model_data['gru'], 
                model_data['rf']['split_data'], 
                output_paths, 
                args
            )
        
        # step 4: evaluate model fairness
        if args.mode in ['full', 'rf_only', 'ensemble_only', 'evaluation_only']:
            # if in evaluation_only mode, load models
            if args.mode == 'evaluation_only':
                # TODO: add loading of models for evaluation-only mode
                pass
            
            # evaluate fairness
            fairness_data = evaluate_model_fairness(
                model_data, 
                clean_data['demographics'], 
                output_paths, 
                args
            )
        else:
            fairness_data = None
        
        # step 5: prepare visualizations and Tableau exports
        if args.mode in ['full', 'visualization_only'] or args.export_tableau:
            if args.mode == 'visualization_only' and not model_data:
                # TODO: add loading of models and fairness data for visualization-only mode
                pass
            
            viz_data = prepare_visualizations(
                model_data, 
                clean_data, 
                fairness_data, 
                output_paths, 
                args
            )
        
        logger.info(f"EduPredict completed successfully in {args.mode} mode")
        
    except Exception as e:
        logger.error(f"Error in EduPredict: {str(e)}", exc_info=True)
        raise
    
    return 0


if __name__ == "__main__":
    sys.exit(main())