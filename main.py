#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EduPredict 2.0: Enhanced Early Warning Academic Performance Prediction System

This main script implements a comprehensive pipeline for academic performance prediction,
incorporating fairness-aware machine learning and advanced visualization capabilities.
"""

import argparse
import logging
import json
import sys
from json import JSONEncoder
import tensorflow as tf
from typing import Dict, Tuple, Any, Optional, List
import pandas as pd
import numpy as np
import psutil
import time
import gc
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import os
import pickle
import datetime

# Data processing imports
from data_processing.data_loader import load_raw_datasets
from data_processing.data_cleaner import (
    clean_demographic_data,
    clean_vle_data, 
    clean_assessment_data,
    clean_registration_data,
    validate_data_consistency
)
from data_processing.data_splitter import (
    create_stratified_splits,
    perform_student_level_split,
    save_data_splits,
    validate_demographic_balance
)
from data_processing.data_monitor import detect_data_quality_issues, analyze_temporal_patterns

# Feature engineering imports
from feature_engineering.categorical_encoder import CategoricalEncoder
from feature_engineering.demographic_features import create_demographic_features
from feature_engineering.sequential_features import create_sequential_features
from feature_engineering.temporal_features import create_temporal_features
from feature_engineering.feature_selector import (
    analyze_feature_importance,
    analyze_feature_correlations,
    remove_correlated_features,
    select_features_by_importance,
    analyze_demographic_impact,
    export_feature_metadata
)

# Model imports
from model_implementation.random_forest_model import RandomForestModel
from model_implementation.gru_model import GRUModel, plot_training_history, visualize_attention_weights
from model_implementation.sequence_preprocessor import SequencePreprocessor
from ensemble.ensemble_model import EnsembleModel
from ensemble.prediction_combiner import optimize_ensemble_weights, PredictionCombiner

# Evaluation imports
from evaluation.performance_metrics import (
    calculate_model_metrics,
    plot_roc_curves,
    plot_precision_recall_curves
)
from evaluation.fairness_metrics import evaluate_model_fairness, analyze_subgroup_fairness, calculate_fairness_metrics
from evaluation.fairness_analysis import analyze_bias_patterns
from evaluation.evaluation_report import generate_fairness_report, run_reporting_pipeline
from evaluation.cross_validation import perform_cross_validation
from evaluation.bias_mitigation import apply_threshold_adjustment, apply_reweighting
from evaluation.model_explainer import explain_model_predictions, generate_explanation_plots

# Visualization imports
from visualization.performance_visualizer import (
    visualize_demographic_distributions,
    visualize_performance_by_demographics,
    plot_fairness_metrics,
    compare_group_performance
)
from visualization.feature_visualizer import (
    visualize_feature_importance,
    visualize_ensemble_weights,
    plot_model_comparison_curves,
    plot_correlation_heatmap
)
from visualization.dashboard_exporter import (
    export_risk_distribution_data,
    export_demographic_performance_data,
    export_temporal_engagement_data,
    export_assessment_performance_data
)

# Utility imports
from utils.logging_utils import setup_logger, log_memory_usage, log_progress
from utils.monitoring_utils import track_execution_time, monitor_memory_usage
from utils.validation_utils import (
    validate_directories,
    validate_feature_engineering_inputs
)

# Configuration imports
from config import (
    DIRS, DATA_PROCESSING, FEATURE_ENGINEERING, 
    FAIRNESS, PROTECTED_ATTRIBUTES,
    LOG_DIR, LOG_LEVEL, RANDOM_SEED, OUTPUT_DIR,
    GRU_CONFIG, RF_CONFIG, ENSEMBLE_CONFIG
)

# Custom JSON encoder for numpy types
class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Initialize logger
logger = setup_logger(LOG_DIR, LOG_LEVEL)

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.
    
    Returns:
        argparse.Namespace: Validated command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='EduPredict 2.0: Academic Performance Prediction System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        type=str, 
        default="full",
        choices=[
            "full", "processing", "feature", "rf", 
            "gru", "ensemble", "evaluate", "visualize"
        ],
        help="Pipeline execution mode"
    )
    
    # Data processing parameters
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(DATA_PROCESSING.get('data_path', './data/OULAD')),
        help='Directory containing raw data files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help='Base directory for all outputs'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=DATA_PROCESSING.get('chunk_size', 10000),
        help='Chunk size for processing large files'
    )
    
    # Pipeline control flags
    parser.add_argument(
        '--load-processed',
        action='store_true',
        help='Load pre-processed data (skip processing)'
    )
    
    parser.add_argument(
        '--load-features',
        action='store_true',
        help='Load pre-engineered features (skip feature engineering)'
    )
    
    # Fairness and visualization options
    parser.add_argument(
        '--fairness-aware',
        action='store_true',
        help='Enable fairness-aware model training'
    )
    
    parser.add_argument(
        '--export-visualizations',
        action='store_true',
        help='Export visualizations and Tableau data'
    )
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.load_features and not args.load_processed:
        parser.error("--load-features requires --load-processed")
        
    if args.mode == "visualize" and not (args.load_processed or args.load_features):
        parser.error("Visualization mode requires processed data or features")
    
    return args

def setup_environment(args: argparse.Namespace) -> Tuple[Dict[str, Path], logging.Logger]:
    """
    Set up execution environment including directories and logging.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple containing directory paths and logger instance
    """
    try:
        # Update output directory if specified in args
        if args.output_dir != str(OUTPUT_DIR):
            for key, path in DIRS.items():
                DIRS[key] = Path(args.output_dir) / path.relative_to(OUTPUT_DIR)
        
        # Create all required directories
        for dir_path in DIRS.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Validate directory structure and permissions
        if not validate_directories(DIRS):
            raise RuntimeError("Directory validation failed")
        
        # Log initial setup information
        logger.info(f"Starting EduPredict 2.0 in {args.mode} mode")
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Chunk size: {args.chunk_size}")
        
        return DIRS, logger
        
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        raise

def validate_configuration() -> bool:
    """
    Validate configuration parameters for consistency and correctness.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        # Validate data processing parameters
        if not isinstance(DATA_PROCESSING['chunk_size'], int) or DATA_PROCESSING['chunk_size'] <= 0:
            logger.error("Invalid chunk_size in DATA_PROCESSING")
            return False
            
        if not isinstance(DATA_PROCESSING['dtypes'], dict):
            logger.error("Invalid dtypes specification in DATA_PROCESSING")
            return False
            
        # Validate feature engineering parameters
        if not all(isinstance(w, int) and w > 0 for w in FEATURE_ENGINEERING['window_sizes']):
            logger.error("Invalid window_sizes in FEATURE_ENGINEERING")
            return False
            
        if not (0 < FEATURE_ENGINEERING['correlation_threshold'] <= 1):
            logger.error("Invalid correlation_threshold in FEATURE_ENGINEERING")
            return False
            
        # Validate fairness parameters
        if not isinstance(PROTECTED_ATTRIBUTES, dict):
            logger.error("Invalid PROTECTED_ATTRIBUTES specification")
            return False
            
        if not (0 < FAIRNESS['threshold'] <= 1):
            logger.error("Invalid threshold in FAIRNESS")
            return False
            
        # Log validation success
        logger.info("Configuration validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False

@track_execution_time
def run_data_processing_workflow(
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Execute the data processing pipeline.
    
    Args:
        args: Command-line arguments
        dirs: Directory paths
        logger: Logger instance
        
    Returns:
        Dictionary containing processed data and splits
    """
    try:
        # Load raw datasets with chunking
        logger.info("Loading raw datasets...")
        datasets = load_raw_datasets(
            data_path=args.data_dir,
            chunk_size=args.chunk_size,
            dtypes=DATA_PROCESSING['dtypes'],
            logger=logger
        )
        
        # Verify data integrity
        logger.info("Verifying data integrity...")
        validation_results = validate_data_consistency(datasets, logger=logger)
        if not validation_results['is_valid']:
            for issue in validation_results['issues']:
                logger.error(f"Data consistency issue: {issue}")
            raise ValueError("Data consistency validation failed")
            
        # Analyze data quality
        logger.info("Analyzing data quality...")
        quality_report = detect_data_quality_issues(datasets, logger=logger)
        
        if quality_report['recommendations']:
            logger.warning("Data quality recommendations:")
            for rec in quality_report['recommendations']:
                logger.warning(f"- {rec}")
        
        # Clean datasets with progress tracking
        logger.info("Cleaning datasets...")
        clean_data = {}
        total_datasets = 4
        
        # Clean demographic data
        log_progress(logger, "Demographic cleaning", 1, total_datasets)
        clean_data['demographics'] = clean_demographic_data(
            student_info=datasets['student_info'],
            missing_value_strategy=None,  # Use default strategies
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
        cleaned_assessments, cleaned_submissions = clean_assessment_data(
            datasets['assessments'],
            datasets['student_assessments'],
            logger=logger
        )
        clean_data['assessments'] = cleaned_assessments
        clean_data['submissions'] = cleaned_submissions
        
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
            strat_cols=list(PROTECTED_ATTRIBUTES.keys()),
            test_size=0.2,
            validation_size=0.2,
            random_state=RANDOM_SEED,
            logger=logger
        )
        
        # Validate demographic balance in splits
        logger.info("Validating demographic balance...")
        for attr in PROTECTED_ATTRIBUTES:
            train_dist = splits['train'][attr].value_counts(normalize=True).to_dict()
            val_dist = splits['validation'][attr].value_counts(normalize=True).to_dict()
            test_dist = splits['test'][attr].value_counts(normalize=True).to_dict()
            
            logger.info(f"{attr} distribution in train: {train_dist}")
            logger.info(f"{attr} distribution in validation: {val_dist}")
            logger.info(f"{attr} distribution in test: {test_dist}")
            
            if not validate_demographic_balance(train_dist, val_dist, test_dist, FAIRNESS['threshold']):
                logger.warning(f"Demographic imbalance detected in {attr}")
                continue
        
        # Save processed datasets
        logger.info("Saving processed datasets...")
        processed_dir = dirs['processed_data']

        # Save demographics
        demographics_path = processed_dir / "demographics.parquet"
        clean_data['demographics'].to_parquet(demographics_path)
        logger.info(f"Saved demographics to {demographics_path}")

        # Save VLE data
        vle_path = processed_dir / "vle.parquet"
        clean_data['vle'].to_parquet(vle_path)
        logger.info(f"Saved vle to {vle_path}")

        # Save assessment data - handle assessments and submissions separately
        assessment_path = processed_dir / "assessments.parquet"
        submissions_path = processed_dir / "submissions.parquet"
        clean_data['assessments'].to_parquet(assessment_path)
        clean_data['submissions'].to_parquet(submissions_path)
        logger.info(f"Saved assessments to {assessment_path}")
        logger.info(f"Saved submissions to {submissions_path}")
        
        # Save registration data
        registration_path = processed_dir / "registration.parquet"
        clean_data['registration'].to_parquet(registration_path)
        logger.info(f"Saved registration to {registration_path}")
        
        return {
            'clean_data': clean_data,
            'splits': splits,
            'quality_report': quality_report
        }
        
    except Exception as e:
        logger.error(f"Data processing workflow failed: {str(e)}")
        raise

@track_execution_time
def run_feature_engineering(
    data_results: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Execute the feature engineering pipeline.
    
    Args:
        data_results: Results from data processing
        dirs: Directory paths
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        Dictionary containing engineered features and metadata
    """
    try:
        clean_data = data_results.get('clean_data', {})
        splits = data_results.get('splits', {})
        
        if not clean_data or not splits:
            raise ValueError("Missing required data processing results")
            
        # Initialize result dictionary
        feature_results = {
            'static_features': {},
            'sequential_features': {},
            'feature_metadata': {},
        }
            
        # Load data if not already in memory
        if not clean_data:
            processed_dir = dirs['processed_data']
            logger.info("Loading processed data from disk...")
            
            demographics_path = processed_dir / "demographics.parquet"
            vle_path = processed_dir / "vle.parquet"
            assessment_path = processed_dir / "assessments.parquet"
            submissions_path = processed_dir / "submissions.parquet"
            registration_path = processed_dir / "registration.parquet"
            
            clean_data = {
                'demographics': pd.read_parquet(demographics_path),
                'vle': pd.read_parquet(vle_path),
                'assessments': pd.read_parquet(assessment_path),
                'submissions': pd.read_parquet(submissions_path),
                'registration': pd.read_parquet(registration_path)
            }
        
        # Create train/validation/test indices for consistent splitting
        if not splits:
            logger.info("Creating data splits...")
            splits = create_stratified_splits(
                clean_data['demographics'],
                target_col='final_result',
                strat_cols=list(PROTECTED_ATTRIBUTES.keys()),
                test_size=0.2,
                validation_size=0.2,
                random_state=RANDOM_SEED,
                logger=logger
            )
            
        # Extract student IDs for each split
        student_ids = {
            'train': splits['train']['id_student'].unique(),
            'validation': splits['validation']['id_student'].unique(),
            'test': splits['test']['id_student'].unique()
        }
        
        # Create static features (demographic and educational)
        logger.info("Creating demographic features...")
        demographic_features = create_demographic_features(
            clean_data['demographics'],
            categorical_cols=FEATURE_ENGINEERING['categorical_cols'],
            one_hot=FEATURE_ENGINEERING['one_hot_encoding'],
            logger=logger
        )
        
        # Filter static features by split
        static_features = {}
        for split_name in ['train', 'validation', 'test']:
            split_mask = demographic_features['id_student'].isin(student_ids[split_name])
            static_features[split_name] = demographic_features[split_mask].copy()
            logger.info(f"Created {len(static_features[split_name])} static features for {split_name}")
            
        # Create temporal features with multiple window sizes
        logger.info("Creating temporal features...")
        temporal_features = create_temporal_features(
            clean_data['vle'],
            student_ids=student_ids,
            window_sizes=FEATURE_ENGINEERING['window_sizes'],
            module_info=clean_data.get('registration', None),
            logger=logger
        )
        
        # Create sequential features for GRU model
        logger.info("Creating sequential features...")
        sequential_features = create_sequential_features(
            clean_data['vle'],
            clean_data['submissions'],
            student_ids=student_ids,
            max_seq_length=FEATURE_ENGINEERING['max_seq_length'],
            categorical_cols=FEATURE_ENGINEERING['sequential_categorical_cols'],
            logger=logger
        )
        
        # Combine all static features
        logger.info("Combining features...")
        for split_name in ['train', 'validation', 'test']:
            if split_name in temporal_features:
                # Merge temporal features with static features
                static_features[split_name] = pd.merge(
                    static_features[split_name],
                    temporal_features[split_name],
                    on='id_student',
                    how='left'
                )
                
                # Add split label
                static_features[split_name]['split'] = split_name
                
        # Feature selection and engineering only on training data
        train_features = static_features['train'].copy()
        
        # Analyze feature importance
        logger.info("Analyzing feature importance...")
        importance_results = analyze_feature_importance(
            train_features,
            target_col='final_result',
            categorical_cols=FEATURE_ENGINEERING['categorical_cols'],
            random_state=RANDOM_SEED,
            logger=logger
        )
        
        # Identify and remove highly correlated features
        logger.info("Analyzing feature correlations...")
        correlation_results = analyze_feature_correlations(
            train_features.drop(columns=['id_student', 'final_result', 'split']),
            threshold=FEATURE_ENGINEERING['correlation_threshold'],
            logger=logger
        )
        
        # Get optimal feature subset
        logger.info("Selecting optimal feature subset...")
        selected_features = select_features_by_importance(
            importance_results,
            correlation_results,
            top_k=FEATURE_ENGINEERING['top_k_features'],
            min_importance=FEATURE_ENGINEERING['min_importance'],
            logger=logger
        )
        
        # Analyze impact of feature selection on demographics
        logger.info("Analyzing demographic impact of feature selection...")
        demographic_impact = analyze_demographic_impact(
            train_features,
            selected_features,
            protected_attributes=PROTECTED_ATTRIBUTES,
            target_col='final_result',
            logger=logger
        )
        
        if demographic_impact['warnings']:
            for warning in demographic_impact['warnings']:
                logger.warning(f"Feature selection impact: {warning}")
                
        # Apply feature selection to all splits
        selected_cols = ['id_student', 'final_result', 'split'] + selected_features
        for split_name in ['train', 'validation', 'test']:
            # Keep only selected features plus ID, target and split columns
            available_cols = [col for col in selected_cols if col in static_features[split_name].columns]
            static_features[split_name] = static_features[split_name][available_cols]
            
            # Count features
            feature_count = len(static_features[split_name].columns) - 3  # Subtract ID, target and split
            logger.info(f"{split_name} set: {feature_count} selected features for {len(static_features[split_name])} students")
        
        # Export feature metadata
        logger.info("Exporting feature metadata...")
        feature_metadata = export_feature_metadata(
            static_features['train'],
            importance_results,
            correlation_results,
            demographic_impact,
            dirs['feature_data'],
            logger=logger
        )
        
        # Save engineered features
        logger.info("Saving engineered features...")
        feature_dir = dirs['feature_data']
        
        # Save all static features
        for split_name in ['train', 'validation', 'test']:
            output_path = feature_dir / f"{split_name}_static_features.parquet"
            static_features[split_name].to_parquet(output_path)
            logger.info(f"Saved {split_name} static features to {output_path}")
            
        # Save all sequential features
        for split_name in ['train', 'validation', 'test']:
            if split_name in sequential_features and 'sequence_data' in sequential_features[split_name]:
                # Save sequences in an appropriate format
                output_path = feature_dir / f"{split_name}_sequences.npz"
                seq_data = sequential_features[split_name]['sequence_data']
                student_ids = sequential_features[split_name]['student_ids']
                targets = sequential_features[split_name].get('targets', None)
                
                np.savez_compressed(
                    output_path,
                    sequences=seq_data,
                    student_ids=student_ids,
                    targets=targets if targets is not None else []
                )
                logger.info(f"Saved {split_name} sequential features to {output_path}")
                
                # Also save any metadata
                if 'metadata' in sequential_features[split_name]:
                    metadata_path = feature_dir / f"{split_name}_sequence_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(
                            sequential_features[split_name]['metadata'],
                            f,
                            cls=NumpyJSONEncoder,
                            indent=2
                        )
                
        # Store results
        feature_results = {
            'static_features': static_features,
            'sequential_features': sequential_features,
            'feature_metadata': feature_metadata,
            'selected_features': selected_features,
            'importance_results': importance_results
        }
        
        return feature_results
        
    except Exception as e:
        logger.error(f"Feature engineering workflow failed: {str(e)}")
        raise

@track_execution_time
def run_random_forest_pipeline(
    feature_results: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Execute the Random Forest model training pipeline.
    
    Args:
        feature_results: Results from feature engineering
        dirs: Directory paths
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        Dictionary containing model results and evaluation metrics
    """
    try:
        logger.info("Starting Random Forest model pipeline...")
        
        # Initialize results dictionary
        rf_results = {
            'model': None,
            'predictions': {},
            'thresholds': {},
            'metrics': {},
            'feature_importance': {}
        }
        
        # Get static features for model training
        static_features = feature_results.get('static_features', {})
        if not static_features or not all(split in static_features for split in ['train', 'validation', 'test']):
            raise ValueError("Missing required static features for model training")
        
        # Extract feature and target data
        X_train = static_features['train'].drop(columns=['id_student', 'final_result', 'split'])
        y_train = static_features['train']['final_result']
        
        X_val = static_features['validation'].drop(columns=['id_student', 'final_result', 'split'])
        y_val = static_features['validation']['final_result']
        
        X_test = static_features['test'].drop(columns=['id_student', 'final_result', 'split'])
        y_test = static_features['test']['final_result']
        
        # Record feature names
        feature_names = X_train.columns.tolist()
        logger.info(f"Training model with {len(feature_names)} features")
        
        # Initialize Random Forest model with configuration
        rf_model = RandomForestModel(
            random_state=RANDOM_SEED,
            **RF_CONFIG
        )
        
        # Tune hyperparameters if specified
        if args.mode in ['complete', 'training'] and not args.load_features:
            logger.info("Tuning Random Forest hyperparameters...")
            rf_model.tune_hyperparameters(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                param_grid=RF_CONFIG.get('param_grid', None),
                cv=RF_CONFIG.get('cv_folds', 5),
                fairness_aware=args.fairness_aware,
                protected_attributes={k: static_features['train'][k] for k in PROTECTED_ATTRIBUTES},
                logger=logger
            )
            
        # Train the Random Forest model
        logger.info("Training Random Forest model...")
        rf_model.train(
            X_train, y_train,
            categorical_features=FEATURE_ENGINEERING['categorical_cols'],
            class_weight=RF_CONFIG.get('class_weight', None),
            logger=logger
        )
        
        # Generate predictions for all datasets
        logger.info("Generating predictions...")
        predictions = {}
        
        for split_name, (X, y) in [
            ('train', (X_train, y_train)), 
            ('validation', (X_val, y_val)), 
            ('test', (X_test, y_test))
        ]:
            # Get raw probabilities
            proba = rf_model.predict_proba(X)
            
            # Create DataFrame with student IDs and predictions
            pred_df = pd.DataFrame({
                'id_student': static_features[split_name]['id_student'],
                'true_label': y,
                'pred_proba': proba[:, 1]  # Probability of positive class
            })
            
            predictions[split_name] = pred_df
            
        # Find optimal classification threshold on validation set
        logger.info("Finding optimal classification threshold...")
        if args.fairness_aware:
            # Find group-specific thresholds
            protected_cols = {k: static_features['validation'][k] for k in PROTECTED_ATTRIBUTES}
            thresholds = rf_model.find_group_specific_thresholds(
                predictions['validation']['pred_proba'],
                predictions['validation']['true_label'],
                protected_attributes=protected_cols,
                optimization_metric=RF_CONFIG.get('threshold_metric', 'f1'),
                logger=logger
            )
            
            logger.info(f"Group-specific thresholds: {thresholds}")
        else:
            # Find single optimal threshold
            threshold = rf_model.find_optimal_threshold(
                predictions['validation']['pred_proba'],
                predictions['validation']['true_label'],
                optimization_metric=RF_CONFIG.get('threshold_metric', 'f1'),
                logger=logger
            )
            
            thresholds = {'default': threshold}
            logger.info(f"Optimal threshold: {threshold:.4f}")
        
        # Apply thresholds and calculate metrics
        logger.info("Evaluating model performance...")
        metrics = {}
        
        for split_name, pred_df in predictions.items():
            # Apply thresholds to get predicted labels
            if args.fairness_aware and len(thresholds) > 1:
                # Apply group-specific thresholds
                pred_labels = []
                protected_data = {k: static_features[split_name][k] for k in PROTECTED_ATTRIBUTES}
                
                # For each student, apply the appropriate threshold based on demographics
                for idx, row in pred_df.iterrows():
                    # Create demographic key for this student
                    demo_key = tuple(protected_data[attr].iloc[idx] for attr in PROTECTED_ATTRIBUTES)
                    # Apply appropriate threshold
                    threshold = thresholds.get(demo_key, thresholds['default'])
                    pred_labels.append(1 if row['pred_proba'] >= threshold else 0)
                    
                pred_df['pred_label'] = pred_labels
            else:
                # Apply single threshold
                pred_df['pred_label'] = (pred_df['pred_proba'] >= thresholds['default']).astype(int)
                
            # Calculate metrics
            split_metrics = calculate_model_metrics(
                pred_df['true_label'],
                pred_df['pred_label'],
                pred_df['pred_proba'],
                logger=logger
            )
            
            metrics[split_name] = split_metrics
            logger.info(f"{split_name} set metrics: {split_metrics}")
            
        # Analyze feature importance
        logger.info("Analyzing feature importance...")
        feature_importance = rf_model.get_feature_importance(feature_names)
        
        # Save top features
        top_features = sorted(
            [(name, importance) for name, importance in feature_importance.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        logger.info("Top 20 features by importance:")
        for feature, importance in top_features:
            logger.info(f"  {feature}: {importance:.4f}")
        
        # Generate precision-recall and ROC curves
        if args.export_visualizations:
            logger.info("Generating performance visualizations...")
            
            # Plot ROC curves
            roc_fig = plot_roc_curves(
                {split: pred_df for split, pred_df in predictions.items()},
                title="Random Forest ROC Curves"
            )
            roc_path = dirs['visualizations'] / "rf_roc_curves.png"
            roc_fig.savefig(roc_path)
            plt.close(roc_fig)
            
            # Plot precision-recall curves
            pr_fig = plot_precision_recall_curves(
                {split: pred_df for split, pred_df in predictions.items()},
                title="Random Forest Precision-Recall Curves"
            )
            pr_path = dirs['visualizations'] / "rf_precision_recall_curves.png"
            pr_fig.savefig(pr_path)
            plt.close(pr_fig)
            
            # Visualize feature importance
            fi_fig = visualize_feature_importance(
                feature_importance, 
                top_n=20,
                title="Random Forest Feature Importance"
            )
            fi_path = dirs['visualizations'] / "rf_feature_importance.png"
            fi_fig.savefig(fi_path)
            plt.close(fi_fig)
            
        # Save model
        logger.info("Saving Random Forest model...")
        model_dir = dirs['models'] / "final"
        model_dir.mkdir(exist_ok=True, parents=True)
        
        model_path = model_dir / "random_forest.pkl"
        rf_model.save(model_path)
        
        # Save model metadata
        metadata = {
            'features': feature_names,
            'hyperparameters': rf_model.get_params(),
            'thresholds': thresholds,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'fairness_aware': args.fairness_aware
        }
        
        metadata_path = model_dir / "random_forest_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, cls=NumpyJSONEncoder, indent=2)
            
        # Store results
        rf_results = {
            'model': rf_model,
            'predictions': predictions,
            'thresholds': thresholds,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        return rf_results
        
    except Exception as e:
        logger.error(f"Random Forest pipeline failed: {str(e)}")
        raise

@track_execution_time
def run_gru_pipeline(
    feature_results: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Execute the GRU model training pipeline for sequential data.
    
    Args:
        feature_results: Results from feature engineering
        dirs: Directory paths
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        Dictionary containing model results and evaluation metrics
    """
    try:
        logger.info("Starting GRU sequential model pipeline...")
        
        # Initialize results dictionary
        gru_results = {
            'model': None,
            'predictions': {},
            'thresholds': {},
            'metrics': {},
            'sequence_metrics': {},
            'attention_weights': None
        }
        
        # Get sequential features for model training
        seq_features = feature_results.get('sequential_features', {})
        if not seq_features or not all(split in seq_features for split in ['train', 'validation', 'test']):
            raise ValueError("Missing required sequential features for GRU model training")
        
        # Load sequence preprocessor module
        seq_processor = SequencePreprocessor(
            max_seq_length=GRU_CONFIG.get('max_sequence_length', 100),
            feature_dims=GRU_CONFIG.get('feature_dimensions', 15),
            random_state=RANDOM_SEED,
            logger=logger
        )
        
        # Process and prepare sequential data
        logger.info("Processing sequential data...")
        seq_data = {}
        
        for split in ['train', 'validation', 'test']:
            # Extract sequential data and targets
            sequences = seq_features[split]['sequence_data']
            sequence_lengths = seq_features[split]['sequence_lengths']
            targets = seq_features[split]['targets']
            student_ids = seq_features[split]['student_ids']
            
            # Process sequences for GRU input
            seq_data[split] = {
                'X_seq': seq_processor.preprocess(sequences, sequence_lengths),
                'lengths': sequence_lengths,
                'y': targets,
                'ids': student_ids
            }
            
        # Initialize GRU model
        gru_model = GRUModel(
            input_dim=seq_processor.get_feature_dim(),
            hidden_dim=GRU_CONFIG.get('hidden_dim', 64),
            output_dim=GRU_CONFIG.get('output_dim', 1),
            num_layers=GRU_CONFIG.get('num_layers', 2),
            dropout=GRU_CONFIG.get('dropout', 0.2),
            bidirectional=GRU_CONFIG.get('bidirectional', True),
            use_attention=GRU_CONFIG.get('use_attention', True),
            learning_rate=GRU_CONFIG.get('learning_rate', 0.001),
            random_state=RANDOM_SEED,
            device=args.device
        )
        
        # Train the model
        logger.info("Training GRU model...")
        train_history = gru_model.train(
            seq_data['train']['X_seq'], 
            seq_data['train']['lengths'],
            seq_data['train']['y'],
            validation_data=(
                seq_data['validation']['X_seq'],
                seq_data['validation']['lengths'],
                seq_data['validation']['y']
            ),
            batch_size=GRU_CONFIG.get('batch_size', 32),
            epochs=GRU_CONFIG.get('epochs', 30),
            early_stopping=GRU_CONFIG.get('early_stopping', True),
            patience=GRU_CONFIG.get('patience', 5),
            class_weights=GRU_CONFIG.get('class_weights', None),
            logger=logger
        )
        
        # Generate predictions for all datasets
        logger.info("Generating GRU model predictions...")
        predictions = {}
        attention_weights = {}
        
        for split_name in ['train', 'validation', 'test']:
            # Get raw prediction probabilities and attention weights
            proba, attn = gru_model.predict_with_attention(
                seq_data[split_name]['X_seq'],
                seq_data[split_name]['lengths']
            )
            
            # Create DataFrame with student IDs and predictions
            pred_df = pd.DataFrame({
                'id_student': seq_data[split_name]['ids'],
                'true_label': seq_data[split_name]['y'],
                'pred_proba': proba.flatten()
            })
            
            predictions[split_name] = pred_df
            attention_weights[split_name] = attn
        
        # Find optimal classification threshold on validation set
        logger.info("Finding optimal GRU classification threshold...")
        
        # Find single optimal threshold (GRU doesn't support fairness-aware thresholds yet)
        threshold = gru_model.find_optimal_threshold(
            predictions['validation']['pred_proba'],
            predictions['validation']['true_label'],
            optimization_metric=GRU_CONFIG.get('threshold_metric', 'f1')
        )
        
        thresholds = {'default': threshold}
        logger.info(f"Optimal GRU threshold: {threshold:.4f}")
        
        # Apply thresholds and calculate metrics
        logger.info("Evaluating GRU model performance...")
        metrics = {}
        sequence_metrics = {}
        
        for split_name, pred_df in predictions.items():
            # Apply threshold to get predicted labels
            pred_df['pred_label'] = (pred_df['pred_proba'] >= threshold).astype(int)
                
            # Calculate standard metrics
            split_metrics = calculate_model_metrics(
                pred_df['true_label'],
                pred_df['pred_label'],
                pred_df['pred_proba'],
                logger=logger
            )
            
            # Calculate sequence-specific metrics if applicable
            seq_split_metrics = {}
            if GRU_CONFIG.get('calculate_sequence_metrics', False):
                seq_split_metrics = gru_model.evaluate_sequence_predictions(
                    seq_data[split_name]['X_seq'],
                    seq_data[split_name]['lengths'],
                    seq_data[split_name]['y'],
                    attention_weights[split_name]
                )
            
            metrics[split_name] = split_metrics
            sequence_metrics[split_name] = seq_split_metrics
            logger.info(f"{split_name} set GRU metrics: {split_metrics}")
        
        # Generate visualizations
        if args.export_visualizations:
            logger.info("Generating GRU performance visualizations...")
            
            # Plot training history
            history_fig = plot_training_history(
                train_history,
                title="GRU Training History"
            )
            history_path = dirs['visualizations'] / "gru_training_history.png"
            history_fig.savefig(history_path)
            plt.close(history_fig)
            
            # Plot ROC curves
            roc_fig = plot_roc_curves(
                {split: pred_df for split, pred_df in predictions.items()},
                title="GRU ROC Curves"
            )
            roc_path = dirs['visualizations'] / "gru_roc_curves.png"
            roc_fig.savefig(roc_path)
            plt.close(roc_fig)
            
            # Plot precision-recall curves
            pr_fig = plot_precision_recall_curves(
                {split: pred_df for split, pred_df in predictions.items()},
                title="GRU Precision-Recall Curves"
            )
            pr_path = dirs['visualizations'] / "gru_precision_recall_curves.png"
            pr_fig.savefig(pr_path)
            plt.close(pr_fig)
            
            # Visualize attention weights for a few examples
            if GRU_CONFIG.get('use_attention', True):
                attn_fig = visualize_attention_weights(
                    attention_weights['test'][:5],  # First 5 test samples
                    seq_data['test']['lengths'][:5],
                    title="GRU Attention Visualization"
                )
                attn_path = dirs['visualizations'] / "gru_attention_weights.png"
                attn_fig.savefig(attn_path)
                plt.close(attn_fig)
        
        # Save model
        logger.info("Saving GRU model...")
        model_dir = dirs['models'] / "final"
        model_dir.mkdir(exist_ok=True, parents=True)
        
        model_path = model_dir / "gru_model.pt"
        gru_model.save(model_path)
        
        # Save model metadata
        metadata = {
            'model_architecture': gru_model.get_config(),
            'sequence_processor': seq_processor.get_config(),
            'threshold': threshold,
            'metrics': metrics,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = model_dir / "gru_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, cls=NumpyJSONEncoder, indent=2)
        
        # Store results
        gru_results = {
            'model': gru_model,
            'seq_processor': seq_processor,
            'predictions': predictions,
            'thresholds': thresholds,
            'metrics': metrics,
            'sequence_metrics': sequence_metrics,
            'attention_weights': attention_weights
        }
        
        return gru_results
        
    except Exception as e:
        logger.error(f"GRU pipeline failed: {str(e)}")
        raise

@track_execution_time
def run_ensemble_integration(
    rf_results: Dict[str, Any],
    gru_results: Dict[str, Any],
    feature_results: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Integrate Random Forest and GRU models into an ensemble model.
    
    Args:
        rf_results: Results from random forest pipeline
        gru_results: Results from GRU pipeline
        feature_results: Results from feature engineering
        dirs: Directory paths
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        Dictionary containing ensemble model results
    """
    try:
        logger.info("Starting ensemble model integration...")
        
        # Initialize results dictionary
        ensemble_results = {
            'model': None,
            'predictions': {},
            'thresholds': {},
            'metrics': {},
            'feature_importance': None
        }
        
        # Get predictions from base models
        rf_predictions = rf_results.get('predictions', {})
        gru_predictions = gru_results.get('predictions', {})
        
        if not rf_predictions or not gru_predictions:
            raise ValueError("Missing required predictions from base models for ensemble integration")
        
        # Extract demographic features for fairness-aware ensembling
        demographic_features = feature_results.get('tabular_features', {}).get('demographic_features', {})
        
        # Initialize prediction combiner
        prediction_combiner = PredictionCombiner(
            method=ENSEMBLE_CONFIG.get('method', 'weighted_average'),
            weights=ENSEMBLE_CONFIG.get('weights', {'rf': 0.6, 'gru': 0.4}),
            optimize_weights=ENSEMBLE_CONFIG.get('optimize_weights', True),
            fairness_constraints=ENSEMBLE_CONFIG.get('fairness_constraints', None),
            random_state=RANDOM_SEED,
            logger=logger
        )
        
        # Preprocess and combine predictions
        logger.info("Combining model predictions...")
        combined_predictions = {}
        
        for split_name in ['train', 'validation', 'test']:
            if split_name not in rf_predictions or split_name not in gru_predictions:
                logger.warning(f"Missing {split_name} predictions from one of the base models")
                continue
                
            # Prepare prediction DataFrames
            rf_pred_df = rf_predictions[split_name]
            gru_pred_df = gru_predictions[split_name]
            
            # Merge predictions by student ID
            merged_df = pd.merge(
                rf_pred_df, 
                gru_pred_df,
                on=['id_student', 'true_label'],
                suffixes=('_rf', '_gru')
            )
            
            # Add demographic features if available and configured
            if ENSEMBLE_CONFIG.get('use_demographics', False) and split_name in demographic_features:
                # Get demographic features for this split
                demo_df = demographic_features[split_name]
                
                # Merge demographic features by student ID
                merged_df = pd.merge(
                    merged_df,
                    demo_df,
                    on='id_student',
                    how='left'
                )
            
            # For validation set, optimize ensemble weights if configured
            if split_name == 'validation' and ENSEMBLE_CONFIG.get('optimize_weights', True):
                logger.info("Optimizing ensemble weights on validation data...")
                optimal_weights = prediction_combiner.optimize_weights(
                    merged_df['pred_proba_rf'],
                    merged_df['pred_proba_gru'],
                    merged_df['true_label'],
                    sensitive_features=merged_df['gender'] if 'gender' in merged_df.columns else None
                )
                logger.info(f"Optimal ensemble weights: {optimal_weights}")
            
            # Combine predictions
            merged_df['ensemble_proba'] = prediction_combiner.combine_predictions(
                merged_df['pred_proba_rf'],
                merged_df['pred_proba_gru'],
                sensitive_features=merged_df['gender'] if 'gender' in merged_df.columns else None
            )
            
            combined_predictions[split_name] = merged_df
        
        # Create ensemble model
        logger.info("Creating ensemble model...")
        ensemble_model = EnsembleModel(
            base_models={
                'random_forest': rf_results.get('model'),
                'gru': gru_results.get('model')
            },
            prediction_combiner=prediction_combiner,
            random_state=RANDOM_SEED
        )
        
        # Find optimal ensemble threshold(s)
        logger.info("Finding optimal ensemble thresholds...")
        
        # Get validation predictions
        val_preds = combined_predictions['validation']
        
        # If fairness-aware thresholds are configured
        if ENSEMBLE_CONFIG.get('fairness_aware_thresholds', False) and 'gender' in val_preds.columns:
            thresholds = {}
            
            # Find separate thresholds for each demographic group
            for group in val_preds['gender'].unique():
                group_idx = val_preds['gender'] == group
                group_thresh = ensemble_model.find_optimal_threshold(
                    val_preds.loc[group_idx, 'ensemble_proba'],
                    val_preds.loc[group_idx, 'true_label'],
                    metric=ENSEMBLE_CONFIG.get('threshold_metric', 'f1')
                )
                thresholds[group] = group_thresh
                logger.info(f"Optimal threshold for {group}: {group_thresh:.4f}")
        else:
            # Find single optimal threshold
            threshold = ensemble_model.find_optimal_threshold(
                val_preds['ensemble_proba'],
                val_preds['true_label'],
                metric=ENSEMBLE_CONFIG.get('threshold_metric', 'f1')
            )
            thresholds = {'default': threshold}
            logger.info(f"Optimal ensemble threshold: {threshold:.4f}")
        
        # Apply thresholds and evaluate metrics
        logger.info("Evaluating ensemble model performance...")
        metrics = {}
        
        for split_name, pred_df in combined_predictions.items():
            # Apply threshold (group-specific if available)
            if len(thresholds) > 1 and 'gender' in pred_df.columns:
                # Apply group-specific thresholds
                pred_df['ensemble_label'] = pred_df.apply(
                    lambda row: 1 if row['ensemble_proba'] >= thresholds.get(row['gender'], 0.5) else 0,
                    axis=1
                )
            else:
                # Apply default threshold
                default_threshold = thresholds.get('default', 0.5)
                pred_df['ensemble_label'] = (pred_df['ensemble_proba'] >= default_threshold).astype(int)
            
            # Calculate metrics
            split_metrics = calculate_model_metrics(
                pred_df['true_label'],
                pred_df['ensemble_label'],
                pred_df['ensemble_proba'],
                logger=logger
            )
            
            metrics[split_name] = split_metrics
            logger.info(f"{split_name} set ensemble metrics: {split_metrics}")
            
            # Compare with base models
            rf_metrics = rf_results.get('metrics', {}).get(split_name, {})
            gru_metrics = gru_results.get('metrics', {}).get(split_name, {})
            
            if rf_metrics and gru_metrics:
                logger.info(f"{split_name} performance comparison:")
                logger.info(f"  Random Forest F1: {rf_metrics.get('f1_score', 0):.4f}")
                logger.info(f"  GRU F1: {gru_metrics.get('f1_score', 0):.4f}")
                logger.info(f"  Ensemble F1: {split_metrics.get('f1_score', 0):.4f}")
        
        # Generate visualizations
        if args.export_visualizations:
            logger.info("Generating ensemble model visualizations...")
            
            # Create comparison plot of ROC curves for all models
            models_roc_data = {}
            for split_name in ['test']:  # Only test for final evaluation
                if split_name in combined_predictions:
                    pred_df = combined_predictions[split_name]
                    models_roc_data = {
                        'Random Forest': (pred_df['true_label'], pred_df['pred_proba_rf']),
                        'GRU': (pred_df['true_label'], pred_df['pred_proba_gru']),
                        'Ensemble': (pred_df['true_label'], pred_df['ensemble_proba'])
                    }
            
            if models_roc_data:
                # Plot ROC curve comparison
                roc_fig = plot_model_comparison_curves(
                    models_roc_data,
                    curve_type='roc',
                    title="Model Comparison: ROC Curves"
                )
                roc_path = dirs['visualizations'] / "ensemble_model_comparison_roc.png"
                roc_fig.savefig(roc_path)
                plt.close(roc_fig)
                
                # Plot precision-recall curve comparison
                pr_fig = plot_model_comparison_curves(
                    models_roc_data,
                    curve_type='precision_recall',
                    title="Model Comparison: Precision-Recall Curves"
                )
                pr_path = dirs['visualizations'] / "ensemble_model_comparison_pr.png"
                pr_fig.savefig(pr_path)
                plt.close(pr_fig)
            
            # Generate model weights visualization
            weights = prediction_combiner.get_weights()
            weight_fig = visualize_ensemble_weights(weights)
            weight_path = dirs['visualizations'] / "ensemble_weights.png"
            weight_fig.savefig(weight_path)
            plt.close(weight_fig)
        
        # Save ensemble model
        logger.info("Saving ensemble model...")
        model_dir = dirs['models'] / "ensemble"
        model_dir.mkdir(exist_ok=True, parents=True)
        
        model_path = model_dir / "ensemble_model.pkl"
        ensemble_model.save(model_path)
        
        # Save ensemble metadata
        metadata = {
            'model_configuration': ensemble_model.get_config(),
            'thresholds': thresholds,
            'metrics': metrics,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = model_dir / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, cls=NumpyJSONEncoder, indent=2)
        
        # Store results
        ensemble_results = {
            'model': ensemble_model,
            'predictions': combined_predictions,
            'thresholds': thresholds,
            'metrics': metrics
        }
        
        return ensemble_results
        
    except Exception as e:
        logger.error(f"Ensemble integration failed: {str(e)}")
        raise

@track_execution_time
def run_fairness_pipeline(
    rf_results: Dict[str, Any],
    gru_results: Dict[str, Any],
    ensemble_results: Dict[str, Any],
    feature_results: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Execute the fairness evaluation workflow.
    
    Args:
        rf_results: Results from Random Forest pipeline
        gru_results: Results from GRU pipeline
        ensemble_results: Results from ensemble pipeline
        feature_results: Results from feature engineering
        dirs: Directory paths
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        Dictionary containing fairness evaluation results
    """
    try:
        # Extract models and test data
        rf_model = rf_results['rf_model']
        gru_model = gru_results['gru_model']
        ensemble = ensemble_results['ensemble']
        
        # Test data
        X_test_rf = rf_results['X_test']
        y_test = rf_results['y_test']
        test_sequences = gru_results['test_sequences']
        test_student_id_map = gru_results['student_id_maps']['test']
        splits = feature_results['splits']
        
        # Get demographic data for test set
        test_demographics = splits['test'][list(PROTECTED_ATTRIBUTES.keys()) + ['id_student']]
        
        # Get predictions from all models on test set
        logger.info("Generating predictions on test set...")
        
        # RF predictions
        rf_probs = rf_model.predict_proba(X_test_rf)
        rf_preds = rf_model.predict(X_test_rf, threshold=rf_results['optimal_threshold'])
        
        # GRU predictions
        gru_probs = gru_model.predict_proba(test_sequences)
        gru_preds = gru_model.predict(test_sequences, threshold=0.5)
        
        # Ensemble predictions
        ensemble_probs = ensemble.predict_proba(
            static_features=X_test_rf,
            sequential_features=test_sequences,
            student_id_map=test_student_id_map
        )
        ensemble_preds = ensemble.predict(
            static_features=X_test_rf,
            sequential_features=test_sequences,
            student_id_map=test_student_id_map,
            use_demographic_thresholds=args.fairness_aware
        )
        
        # Evaluate fairness for each model
        logger.info("Evaluating model fairness across demographic groups...")
        
        # Create protected attribute arrays
        protected_attributes_dict = {}
        for attr in PROTECTED_ATTRIBUTES:
            protected_attributes_dict[attr] = test_demographics[attr].values
        
        # RF fairness evaluation
        logger.info("Evaluating Random Forest fairness...")
        rf_fairness = evaluate_model_fairness(
            y_true=y_test,
            y_pred=rf_preds,
            y_prob=rf_probs,
            protected_attributes=protected_attributes_dict,
            thresholds=FAIRNESS['thresholds'],
            metrics=['accuracy', 'precision', 'recall', 'f1', 'auc'],
            fairness_metrics=['demographic_parity_difference', 'disparate_impact_ratio', 'equal_opportunity_difference'],
            min_group_size=FAIRNESS['min_group_size'],
            logger=logger
        )
        
        # GRU fairness evaluation
        logger.info("Evaluating GRU fairness...")
        gru_fairness = evaluate_model_fairness(
            y_true=y_test,
            y_pred=gru_preds,
            y_prob=gru_probs,
            protected_attributes=protected_attributes_dict,
            thresholds=FAIRNESS['thresholds'],
            metrics=['accuracy', 'precision', 'recall', 'f1', 'auc'],
            fairness_metrics=['demographic_parity_difference', 'disparate_impact_ratio', 'equal_opportunity_difference'],
            min_group_size=FAIRNESS['min_group_size'],
            logger=logger
        )
        
        # Ensemble fairness evaluation
        logger.info("Evaluating Ensemble fairness...")
        ensemble_fairness = evaluate_model_fairness(
            y_true=y_test,
            y_pred=ensemble_preds,
            y_prob=ensemble_probs,
            protected_attributes=protected_attributes_dict,
            thresholds=FAIRNESS['thresholds'],
            metrics=['accuracy', 'precision', 'recall', 'f1', 'auc'],
            fairness_metrics=['demographic_parity_difference', 'disparate_impact_ratio', 'equal_opportunity_difference'],
            min_group_size=FAIRNESS['min_group_size'],
            logger=logger
        )
        
        # Analyze intersectional fairness for ensemble
        logger.info("Analyzing intersectional fairness...")
        intersectional_fairness = analyze_subgroup_fairness(
            y_true=y_test,
            y_pred=ensemble_preds,
            y_prob=ensemble_probs,
            protected_attributes=protected_attributes_dict,
            metrics=['accuracy', 'f1'],
            fairness_metrics=['demographic_parity_difference', 'equal_opportunity_difference'],
            min_group_size=FAIRNESS['min_group_size'],
            logger=logger
        )
        
        # Generate fairness reports
        logger.info("Generating fairness reports...")
        
        # RF fairness report
        rf_report = generate_fairness_report(
            fairness_results=rf_fairness,
            thresholds=FAIRNESS['thresholds'],
            output_path=dirs['reports_fairness'] / "rf_fairness_report.md",
            logger=logger
        )
        
        # GRU fairness report
        gru_report = generate_fairness_report(
            fairness_results=gru_fairness,
            thresholds=FAIRNESS['thresholds'],
            output_path=dirs['reports_fairness'] / "gru_fairness_report.md",
            logger=logger
        )
        
        # Ensemble fairness report
        ensemble_report = generate_fairness_report(
            fairness_results=ensemble_fairness,
            thresholds=FAIRNESS['thresholds'],
            output_path=dirs['reports_fairness'] / "ensemble_fairness_report.md",
            logger=logger
        )
        
        # Analyze bias patterns
        logger.info("Analyzing bias patterns...")
        bias_analysis = analyze_bias_patterns(
            ensemble_fairness,
            feature_importance=rf_results['feature_importance'],
            demographic_impact=feature_results['demographic_impact'],
            logger=logger
        )
        
        # Save bias analysis
        bias_report_path = dirs['reports_fairness'] / "bias_analysis_report.json"
        with open(bias_report_path, 'w') as f:
            json.dump(bias_analysis, f, cls=NumpyJSONEncoder, indent=2)
        logger.info(f"Bias analysis saved to {bias_report_path}")
        
        # Return results
        return {
            'rf__fairness': rf_fairness,
            'gru_fairness': gru_fairness,
            'ensemble_fairness': ensemble_fairness,
            'intersectional_fairness': intersectional_fairness,
            'bias_analysis': bias_analysis,
            'reports': {
                'rf': rf_report,
                'gru': gru_report,
                'ensemble': ensemble_report
            },
            'predictions': {
                'rf_probs': rf_probs,
                'rf_preds': rf_preds,
                'gru_probs': gru_probs,
                'gru_preds': gru_preds,
                'ensemble_probs': ensemble_probs,
                'ensemble_preds': ensemble_preds
            }
        }
        
    except Exception as e:  
        logger.error(f"Fairness evaluation workflow failed: {str(e)}")
        raise


def run_pipeline(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the complete EduPredict 2.0 pipeline with enhanced logging and error handling.
    """
    try:
        # Initialize directory structure
        dirs, logger = setup_environment(args)
        
        # Process data
        if args.mode == 'processing':
            data_results = run_data_processing_workflow(args, dirs, logger)
            # Skip other phases but create minimal report
            run_reporting_pipeline(
                data_results=data_results,
                feature_results={},  # Empty since feature engineering not run yet
                model_results={},    # Empty since models not trained yet
                fairness_results={}, # Empty since fairness analysis not run yet
                dirs=dirs,
                args=args,
                logger=logger
            )
            return

        # Continue with feature engineering if not just processing
        if args.mode in ['feature', 'training', 'evaluation', 'complete']:
            data_results = run_data_processing_workflow(args, dirs, logger)
            feature_results = run_feature_engineering(
                data_results, dirs, args, logger
            )
            
            if args.mode == 'feature':
                run_reporting_pipeline(data_results, feature_results, {}, {}, dirs, args, logger)
                return
                
        # Continue with model training if applicable
        if args.mode in ['training', 'evaluation', 'complete']:
            model_results = run_random_forest_pipeline(
                feature_results, dirs, args, logger
            )
            
            if args.mode == 'training':
                run_reporting_pipeline(data_results, feature_results, model_results, {}, dirs, args, logger)
                return
                
        # Run GRU model pipeline if applicable
        if args.mode in ['gru', 'complete']:
            gru_results = run_gru_pipeline(
                feature_results, dirs, args, logger
            )
            if args.mode == 'gru':
                run_reporting_pipeline(data_results, feature_results, {}, {}, dirs, args, logger)
                return

        # Run ensemble integration if applicable
        if args.mode in ['ensemble', 'complete']:
            ensemble_results = run_ensemble_integration(
                model_results, gru_results, feature_results, dirs, args, logger
            )
            if args.mode == 'ensemble':
                run_reporting_pipeline(data_results, feature_results, {}, {}, dirs, args, logger)
                return

        # Run evaluation if applicable
        if args.mode in ['evaluation', 'complete']:
            fairness_results = run_fairness_pipeline(
                model_results, feature_results, dirs, args, logger
            )
            run_reporting_pipeline(data_results, feature_results, model_results, fairness_results, dirs, args, logger)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

def main():
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize logger
        logger = setup_logger(LOG_DIR, LOG_LEVEL)
        
        # Validate configuration
        if not validate_configuration():
            raise ValueError("Configuration validation failed")
        
        # Run pipeline
        run_pipeline(args, logger)
        
        logger.info("Pipeline execution completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())