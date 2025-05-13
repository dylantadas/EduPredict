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
from feature_engineering.sequential_features import (
    create_sequential_features,
    SequentialFeatureProcessor
)
from feature_engineering.temporal_features import create_temporal_features
from feature_engineering.feature_selector import (
    FeatureSelector,
    analyze_feature_importance,
    analyze_feature_correlations
)

# Model imports
from model_implementation.random_forest_model import RandomForestModel
from model_implementation.gru_model import (
    GRUModel,
    tune_gru_model,
    find_optimal_threshold
)
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
from evaluation.evaluation_report import generate_fairness_report, run_reporting_pipeline, generate_evaluation_report, generate_performance_report, compare_model_versions
from evaluation.cross_validation import perform_cross_validation
from evaluation.bias_mitigation import apply_reweighting
from evaluation.model_explainer import generate_global_explanations, create_feature_impact_visualizations

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
    GRU_CONFIG, RF_CONFIG
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
    
    parser.add_argument(
        '--use-full-dataset',
        action='store_true',
        help='Use the full dataset instead of sampling'
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
        
        # Apply fairness-aware sampling by default unless full dataset is requested
        if not args.use_full_dataset:
            sample_size = DATA_PROCESSING.get('sample_size', 5000)
            logger.info(f"Using fairness-aware sampling with sample size: {sample_size}")
            
            # Create balanced sample using protected attributes
            from data_processing.data_loader import create_fairness_aware_sample
            datasets = create_fairness_aware_sample(
                datasets,
                sample_size=sample_size,
                random_state=RANDOM_SEED,
                logger=logger
            )
            logger.info(f"Using sampled dataset with {len(datasets['student_info'])} students")
        else:
            logger.info(f"Using full dataset with {len(datasets['student_info'])} students")
        
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
    """Run feature engineering workflow with proper sequence processing"""
    try:
        # Get processed data directory and splits
        processed_dir = dirs['processed_data']
        clean_data = data_results.get('clean_data', {})
        splits = data_results.get('splits', {})

        if not clean_data:
            logger.info("Loading cleaned data from disk...")
            clean_data = {
                'demographics': pd.read_parquet(processed_dir / "demographics.parquet"),
                'vle': pd.read_parquet(processed_dir / "vle.parquet"),
                'assessments': pd.read_parquet(processed_dir / "assessments.parquet"),
                'submissions': pd.read_parquet(processed_dir / "submissions.parquet"),
                'registration': pd.read_parquet(processed_dir / "registration.parquet")
            }

        # Get student IDs for each split
        student_ids = {split: df['id_student'].unique() 
                      for split, df in splits.items()}

        # Create demographic features
        logger.info("Creating demographic features...")
        static_features = create_demographic_features(
            clean_data['demographics'],
            categorical_cols=FEATURE_ENGINEERING['categorical_cols'],
            one_hot=FEATURE_ENGINEERING.get('one_hot_encoding', True),
            logger=logger
        )

        # Create temporal features
        logger.info("Creating temporal features...")
        temporal_features = create_temporal_features(
            clean_data['vle'],
            student_ids=student_ids,
            window_sizes=FEATURE_ENGINEERING['window_sizes'],
            module_info=clean_data.get('registration', None),
            logger=logger
        )

        # Create sequential features with proper configuration
        logger.info("Creating sequential features...")
        seq_config = FEATURE_ENGINEERING['sequential_processing']
        
        # Initialize SequentialFeatureProcessor with configuration
        processor = SequentialFeatureProcessor(
            sequence_length=seq_config['max_seq_length'],
            padding=seq_config['padding_strategy'],
            truncating=seq_config['truncating_strategy'],
            normalize=seq_config['normalize_sequences']
        )

        # Process sequential features
        sequential_features = create_sequential_features(
            vle_data=clean_data['vle'],
            submission_data=clean_data['submissions'],
            student_ids=student_ids,
            max_seq_length=seq_config['max_seq_length'],
            categorical_cols=seq_config['feature_columns'],
            logger=logger,
            courses_df=clean_data.get('registration', None),
            vle_materials_df=clean_data.get('vle_materials', None)
        )

        # Merge temporal features with static features
        logger.info("Merging features...")
        combined_features = {}
        
        for split_name in ['train', 'validation', 'test']:
            # Get base features for this split
            split_mask = static_features['id_student'].isin(student_ids[split_name])
            combined_features[split_name] = static_features[split_mask].copy()
            
            # Add temporal features if available
            if split_name in temporal_features:
                temporal_split = temporal_features[split_name]
                combined_features[split_name] = combined_features[split_name].merge(
                    temporal_split,
                    on='id_student',
                    how='left'
                )
            
            # Fill missing values appropriately
            combined_features[split_name] = combined_features[split_name].fillna(0)
            
            logger.info(f"{split_name} set shape: {combined_features[split_name].shape}")

        # Initialize feature selector
        feature_selector = FeatureSelector(
            method='importance',
            threshold=FEATURE_ENGINEERING['min_importance'],
            random_state=RANDOM_SEED
        )

        # Select features using training data
        selected_features = {}
        feature_names = None
        
        for split_name in ['train', 'validation', 'test']:
            split_features = combined_features[split_name]
            
            if split_name == 'train':
                # Fit selector on training data
                target_col = 'final_result'
                feature_cols = [col for col in split_features.columns 
                              if col not in ['id_student', 'final_result', 'split']]
                
                X = split_features[feature_cols]
                y = split_features[target_col]
                
                selected_features[split_name] = feature_selector.fit_transform(
                    X, y,
                    protected_attributes=list(PROTECTED_ATTRIBUTES.keys())
                )
                feature_names = feature_selector.selected_features_
            else:
                # Transform other splits using fitted selector
                selected_features[split_name] = feature_selector.transform(
                    split_features[feature_cols]
                )

        # Save feature metadata
        logger.info("Saving feature metadata...")
        metadata_dir = dirs['feature_metadata']
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            'static_features': {
                'categorical_cols': FEATURE_ENGINEERING['categorical_cols'],
                'numeric_cols': FEATURE_ENGINEERING['numeric_cols'],
                'selected_features': feature_names
            },
            'temporal_features': {
                'window_sizes': FEATURE_ENGINEERING['window_sizes'],
                'statistics': temporal_features.get('statistics', {})
            },
            'sequential_features': {
                'config': seq_config,
                'feature_mapping': sequential_features.get('train', {}).get('metadata', {}).get('feature_mapping', {}),
                'sequence_stats': sequential_features.get('train', {}).get('metadata', {}).get('sequence_stats', {})
            },
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'selected_feature_count': len(feature_names) if feature_names else 0
            }
        }

        with open(metadata_dir / 'feature_engineering_metadata.json', 'w') as f:
            json.dump(metadata, f, cls=NumpyJSONEncoder, indent=2)

        # Save processed features
        logger.info("Saving processed features...")
        feature_dir = dirs['features']
        
        for split_name in ['train', 'validation', 'test']:
            # Save static/temporal features
            static_path = feature_dir / f"{split_name}_static_features.parquet"
            selected_features[split_name].to_parquet(static_path)
            
            # Save sequential features if available
            if split_name in sequential_features:
                seq_path = feature_dir / f"{split_name}_sequences.npz"
                np.savez_compressed(
                    seq_path,
                    sequences=sequential_features[split_name]['sequence_data'],
                    sequence_lengths=sequential_features[split_name]['sequence_lengths'],
                    student_ids=sequential_features[split_name]['student_ids'],
                    targets=sequential_features[split_name].get('targets', []),
                    auxiliary_features=sequential_features[split_name].get('auxiliary_features', {})
                )

        logger.info("Feature engineering complete")
        return {
            'static_features': selected_features,
            'sequential_features': sequential_features,
            'feature_metadata': metadata,
            'feature_selector': feature_selector
        }

    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
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
            
            # Plot training history using GRUModel's built-in method
            gru_model._plot_training_history(train_history.history)
            
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
            
            # Visualize attention weights if the model uses attention
            if GRU_CONFIG.get('use_attention', True):
                # Create attention visualization using model's internal data
                test_attention = attention_weights['test'][:5]  # First 5 test samples
                test_lengths = seq_data['test']['lengths'][:5]
                
                # Create figure for attention visualization
                fig, axes = plt.subplots(len(test_attention), 1, figsize=(15, 3*len(test_attention)))
                if len(test_attention) == 1:
                    axes = [axes]
                else:
                    axes = np.array(axes).flatten()
                
                for idx, (attn, seq_len) in zip(range(len(test_attention)), zip(test_attention, test_lengths)):
                    # Plot attention weights for this sequence
                    axes[idx].imshow(attn[:seq_len].reshape(1, -1), cmap='hot', aspect='auto')
                    axes[idx].set_title(f'Sequence {idx+1} Attention Weights')
                    axes[idx].set_xlabel('Timestep')
                
                plt.tight_layout()
                attn_path = dirs['visualizations'] / "gru_attention_weights.png"
                plt.savefig(attn_path)
                plt.close()

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
    Execute the ensemble model integration pipeline.
    
    Args:
        rf_results: Results from Random Forest pipeline
        gru_results: Results from GRU pipeline
        feature_results: Results from feature engineering
        dirs: Directory paths
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        Dictionary containing ensemble results
    """
    try:
        logger.info("Starting ensemble integration pipeline...")
        
        # Initialize results dictionary
        ensemble_results = {
            'model': None,
            'predictions': {},
            'thresholds': {},
            'metrics': {},
            'weights': None
        }
        
        # Extract predictions from both models
        rf_predictions = rf_results['predictions']
        gru_predictions = gru_results['predictions']
        
        # Initialize ensemble combiner
        ensemble = PredictionCombiner(
            method='weighted_average',
            threshold=0.5,  # Initial threshold, will be optimized
            fairness_constraints=FAIRNESS['thresholds'] if args.fairness_aware else None,
            random_state=RANDOM_SEED,
            logger=logger
        )
        
        # Optimize ensemble weights using validation set
        logger.info("Optimizing ensemble weights...")
        optimal_weights = ensemble.optimize_weights(
            rf_proba=rf_predictions['validation']['pred_proba'],
            gru_proba=gru_predictions['validation']['pred_proba'],
            true_labels=rf_predictions['validation']['true_label'],
            sensitive_features=feature_results['static_features']['validation'][list(PROTECTED_ATTRIBUTES.keys())] if args.fairness_aware else None
        )
        
        logger.info(f"Optimal ensemble weights: {optimal_weights}")
        
        # Generate ensemble predictions for all splits
        predictions = {}
        metrics = {}
        
        for split in ['train', 'validation', 'test']:
            # Combine predictions
            combined_proba = ensemble.combine_predictions(
                rf_proba=rf_predictions[split]['pred_proba'],
                gru_proba=gru_predictions[split]['pred_proba'],
                sensitive_features=feature_results['static_features'][split][list(PROTECTED_ATTRIBUTES.keys())] if args.fairness_aware else None
            )
            
            # Create prediction DataFrame
            pred_df = pd.DataFrame({
                'id_student': rf_predictions[split]['id_student'],
                'true_label': rf_predictions[split]['true_label'],
                'pred_proba': combined_proba
            })
            
            # Calculate metrics
            split_metrics = calculate_model_metrics(
                pred_df['true_label'],
                (pred_df['pred_proba'] >= ensemble.threshold).astype(int),
                pred_df['pred_proba'],
                logger=logger
            )
            
            predictions[split] = pred_df
            metrics[split] = split_metrics
            
            logger.info(f"{split} set ensemble metrics: {split_metrics}")
        
        # Generate visualizations
        if args.export_visualizations:
            logger.info("Generating ensemble visualizations...")
            
            # Plot ROC curves
            roc_fig = plot_roc_curves(
                {split: pred_df for split, pred_df in predictions.items()},
                title="Ensemble ROC Curves"
            )
            roc_path = dirs['visualizations'] / "ensemble_roc_curves.png"
            roc_fig.savefig(roc_path)
            plt.close(roc_fig)
            
            # Plot precision-recall curves
            pr_fig = plot_precision_recall_curves(
                {split: pred_df for split, pred_df in predictions.items()},
                title="Ensemble Precision-Recall Curves"
            )
            pr_path = dirs['visualizations'] / "ensemble_precision_recall_curves.png"
            pr_fig.savefig(pr_path)
            plt.close(pr_fig)
            
            # Visualize ensemble weights
            weight_fig = visualize_ensemble_weights(optimal_weights)
            weight_path = dirs['visualizations'] / "ensemble_weights.png"
            weight_fig.savefig(weight_path)
            plt.close(weight_fig)
        
        # Save ensemble configuration
        logger.info("Saving ensemble configuration...")
        ensemble_dir = dirs['models'] / "ensemble"
        ensemble_dir.mkdir(exist_ok=True, parents=True)
        
        # Save metadata
        metadata = {
            'weights': optimal_weights,
            'threshold': ensemble.threshold,
            'fairness_constraints': ensemble.fairness_constraints,
            'metrics': metrics,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = ensemble_dir / "ensemble_config.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, cls=NumpyJSONEncoder, indent=2)
        
        # Store results
        ensemble_results = {
            'model': ensemble,
            'predictions': predictions,
            'metrics': metrics,
            'weights': optimal_weights
        }
        
        return ensemble_results
        
    except Exception as e:
        logger.error(f"Ensemble integration pipeline failed: {str(e)}")
        raise

@track_execution_time
def run_evaluation_pipeline(
    rf_results: Dict[str, Any],
    gru_results: Dict[str, Any],
    ensemble_results: Dict[str, Any],
    feature_results: Dict[str, Any],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Runs comprehensive model evaluation including explanations."""
    
    evaluation_results = {}
    
    try:
        # Get test data
        X_test_rf = rf_results.get('test_features')
        test_sequences = gru_results.get('test_sequences')
        y_test = rf_results.get('test_labels')
        
        # Generate model explanations
        logger.info("Generating model explanations...")
        
        # RF model explanations
        rf_explanations = generate_global_explanations(
            model=rf_results['model'],
            X=X_test_rf,
            feature_names=X_test_rf.columns.tolist()
        )
        
        # Create visualization paths
        rf_viz_path = dirs['visualizations'] / 'rf_explanations'
        rf_viz_path.mkdir(exist_ok=True)
        
        # Generate RF visualizations
        rf_viz_paths = create_feature_impact_visualizations(
            rf_explanations,
            str(rf_viz_path / 'rf')
        )
        
        # GRU model explanations if available
        gru_explanations = None
        gru_viz_paths = []
        if hasattr(gru_results.get('model'), 'get_attention_weights'):
            logger.info("Generating GRU attention-based explanations...")
            gru_viz_path = dirs['visualizations'] / 'gru_explanations'
            gru_viz_path.mkdir(exist_ok=True)
            
            # Generate attention-based explanations
            attention_weights = gru_results['model'].get_attention_weights(test_sequences)
            gru_explanations = {
                'attention_weights': attention_weights,
                'timesteps': list(range(len(attention_weights[0]))),
                'features': gru_results.get('feature_names', [])
            }
            
            # Generate GRU visualizations
            gru_viz_paths = create_feature_impact_visualizations(
                gru_explanations,
                str(gru_viz_path / 'gru')
            )
        
        # Store explanations
        evaluation_results['explanations'] = {
            'rf': {
                'global_explanations': rf_explanations,
                'visualization_paths': rf_viz_paths
            }
        }
        if gru_explanations:
            evaluation_results['explanations']['gru'] = {
                'attention_explanations': gru_explanations,
                'visualization_paths': gru_viz_paths
            }
        
        # Generate individual model performance reports
        logger.info("Generating individual model performance reports...")
        
        if rf_results.get('metrics'):
            rf_perf_path = dirs['reports'] / 'rf_performance_report.md'
            generate_performance_report(rf_results['metrics'], str(rf_perf_path))
            evaluation_results['rf_performance_report'] = str(rf_perf_path)
            
        if gru_results.get('metrics'):
            gru_perf_path = dirs['reports'] / 'gru_performance_report.md'
            generate_performance_report(gru_results['metrics'], str(gru_perf_path))
            evaluation_results['gru_performance_report'] = str(gru_perf_path)
            
        if ensemble_results.get('metrics'):
            ensemble_perf_path = dirs['reports'] / 'ensemble_performance_report.md'
            generate_performance_report(ensemble_results['metrics'], str(ensemble_perf_path))
            evaluation_results['ensemble_performance_report'] = str(ensemble_perf_path)
        
        # Generate fairness reports if fairness analysis was performed
        if args.fairness_aware:
            logger.info("Generating fairness reports...")
            
            for model_name, results in [
                ('rf', rf_results),
                ('gru', gru_results),
                ('ensemble', ensemble_results)
            ]:
                if results.get('fairness_metrics'):
                    fairness_path = dirs['reports'] / f'{model_name}_fairness_report.md'
                    generate_fairness_report(
                        results['fairness_metrics'],
                        thresholds=FAIRNESS['thresholds'],
                        output_path=str(fairness_path)
                    )
                    evaluation_results[f'{model_name}_fairness_report'] = str(fairness_path)
        
        # Compare model versions if multiple versions exist
        if all(results.get('metrics') for results in [rf_results, gru_results, ensemble_results]):
            logger.info("Comparing model versions...")
            comparison_df = compare_model_versions(
                metrics_list=[
                    rf_results['metrics'],
                    gru_results['metrics'],
                    ensemble_results['metrics']
                ],
                model_names=['Random Forest', 'GRU', 'Ensemble']
            )
            
            # Save comparison results
            comparison_path = dirs['reports'] / 'model_comparison.csv'
            comparison_df.to_csv(comparison_path)
            evaluation_results['model_comparison'] = str(comparison_path)
        
        # Generate comprehensive evaluation report
        logger.info("Generating comprehensive evaluation report...")
        eval_report_path = dirs['reports'] / 'model_evaluation.md'
        evaluation_results['evaluation_report'] = generate_evaluation_report(
            rf_results=rf_results,
            gru_results=gru_results,
            ensemble_results=ensemble_results,
            explanations=evaluation_results['explanations'],
            output_path=str(eval_report_path),
            logger=logger
        )
        
        logger.info("Model evaluation pipeline completed")
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")
        raise
    
    return evaluation_results

def run_pipeline(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the complete EduPredict 2.0 pipeline with enhanced logging and error handling.
    Pipeline modes:
    - 'processing': Only run data processing
    - 'feature': Run processing and feature engineering
    - 'rf': Run through Random Forest model training
    - 'gru': Run through GRU model training
    - 'ensemble': Run full pipeline including ensemble integration
    - 'evaluation': Run full pipeline with evaluation
    - 'full': Run everything including evaluation and reporting
    """
    try:
        # Initialize directory structure and results dictionaries
        dirs, logger = setup_environment(args)
        data_results = {}
        feature_results = {}
        rf_results = {}
        gru_results = {}
        ensemble_results = {}
        evaluation_results = {}

        # Data Processing Stage (required for all modes except when loading processed data)
        if not args.load_processed:
            logger.info("Starting data processing pipeline...")
            data_results = run_data_processing_workflow(args, dirs, logger)
        else:
            logger.info("Loading pre-processed data...")
            processed_dir = dirs['processed_data']
            
            # Load pre-processed data
            demographics = pd.read_parquet(processed_dir / "demographics.parquet")
            vle = pd.read_parquet(processed_dir / "vle.parquet")
            assessments = pd.read_parquet(processed_dir / "assessments.parquet")
            submissions = pd.read_parquet(processed_dir / "submissions.parquet")
            registration = pd.read_parquet(processed_dir / "registration.parquet")
            
            # Structure data_results to match run_data_processing_workflow output
            data_results = {
                'clean_data': {
                    'demographics': demographics,
                    'vle': vle,
                    'assessments': assessments,
                    'submissions': submissions,
                    'registration': registration
                },
                'splits': {
                    'train': demographics[demographics['split'] == 'train'],
                    'validation': demographics[demographics['split'] == 'validation'],
                    'test': demographics[demographics['split'] == 'test']
                }
            }
            logger.info("Successfully loaded pre-processed data")

        if args.mode == 'processing':
            logger.info("Processing mode completed")
            run_reporting_pipeline(data_results, {}, {}, {}, dirs, args, logger)
            return

        # Feature Engineering Stage
        if not args.load_features:
            logger.info("Starting feature engineering pipeline...")
            feature_results = run_feature_engineering(data_results, dirs, args, logger)
        else:
            logger.info("Loading pre-engineered features...")
            feature_dir = dirs['feature_data']
            # Load pre-engineered features
            feature_results = {
                'static_features': {},
                'sequential_features': {},
                'feature_metadata': {}
            }
            for split in ['train', 'validation', 'test']:
                feature_results['static_features'][split] = pd.read_parquet(
                    feature_dir / f"{split}_static_features.parquet"
                )
                if os.path.exists(feature_dir / f"{split}_sequences.npz"):
                    seq_data = np.load(feature_dir / f"{split}_sequences.npz", allow_pickle=True)
                    feature_results['sequential_features'][split] = {
                        'sequence_data': seq_data['sequences'],
                        'student_ids': seq_data['student_ids'],
                        'targets': seq_data['targets']
                    }

        if args.mode == 'feature':
            logger.info("Feature engineering mode completed")
            run_reporting_pipeline(data_results, feature_results, {}, {}, dirs, args, logger)
            return

        # Random Forest Model Stage
        if args.mode in ['rf', 'ensemble', 'evaluation', 'full']:
            logger.info("Starting Random Forest pipeline...")
            rf_results = run_random_forest_pipeline(feature_results, dirs, args, logger)

        if args.mode == 'rf':
            logger.info("Random Forest mode completed")
            run_reporting_pipeline(data_results, feature_results, {'rf': rf_results}, {}, dirs, args, logger)
            return

        # GRU Model Stage
        if args.mode in ['gru', 'ensemble', 'evaluation', 'full']:
            logger.info("Starting GRU pipeline...")
            gru_results = run_gru_pipeline(feature_results, dirs, args, logger)

        if args.mode == 'gru':
            logger.info("GRU mode completed")
            run_reporting_pipeline(data_results, feature_results, {'gru': gru_results}, {}, dirs, args, logger)
            return

        # Ensemble Integration Stage
        if args.mode in ['ensemble', 'evaluation', 'full']:
            logger.info("Starting ensemble integration...")
            if not rf_results or not gru_results:
                raise ValueError("Both RF and GRU results required for ensemble integration")
            ensemble_results = run_ensemble_integration(
                rf_results, gru_results, feature_results, dirs, args, logger
            )

        if args.mode == 'ensemble':
            logger.info("Ensemble mode completed")
            run_reporting_pipeline(
                data_results, 
                feature_results,
                {
                    'rf': rf_results, 
                    'gru': gru_results, 
                    'ensemble': ensemble_results
                },
                {},
                dirs, args, logger
            )
            return

        # Evaluation Stage
        if args.mode in ['evaluation', 'full']:
            logger.info("Starting evaluation pipeline...")
            if not ensemble_results:
                raise ValueError("Ensemble results required for evaluation")
            evaluation_results = run_evaluation_pipeline(
                rf_results, gru_results, ensemble_results, feature_results, dirs, args, logger
            )

        # For evaluation or full mode, run complete reporting
        if args.mode in ['evaluation', 'full']:
            logger.info("Generating final reports...")
            run_reporting_pipeline(
                data_results,
                feature_results,
                {
                    'rf': rf_results,
                    'gru': gru_results,
                    'ensemble': ensemble_results
                },
                evaluation_results,
                dirs,
                args,
                logger
            )

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
    exit_code = main()
    sys.exit(exit_code)