#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Configuration parameters for the EduPredict system."""

import os

# get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# data paths and directories with absolute paths
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'OULAD')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')


# create directory structure
def create_directories():
    """create necessary directories for output files"""
    directories = {
        'output_dir': OUTPUT_DIR,
        'model_dir': MODEL_DIR,
        'viz_dir': os.path.join(OUTPUT_DIR, 'visualizations'),
        'report_dir': os.path.join(OUTPUT_DIR, 'reports'),
        'feature_dir': os.path.join(OUTPUT_DIR, 'feature_data'),
        'tableau_dir': os.path.join(OUTPUT_DIR, 'tableau_data')
    }
    
    for path in directories.values():
        os.makedirs(path, exist_ok=True)
        
    return directories

# data processing parameters
RANDOM_SEED = 0
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# feature engineering parameters
WINDOW_SIZES = [7, 14, 30]  # weekly, bi-weekly, monthly
CORRELATION_THRESHOLD = 0.85
IMPORTANCE_THRESHOLD = 0.01

# random forest parameters
RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

RF_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# gru model parameters
GRU_PARAM_GRID = {
    'gru_units': [32, 64, 128],
    'dense_units': [[32], [64], [32, 16]],
    'dropout_rate': [0.2, 0.3, 0.5],
    'learning_rate': [0.001, 0.0005]
}

GRU_DEFAULT_PARAMS = {
    'gru_units': 64,
    'dense_units': [32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'max_seq_length': 100
}

# training parameters
MAX_SEQ_LENGTH = 100
GRU_EPOCHS = 30
GRU_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5

# Protected attributes for fairness analysis
PROTECTED_ATTRIBUTES = {
    'gender': {
        'name': 'gender',
        'values': ['f', 'm'],
        'sensitive': True
    },
    'age_band': {
        'name': 'age_band',
        'values': ['0-35', '35-55', '55<='],
        'sensitive': True
    },
    'imd_band': {
        'name': 'imd_band',
        'values': ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
        'sensitive': True
    }
}

# Fairness thresholds for model evaluation
FAIRNESS_THRESHOLDS = {
    'demographic_parity_difference': 0.05,
    'disparate_impact_ratio': 0.8,
    'equal_opportunity_difference': 0.05
}

# demographic columns for visualization
DEMOGRAPHIC_COLS = ['gender', 'age_band', 'imd_band', 'region', 'highest_education']