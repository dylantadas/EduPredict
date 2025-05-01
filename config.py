import os
import hashlib
import json
from typing import Dict, List
import pandas as pd
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.absolute()
DATA_PATH = BASE_DIR / "data" / "OULAD"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
LOG_LEVEL = "INFO"

# Directory structure
DIRS = {
    "model_checkpoints": MODEL_DIR / "checkpoints",
    "model_final": MODEL_DIR / "final",
    "model_ensemble": MODEL_DIR / "ensemble",
    "reports": OUTPUT_DIR / "reports",
    "reports_performance": OUTPUT_DIR / "reports/model_performance",
    "reports_fairness": OUTPUT_DIR / "reports/fairness",
    "intermediate": OUTPUT_DIR / "intermediate",
    "processed_data": OUTPUT_DIR / "intermediate/processed_data",
    "features": OUTPUT_DIR / "intermediate/features",
    "predictions": OUTPUT_DIR / "intermediate/predictions",
    "visualizations": OUTPUT_DIR / "visualizations",
    "viz_performance": OUTPUT_DIR / "visualizations/performance",
    "viz_fairness": OUTPUT_DIR / "visualizations/fairness",
    "viz_features": OUTPUT_DIR / "visualizations/features",
    "tableau": OUTPUT_DIR / "tableau",
    "tableau_data": OUTPUT_DIR / "tableau/data",
    "tableau_workbooks": OUTPUT_DIR / "tableau/workbooks"
}

# Data processing configuration
DATA_PROCESSING = {
    "chunk_size": 10000,
    "dtypes": {
        "id_student": "int32",
        "code_module": "category",
        "code_presentation": "category",
        "gender": "category"
    },
    "missing_value_strategy": "median"
}

# Feature engineering parameters
FEATURE_ENGINEERING = {
    "window_sizes": [7, 14, 30],
    "correlation_threshold": 0.85,
    "importance_threshold": 0.01
}

# Model parameters
RANDOM_SEED = 0
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

RF_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_SEED
}

GRU_CONFIG = {
    'gru_units': 64,
    'dense_units': [32, 16],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'max_seq_length': 50,
    'epochs': 100,
    'batch_size': 32,
    'early_stopping_patience': 5
}

GRU_PARAM_GRID = {
    'gru_units': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0001]
}

# Fairness parameters
FAIRNESS = {
    'threshold': 0.1,
    'thresholds': {
        'demographic_parity_difference': 0.1,
        'disparate_impact_ratio': 0.8,
        'equal_opportunity_difference': 0.1
    },
    'protected_attributes': ['gender', 'region', 'age_band'],
    'demographic_cols': ['gender', 'region', 'age_band', 'disability'],
    'min_group_size': 50
}

# Protected attributes mapping
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
        'values': ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                  '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
        'sensitive': True
    }
}

# Bias mitigation parameters
BIAS_MITIGATION = {
    'method': 'reweight',  # Options: 'reweight', 'oversample', 'undersample', 'none'
    'balance_strategy': 'group_balanced',  # Options: 'group_balanced', 'stratified'
    'target_ratios': None,
    'min_group_size': 50,
    'max_ratio': 3.0
}

# Evaluation parameters
EVALUATION = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'cv_folds': 5,
    'stratify_cols': ['gender', 'region']
}

# Version control parameters
VERSION_CONTROL = {
    'enable_data_versioning': True,
    'version_format': 'v{major}.{minor}.{patch}',
    'metadata_fields': ['timestamp', 'features', 'processing_params']
}

def create_directories() -> Dict[str, str]:
    """Creates necessary directories for output files"""
    created_dirs = {}
    for name, path in DIRS.items():
        os.makedirs(path, exist_ok=True)
        created_dirs[name] = str(path)
    return created_dirs

def validate_config() -> bool:
    """Validates configuration parameters for consistency"""
    try:
        # Check if all directories are properly defined
        assert all(isinstance(path, Path) for path in DIRS.values())
        
        # Validate model parameters
        assert 0 < TEST_SIZE < 1
        assert 0 < VALIDATION_SIZE < 1
        assert TEST_SIZE + VALIDATION_SIZE < 1
        
        # Validate GRU configuration
        assert all(isinstance(x, int) for x in GRU_CONFIG['dense_units'])
        assert 0 < GRU_CONFIG['dropout_rate'] < 1
        assert GRU_CONFIG['learning_rate'] > 0
        
        # Validate fairness thresholds
        assert 0 <= FAIRNESS['threshold'] <= 1
        assert all(0 <= v <= 1 for v in FAIRNESS['thresholds'].values())
        
        return True
    except AssertionError:
        return False

def get_data_version(data_dict: Dict[str, pd.DataFrame]) -> str:
    """Generates a version hash for processed datasets"""
    version_data = []
    for name, df in sorted(data_dict.items()):
        # Create a fingerprint of the dataframe
        df_hash = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()
        version_data.append(f"{name}:{df_hash}")
    
    # Combine all fingerprints and create final hash
    combined_hash = hashlib.md5(
        json.dumps(version_data, sort_keys=True).encode()
    ).hexdigest()
    
    return combined_hash[:8]  # Return first 8 characters as version