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
    "feature_metadata": OUTPUT_DIR / "intermediate/features/metadata",
    "predictions": OUTPUT_DIR / "intermediate/predictions",
    "visualizations": OUTPUT_DIR / "visualizations",
    "viz_performance": OUTPUT_DIR / "visualizations/performance",
    "viz_fairness": OUTPUT_DIR / "visualizations/fairness",
    "viz_features": OUTPUT_DIR / "visualizations/features",
    "tableau": OUTPUT_DIR / "tableau",
    "tableau_data": OUTPUT_DIR / "tableau/data",
    "tableau_workbooks": OUTPUT_DIR / "tableau/workbooks",
    "logs": LOG_DIR
}

# Data processing configuration
DATA_PROCESSING = {
    "chunk_size": 10000,
    "dtypes": {
        "id_student": "int32",
        "code_module": "category",
        "code_presentation": "category",
        "gender": "category",
        "age_band": "category",
        "imd_band": "category",
        "region": "category",
        "highest_education": "category",
        "disability": "category",
        "sum_click": "int32",
        "date": "float32",           # Days from module start
        "date_registration": "float32",  # Days from module start
        "date_unregistration": "float32",  # Days from module start
        "date_submitted": "float32",  # Days from module start
        "score": "float32"
    },
    "missing_value_strategy": "median",
    "time_handling": {
        "negative_dates_allowed": True,  # Allow dates before module start
        "max_negative_days": -60,  # Maximum days before module start
        "max_positive_days": 180   # Maximum days after module start
    }
}

# Timing and temporal analysis configuration
TEMPORAL_CONFIG = {
    'session_gap_days': 0.125,  # 3 hours between sessions
    'module_length_days': 180,  # Typical module duration
    'registration_windows': {
        'very_early': [-float('inf'), -30],
        'early': [-30, -7],
        'on_time': [-7, 0],
        'late': [0, float('inf')]
    },
    'activity_windows': {
        'pre_module': [-30, 0],  # Allow up to 30 days pre-module activity
        'early_phase': [0, 60],  # First third of module
        'mid_phase': [60, 120],  # Middle third
        'late_phase': [120, 180]  # Final third
    },
    'padding_strategy': 'pre',  # Strategy for padding sequences ('pre' or 'post')
    'truncating_strategy': 'pre',  # Strategy for truncating sequences ('pre' or 'post')
    'normalize_sequences': True,  # Whether to normalize sequence values
    'assessment_timing': {
        'max_early_days': 7,  # Maximum days an assessment can be due before module starts
        'min_gap_days': 6,    # Minimum days between assessments
        'late_submission_threshold': 30  # Days after which a submission is considered very late
    }
}

# Module and presentation codes
MODULE_CODES = {
    'name': 'code_module',
    'values': ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'],
    'description': {
        'AAA': 'Social sciences',
        'BBB': 'STEM',
        'CCC': 'STEM',
        'DDD': 'STEM',
        'EEE': 'STEM',
        'FFF': 'STEM',
        'GGG': 'Social sciences'
    }
}

PRESENTATION_CODES = {
    'name': 'code_presentation',
    'values': ['2013B', '2013J', '2014B', '2014J'],
    'semester_map': {
        '2013B': 'Second semester 2013',
        '2013J': 'First semester 2013',
        '2014B': 'Second semester 2014',
        '2014J': 'First semester 2014'
    }
}

# Feature engineering configuration
FEATURE_ENGINEERING = {
    "window_sizes": [7, 14, 30],  # weekly, bi-weekly, monthly
    "max_seq_length": 100,
    "min_importance": 0.01,
    "correlation_threshold": 0.85,
    "categorical_cols": [
        "code_module",
        "code_presentation",
        "gender",
        "age_band",
        "imd_band",
        "region",
        "highest_education",
        "disability"
    ],
    "sequential_categorical_cols": [
        "activity_type",
        "assessment_type"
    ],
    "sequential_processing": {
        "padding_strategy": "pre",
        "truncating_strategy": "pre",
        "normalize_sequences": True,
        "session_gap_days": 0.125,  # 3 hours between sessions
        "max_seq_length": 100,
        "feature_columns": [
            "type",
            "activity_type",
            "is_pre_module",
            "week_number",
            "day_of_week",
            "is_assessment",
            "effective_score"
        ],
        "auxiliary_features": [
            "sequence_length",
            "interaction_count",
            "avg_interactions_per_step",
            "total_clicks",
            "avg_score",
            "submission_count"
        ],
        "vle_activity_weights": {
            "resource": 1.0,
            "url": 1.0,
            "quiz": 2.0,
            "forum": 1.5,
            "oucontent": 1.0,
            "subpage": 1.0,
            "homepage": 0.5,
            "page": 1.0,
            "questionnaire": 1.5,
            "ouelluminate": 2.0,
            "sharedsubpage": 1.0,
            "externalquiz": 2.0,
            "dataplus": 1.5,
            "glossary": 1.0,
            "htmlactivity": 1.5,
            "oucollaborate": 2.0,
            "dualpane": 1.0
        }
    },
    "standardize_numeric": True,  # Added standardization flag
    "numeric_cols": [
        "studied_credits",
        "num_of_prev_attempts",
        "score",
        "weight",
        "date",
        "sum_click"
    ],
    "missing_value_strategy": {
        "numeric": "median",
        "categorical": "mode",
        "datetime": "drop"
    },
    "target_encoding": {
        "column": "final_result",
        "encoding": {
            "Distinction": 0,  # not at risk
            "Pass": 0,        # not at risk
            "Fail": 1,        # at risk
            "Withdrawn": 1    # at risk
        },
        "description": {
            0: "not at risk (Pass/Distinction)",
            1: "at risk (Fail/Withdrawn)"
        }
    },
    "one_hot_encoding": True,
    "activity_windows": {
        "early": (0, 30),
        "mid": (31, 150),
        "late": (151, 240)
    },
    "standardization": {
        "method": "standard",  # Options: 'standard', 'minmax', 'robust'
        "exclude_cols": [
            "id_student",
            "final_result",
            "code_module",
            "code_presentation"
        ],
        "handle_outliers": True,
        "outlier_threshold": 3  # Standard deviations for outlier detection
    },
    "target_handling": {
        "encode_before_standardization": True,
        "exclude_from_standardization": ["final_result"],
        "preserve_columns": ["id_student", "code_module", "code_presentation", "final_result"]
    }
}

# Model parameters
RANDOM_SEED = 0
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

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
    'threshold': 0.1,  # General threshold for fairness metrics
    'thresholds': {
        'demographic_parity_difference': 0.1,
        'disparate_impact_ratio': 0.8,
        'equal_opportunity_difference': 0.1,
        'average_odds_difference': 0.1
    },
    'protected_attributes': [
        'gender', 'age_band', 'imd_band', 'region', 'disability'
    ],
    'demographic_cols': [
        'gender', 'age_band', 'imd_band', 'region', 'disability', 
        'highest_education', 'studied_credits', 'num_of_prev_attempts'
    ],
    'min_group_size': 50,
    'reporting_frequency': 'epoch'  # When to calculate fairness metrics during training
}

# Protected attributes mapping with standardized values
PROTECTED_ATTRIBUTES = {
    'gender': {
        'name': 'gender',
        'values': ['f', 'm'],
        'sensitive': True,
        'balanced_threshold': 0.45  # Minimum representation ratio
    },
    'age_band': {
        'name': 'age_band',
        'values': ['0-35', '35+'],  # Merged 35-55 and 55<= into 35+
        'sensitive': True,
        'balanced_threshold': 0.25
    },
    'imd_band': {
        'name': 'imd_band',
        'values': [
            '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
            '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'
        ],
        'sensitive': True,
        'balanced_threshold': 0.08  # Minimum 8% in any decile
    },
    'region': {
        'name': 'region',
        'values': [
            'east anglian region', 'scotland', 'north region', 
            'south east region', 'south region', 'wales', 
            'west midlands region', 'north western region',
            'south west region', 'east midlands region', 
            'yorkshire region', 'ireland'
        ],
        'sensitive': True,
        'balanced_threshold': 0.05
    },
    'disability': {
        'name': 'disability',
        'values': ['Y', 'N'],
        'sensitive': True,
        'balanced_threshold': 0.1
    }
}

# Demographic value standardization
DEMOGRAPHIC_STANDARDIZATION = {
    'gender': {'F': 'f', 'M': 'm'},
    'region': 'lower',  # Convert to lowercase
    'highest_education': {
        'No Formal quals': 'no_formal',
        'Lower Than A Level': 'below_a_level',
        'A Level or Equivalent': 'a_level',
        'HE Qualification': 'he_qualification',
        'Post Graduate Qualification': 'post_graduate'
    }
}

# Bias mitigation parameters
BIAS_MITIGATION = {
    'method': 'reweight',  # Options: 'reweight', 'oversample', 'undersample', 'none'
    'balance_strategy': 'group_balanced',
    'target_ratios': None,  # Optional custom ratios per group
    'min_group_size': 50,
    'max_ratio': 3.0,  # Maximum allowed ratio between group sizes
    'reweight_options': {
        'weight_clipping': 10.0,  # Maximum instance weight
        'epsilon': 0.01  # Small constant for numerical stability
    }
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