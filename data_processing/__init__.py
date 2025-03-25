# re-export functions from data_processing.py
from .data_processing import (
    load_raw_datasets,
    clean_demographic_data,
    clean_vle_data,
    clean_assessment_data,
    validate_data_consistency
)

# re-export functions from feature_engineering.py
from .feature_engineering import (
    create_demographic_features,
    create_temporal_features,
    create_assessment_features,
    create_sequential_features,
    prepare_dual_path_features,
    create_stratified_splits,
    prepare_target_variable
)