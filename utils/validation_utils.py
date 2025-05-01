from typing import Dict, List, Set, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger('edupredict')

def validate_directories(dirs: Dict[str, Path]) -> bool:
    """
    Validates that all required directories exist and are properly configured.
    
    Args:
        dirs: Dictionary of directory paths
        
    Returns:
        bool: True if all directories are valid
    """
    try:
        # Check if all paths are Path objects
        if not all(isinstance(path, Path) for path in dirs.values()):
            logger.error("Invalid directory configuration: all paths must be Path objects")
            return False
            
        # Check if all required directories exist
        for name, path in dirs.items():
            if not path.exists():
                logger.warning(f"Directory {name} at {path} does not exist")
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory {path}")
                
        return True
    except Exception as e:
        logger.error(f"Error validating directories: {str(e)}")
        return False

def validate_model_parameters(params: Dict[str, Any]) -> bool:
    """
    Validates model configuration parameters.
    
    Args:
        params: Dictionary of model parameters
        
    Returns:
        bool: True if all parameters are valid
    """
    try:
        # Validate test and validation split sizes
        test_size = params.get('TEST_SIZE', 0)
        val_size = params.get('VALIDATION_SIZE', 0)
        
        if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
            logger.error("Invalid test/validation split configuration")
            return False
            
        # Validate random seed
        if not isinstance(params.get('RANDOM_SEED', 0), int):
            logger.error("Random seed must be an integer")
            return False
            
        # Validate feature engineering parameters
        fe_params = params.get('FEATURE_ENGINEERING', {})
        if not (0 < fe_params.get('correlation_threshold', 1) <= 1):
            logger.error("Correlation threshold must be between 0 and 1")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating model parameters: {str(e)}")
        return False

def validate_data_consistency(datasets: Dict[str, pd.DataFrame]) -> bool:
    """
    Validates consistency between different datasets.
    
    Args:
        datasets: Dictionary of pandas DataFrames
        
    Returns:
        bool: True if data is consistent
    """
    try:
        warnings = []
        
        # Get reference sets
        student_ids = set(datasets['student_info']['id_student'])
        
        # Check VLE data consistency
        if 'vle_interactions' in datasets:
            vle_student_ids = set(datasets['vle_interactions']['id_student'])
            if not vle_student_ids.issubset(student_ids):
                warnings.append("VLE interactions contain unknown student IDs")
                
            if 'vle_materials' in datasets:
                vle_ids = set(datasets['vle_materials']['id_site'])
                interaction_vle_ids = set(datasets['vle_interactions']['id_site'])
                if not interaction_vle_ids.issubset(vle_ids):
                    warnings.append("VLE interactions contain unknown activity IDs")
        
        # Check assessment data consistency
        if all(k in datasets for k in ['assessments', 'student_assessments']):
            assessment_ids = set(datasets['assessments']['id_assessment'])
            student_assessment_ids = set(datasets['student_assessments']['id_assessment'])
            if not student_assessment_ids.issubset(assessment_ids):
                warnings.append("Student assessment data contain unknown assessment IDs")
                
            assessment_student_ids = set(datasets['student_assessments']['id_student'])
            if not assessment_student_ids.issubset(student_ids):
                warnings.append("Assessment data contain unknown student IDs")
        
        # Log any warnings
        for warning in warnings:
            logger.warning(warning)
            
        # Return True even with warnings, as they may not be critical
        return True
    except Exception as e:
        logger.error(f"Error validating data consistency: {str(e)}")
        return False

def validate_feature_engineering_inputs(
    data: Dict[str, pd.DataFrame],
    required_columns: Dict[str, List[str]]
) -> bool:
    """
    Validates input data for feature engineering.
    
    Args:
        data: Dictionary of input DataFrames
        required_columns: Dictionary specifying required columns for each DataFrame
        
    Returns:
        bool: True if all inputs are valid
    """
    try:
        for df_name, columns in required_columns.items():
            if df_name not in data:
                logger.error(f"Missing required dataset: {df_name}")
                return False
                
            df = data[df_name]
            missing_cols = [col for col in columns if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns in {df_name}: {missing_cols}")
                return False
                
            # Check for null values in required columns
            null_cols = [col for col in columns if df[col].isnull().any()]
            if null_cols:
                logger.warning(f"Null values found in {df_name} columns: {null_cols}")
        
        return True
    except Exception as e:
        logger.error(f"Error validating feature engineering inputs: {str(e)}")
        return False