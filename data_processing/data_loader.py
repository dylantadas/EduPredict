import os
import json
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from config import DATA_PROCESSING, DIRS

logger = logging.getLogger('edupredict')

def load_raw_datasets(
    data_path: str,
    chunk_size: Optional[int] = None,
    dtypes: Optional[Dict[str, Dict[str, str]]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.DataFrame]:
    """
    Loads OULAD dataset files with optimized memory usage.

    Args:
        data_path: Path to dataset directory
        chunk_size: Size of chunks for loading large files
        dtypes: Dictionary mapping file names to column data types
        logger: Logger for tracking progress

    Returns:
        Dictionary mapping dataset names to DataFrames
    """
    required_files = [
        ('studentInfo.csv', 'student_info'),
        ('vle.csv', 'vle_materials'),
        ('studentVle.csv', 'vle_interactions'),
        ('assessments.csv', 'assessments'),
        ('studentAssessment.csv', 'student_assessments')
    ]

    datasets = {}
    chunk_size = chunk_size or DATA_PROCESSING['chunk_size']
    dtypes = dtypes or DATA_PROCESSING['dtypes']
    logger = logger or logging.getLogger('edupredict')

    for file_name, dataset_key in required_files:
        file_path = os.path.join(data_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file {file_name} not found in {data_path}")

        logger.info(f"Loading {file_name}...")
        
        # Use chunked reading for large files
        if chunk_size and file_name in ['studentVle.csv']:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=dtypes.get(file_name)):
                chunks.append(chunk)
            datasets[dataset_key] = pd.concat(chunks, ignore_index=True)
        else:
            datasets[dataset_key] = pd.read_csv(file_path, dtype=dtypes.get(file_name))

        logger.info(f"Loaded {dataset_key} with shape {datasets[dataset_key].shape}")

    return datasets

def verify_data_integrity(
    datasets: Dict[str, pd.DataFrame],
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Checks for data consistency and integrity across loaded files.

    Args:
        datasets: Dictionary of loaded datasets
        logger: Logger for tracking verification

    Returns:
        Boolean indicating whether data is consistent
    """
    logger = logger or logging.getLogger('edupredict')
    warnings = []

    try:
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

        return len(warnings) == 0

    except Exception as e:
        logger.error(f"Error verifying data integrity: {str(e)}")
        return False

def export_dataset_summary(
    datasets: Dict[str, pd.DataFrame],
    output_path: str,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Generates summary statistics for loaded datasets.

    Args:
        datasets: Dictionary of loaded datasets
        output_path: Path to save summary report
        logger: Logger for tracking export process

    Returns:
        Path to saved summary report
    """
    logger = logger or logging.getLogger('edupredict')
    
    try:
        summary = {}
        
        for name, df in datasets.items():
            # Basic dataset info
            dataset_summary = {
                "shape": df.shape,
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
                "missing_values": df.isnull().sum().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
            
            # Numeric column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                dataset_summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
            
            # Categorical column statistics
            cat_cols = df.select_dtypes(include=['category', 'object']).columns
            if len(cat_cols) > 0:
                dataset_summary["categorical_stats"] = {
                    col: df[col].value_counts().to_dict() for col in cat_cols
                }
            
            summary[name] = dataset_summary

        # Save summary to file
        summary_path = os.path.join(output_path, 'dataset_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"Exported dataset summary to {summary_path}")
        return summary_path

    except Exception as e:
        logger.error(f"Error exporting dataset summary: {str(e)}")
        raise

def save_datasets_versioned(
    datasets: Dict[str, pd.DataFrame],
    output_dir: str,
    metadata: Dict[str, Any],
    enable_versioning: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    """
    Saves datasets with versioning information.

    Args:
        datasets: Dictionary of datasets to save
        output_dir: Directory to save datasets
        metadata: Metadata to include with version
        enable_versioning: Whether to enable versioning
        logger: Logger for tracking save process

    Returns:
        Dictionary mapping dataset names to saved paths and version ID
    """
    logger = logger or logging.getLogger('edupredict')
    
    try:
        # Create version ID if versioning is enabled
        version_id = None
        if enable_versioning:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content_hash = hashlib.md5(str(metadata).encode()).hexdigest()[:8]
            version_id = f"v_{timestamp}_{content_hash}"

            # Add version info to metadata
            metadata['version_id'] = version_id
            metadata['timestamp'] = timestamp

        # Create output directory
        save_dir = os.path.join(output_dir, version_id) if version_id else output_dir
        os.makedirs(save_dir, exist_ok=True)

        # Save datasets
        saved_paths = {}
        for name, df in datasets.items():
            file_path = os.path.join(save_dir, f"{name}.parquet")
            df.to_parquet(file_path, index=False)
            saved_paths[name] = {
                'path': file_path,
                'version_id': version_id
            }
            logger.info(f"Saved {name} to {file_path}")

        # Save metadata
        if metadata:
            metadata_path = os.path.join(save_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        return saved_paths

    except Exception as e:
        logger.error(f"Error saving versioned datasets: {str(e)}")
        raise