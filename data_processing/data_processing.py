import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import logging
from typing import Dict, List, Optional

logger = logging.getLogger('edupredict')

def load_raw_datasets(dataset_paths):
    """Load all raw datasets from specified paths."""
    required_files = [
        'studentInfo.csv',
        'vle.csv',
        'studentVle.csv',
        'assessments.csv',
        'studentAssessment.csv'
    ]
    
    datasets = {}
    for file in required_files:
        file_path = os.path.join(dataset_paths, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file {file} not found in {dataset_paths}")
        
        name = file.split('.')[0]
        datasets[name] = pd.read_csv(file_path)
    
    return {
        'student_info': datasets['studentInfo'],
        'vle_materials': datasets['vle'],
        'vle_interactions': datasets['studentVle'],
        'assessments': datasets['assessments'],
        'student_assessments': datasets['studentAssessment']
    }

def clean_demographic_data(student_info: pd.DataFrame) -> pd.DataFrame:
    """Performs initial cleaning of demographic data and handles missing values."""

    cleaned_data = student_info.copy()
    original_count = len(cleaned_data)
    
    # filter rows with missing final_result 
    cleaned_data = cleaned_data.dropna(subset=['final_result'])
    
    # log filtered out rows
    removed_count = original_count - len(cleaned_data)
    if removed_count > 0:
        print(f"Removed {removed_count} rows ({removed_count/original_count:.2%}) with missing final_result values")

    # standardize string columns
    string_columns = ['gender', 'region', 'highest_education', 'imd_band', 'age_band']
    for col in string_columns:
        cleaned_data[col] = cleaned_data[col].str.strip().str.lower()

    # handle missing values
    cleaned_data['imd_band'] = cleaned_data['imd_band'].fillna('unknown')
    cleaned_data['disability'] = cleaned_data['disability'].fillna('N')

    return cleaned_data


def clean_vle_data(vle_interactions, vle_materials, chunk_size=100000):
    """Clean and merge VLE interaction data with chunked processing."""
    logger.info("Cleaning VLE data...")
    
    # First, clean materials data (usually smaller)
    clean_materials = vle_materials.copy()
    clean_materials['activity_type'] = clean_materials['activity_type'].fillna('unknown')
    
    # Process interactions in chunks
    cleaned_chunks = []
    for chunk_start in range(0, len(vle_interactions), chunk_size):
        chunk = vle_interactions.iloc[chunk_start:chunk_start + chunk_size].copy()
        
        # Basic cleaning
        chunk = chunk[chunk['sum_click'] > 0]  # Filter invalid clicks
        
        # Merge with materials (using merge_asof if temporal data)
        chunk = pd.merge(
            chunk,
            clean_materials[['id_site', 'activity_type']],
            on='id_site',
            how='left'
        )
        
        cleaned_chunks.append(chunk)
        
    # Combine chunks
    cleaned_data = pd.concat(cleaned_chunks, ignore_index=True)
    
    return cleaned_data


def clean_assessment_data(assessments: pd.DataFrame,
                         student_assessments: pd.DataFrame) -> pd.DataFrame:
    """Cleans assessment data, handling missing scores and invalid dates."""

    # remove invalid scores
    valid_assessments = student_assessments[
        (student_assessments['score'] >= 0) &
        (student_assessments['score'] <= 100)
    ]

    # merge with assessment data
    cleaned_data = valid_assessments.merge(
        assessments,
        on='id_assessment',
        how='left'
    )

    return cleaned_data


def validate_data_consistency(datasets):
    """Validate consistency across different datasets."""
    # Check student IDs consistency
    student_ids = set(datasets['student_info']['id_student'])
    vle_student_ids = set(datasets['vle_interactions']['id_student'])
    assessment_student_ids = set(datasets['student_assessments']['id_student'])
    
    # All students in interactions should be in student_info
    if not vle_student_ids.issubset(student_ids):
        raise ValueError("VLE interactions contain unknown student IDs")
    if not assessment_student_ids.issubset(student_ids):
        raise ValueError("Assessment data contain unknown student IDs")
    
    # Check assessment IDs consistency
    assessment_ids = set(datasets['assessments']['id_assessment'])
    student_assessment_ids = set(datasets['student_assessments']['id_assessment'])
    if not student_assessment_ids.issubset(assessment_ids):
        raise ValueError("Student assessment data contain unknown assessment IDs")
    
    # Check VLE activity IDs consistency
    vle_ids = set(datasets['vle_materials']['id_site'])
    interaction_vle_ids = set(datasets['vle_interactions']['id_site'])
    if not interaction_vle_ids.issubset(vle_ids):
        raise ValueError("VLE interactions contain unknown activity IDs")
    
    return True
