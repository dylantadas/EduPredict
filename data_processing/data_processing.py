import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def load_raw_datasets(data_path: str) -> Dict[str, pd.DataFrame]:
    """Loads all csv files from OULAD dataset with optimized memory usage."""

    # define column types to reduce memory
    dtype_dict = {
        'id_student': 'int32',
        'code_module': 'category',
        'code_presentation': 'category',
        'gender': 'category',
        'region': 'category',
        'highest_education': 'category',
        'imd_band': 'category',
        'age_band': 'category',
        'num_of_prev_attempts': 'int8',
        'disability': 'category',
        'date': 'int16',
        'sum_click': 'int16',
        'module_presentation_length': 'int16'
    }

    # define which columns needed from each file
    dataset_files = {
        'student_info': {
            'file': 'studentInfo.csv',
            'columns': [
                'id_student', 'code_module', 'code_presentation',
                'gender', 'region', 'highest_education', 'imd_band', 'age_band',
                'num_of_prev_attempts', 'studied_credits', 'disability', 'final_result'
            ]
        },
        'vle_interactions': {
            'file': 'studentVle.csv',
            'columns': [
                'id_student', 'code_module', 'code_presentation',
                'date', 'sum_click', 'id_site'
            ]
        },
        'vle_materials': {
            'file': 'vle.csv',
            'columns': None  # load all columns
        },
        'assessments': {
            'file': 'assessments.csv',
            'columns': None  # load all columns
        },
        'student_assessments': {
            'file': 'studentAssessment.csv',
            'columns': None  # load all columns
        },
        'courses': {
            'file': 'courses.csv',
            'columns': [
                'code_module',
                'code_presentation',
                'module_presentation_length'
            ]
        }
    }

    datasets = {}
    total_rows = 0

    for key, file_info in dataset_files.items():
        try:
            # load csv with specified columns and optimized dtypes
            df = pd.read_csv(
                    os.path.join(data_path, file_info['file']),
                    usecols=file_info['columns'],
                    dtype={col: dtype_dict.get(col) for col in (file_info['columns'] or [])
                           if col in dtype_dict}
            )
            
            total_rows += len(df)
            print(f"Loaded {file_info['file']}: {len(df)} rows, {len(df.columns)} columns")
            datasets[key] = df

        except FileNotFoundError:
            print(f"Error: {file_info['file']} not found")
            datasets[key] = None

        except ValueError as e:
            print(f"Error loading {file_info['file']}: {str(e)}")
            print("Loading all columns instead...")
            # fallback: load all columns if unmatched specified columns
            df = pd.read_csv(os.path.join(data_path, file_info['file']))
            total_rows += len(df)
            print(f"Loaded {file_info['file']}: {len(df)} rows, {len(df.columns)} columns")
            datasets[key] = df

    print(f"\nTotal rows loaded across all files: {total_rows:,}")
    return datasets


def clean_demographic_data(student_info: pd.DataFrame) -> pd.DataFrame:
    """Performs initial cleaning of demographic data and handles missing values."""

    cleaned_data = student_info.copy()

    # standardize string columns
    string_columns = ['gender', 'region', 'highest_education', 'imd_band', 'age_band']
    for col in string_columns:
        cleaned_data[col] = cleaned_data[col].str.strip().str.lower()

    # handle missing values
    cleaned_data['imd_band'] = cleaned_data['imd_band'].fillna('unknown')
    cleaned_data['disability'] = cleaned_data['disability'].fillna('N')

    return cleaned_data


def clean_vle_data(vle_interactions: pd.DataFrame, 
                  vle_materials: pd.DataFrame) -> pd.DataFrame:
    """Cleans and merges vle interaction data, removes outliers/invalid entries."""

    # remove interactions with invalid click counts
    cleaned_interactions = vle_interactions[vle_interactions['sum_click'] > 0]

    # merge with materials data
    merged_data = cleaned_interactions.merge(
        vle_materials,
        on=['id_site', 'code_module', 'code_presentation'],
        how='left'
    )

    # remove entries with missing material types
    merged_data = merged_data.dropna(subset=['activity_type'])

    return merged_data


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


def validate_data_consistency(datasets: Dict[str, pd.DataFrame]) -> bool:
    """Validates consistency across datasets."""

    try:
        # check student IDs consistency across files
        student_ids = set(datasets['student_info']['id_student'])
        vle_ids = set(datasets['vle_interactions']['id_student'])
        assessment_ids = set(datasets['student_assessments']['id_student'])

        # ensure all students in VLE and assessments exist in student_info
        if not (vle_ids.issubset(student_ids) and assessment_ids.issubset(student_ids)):
            print("Warning: Some VLE or assessment records have unknown student IDs")

        # check module-presentation pairs consistency
        modules_presentations = set(zip(datasets['courses']['code_module'],
                                     datasets['courses']['code_presentation']))
        student_modules = set(zip(datasets['student_info']['code_module'],
                                datasets['student_info']['code_presentation']))

        if not student_modules.issubset(modules_presentations):
            print("Warning: Invalid module-presentation combinations found")

        return True

    except Exception as e:
        print(f"Validation failed: {str(e)}")
        return False
