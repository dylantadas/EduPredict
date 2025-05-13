import os
import json
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import defaultdict
from pathlib import Path
from config import DATA_PROCESSING, DIRS

logger = logging.getLogger('edupredict')

def validate_date_fields(
    datasets: Dict[str, pd.DataFrame],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validates date fields across all datasets for consistency and proper interpretation.
    Ensures dates are properly interpreted relative to module start (day 0).
    
    Args:
        datasets: Dictionary of DataFrames to validate
        logger: Optional logger instance
        
    Returns:
        Tuple of (is_valid, dict of validation issues by dataset)
    """
    logger = logger or logging.getLogger('edupredict')
    issues = defaultdict(list)
    
    try:
        # Validate VLE dates
        if 'vle_interactions' in datasets:
            vle = datasets['vle_interactions']
            if 'date' not in vle.columns:
                issues['vle_interactions'].append("Missing required date column")
            else:
                # Check date range
                date_range = vle['date'].agg(['min', 'max'])
                if date_range['min'] < -90:  # More than 90 days before module start
                    issues['vle_interactions'].append(
                        f"Unusually early activity detected: {date_range['min']} days"
                    )
                if date_range['max'] > 365:  # More than a year after start
                    issues['vle_interactions'].append(
                        f"Activity extends beyond expected module duration: {date_range['max']} days"
                    )
        
        # Validate assessment dates
        if 'assessments' in datasets:
            assessments = datasets['assessments']
            if 'date' not in assessments.columns:
                issues['assessments'].append("Missing required date column")
            else:
                # Check assessment scheduling
                date_range = assessments['date'].agg(['min', 'max'])
                if date_range['min'] < -30:  # Assessments >30 days before start
                    issues['assessments'].append(
                        f"Assessment scheduled too early: {date_range['min']} days"
                    )
                
                # Check for reasonable assessment spacing
                assessment_dates = sorted(assessments['date'].unique())
                for i in range(len(assessment_dates) - 1):
                    gap = assessment_dates[i+1] - assessment_dates[i]
                    if gap < 7:  # Less than a week between assessments
                        issues['assessments'].append(
                            f"Short gap between assessments: {gap} days between "
                            f"days {assessment_dates[i]} and {assessment_dates[i+1]}"
                        )
        
        # Validate registration dates
        if 'student_registration' in datasets:
            registration = datasets['student_registration']
            # Check registration dates
            if 'date_registration' in registration.columns:
                date_range = registration['date_registration'].agg(['min', 'max'])
                if date_range['min'] < -180:  # More than 6 months before module start
                    issues['student_registration'].append(
                        f"Very early registrations detected: {date_range['min']} days before module start"
                    )
                if date_range['max'] > 60:  # More than 2 months after start
                    issues['student_registration'].append(
                        f"Very late registrations detected: {date_range['max']} days after module start"
                    )
            
            # Check unregistration dates if they exist
            if 'date_unregistration' in registration.columns:
                # Filter out NaN values which indicate completed students
                unregistered = registration['date_unregistration'].dropna()
                if not unregistered.empty:
                    date_range = unregistered.agg(['min', 'max'])
                    if date_range['min'] < 0:  # Unregistration before module start
                        issues['student_registration'].append(
                            f"Unregistrations before module start detected: {date_range['min']} days"
                        )
                    early_drops = unregistered[unregistered < 14].count()
                    if early_drops > 0:
                        issues['student_registration'].append(
                            f"Found {early_drops} very early unregistrations (<14 days after module start)"
                        )
        
        # Validate submission dates
        if all(k in datasets for k in ['assessments', 'student_assessments']):
            merged = pd.merge(
                datasets['student_assessments'],
                datasets['assessments'][['id_assessment', 'date']],
                on='id_assessment'
            )
            
            # Check submission timing
            submission_lag = merged['date_submitted'] - merged['date']
            early_submissions = submission_lag[submission_lag < 0]
            if len(early_submissions) > 0:
                issues['student_assessments'].append(
                    f"Found {len(early_submissions)} submissions before due date"
                )
            
            very_late = submission_lag[submission_lag > 30]
            if len(very_late) > 0:
                issues['student_assessments'].append(
                    f"Found {len(very_late)} submissions more than 30 days late"
                )
        
        # Cross-dataset timeline validation
        all_dates = []
        date_sources = []
        
        if 'vle_interactions' in datasets:
            all_dates.extend(datasets['vle_interactions']['date'].unique())
            date_sources.append('VLE')
        
        if 'assessments' in datasets:
            all_dates.extend(datasets['assessments']['date'].unique())
            date_sources.append('Assessments')
        
        if 'student_assessments' in datasets:
            all_dates.extend(datasets['student_assessments']['date_submitted'].unique())
            date_sources.append('Submissions')
            
        if 'student_registration' in datasets and 'date_registration' in datasets['student_registration'].columns:
            all_dates.extend(datasets['student_registration']['date_registration'].unique())
            date_sources.append('Registrations')
            # Add unregistration dates if they exist (excluding NaN values)
            if 'date_unregistration' in datasets['student_registration'].columns:
                unregistration_dates = datasets['student_registration']['date_unregistration'].dropna().unique()
                all_dates.extend(unregistration_dates)
                date_sources.append('Unregistrations')
        
        if all_dates:
            timeline_range = max(all_dates) - min(all_dates)
            if timeline_range > 365:  # More than a year
                issues['timeline'].append(
                    f"Overall activity span ({timeline_range} days) exceeds expected "
                    f"module length. Sources: {', '.join(date_sources)}"
                )
        
        return len(issues) == 0, dict(issues)
        
    except Exception as e:
        logger.error(f"Error validating date fields: {str(e)}")
        return False, {'error': [str(e)]}

def load_raw_datasets(
    data_path: str,
    chunk_size: Optional[int] = None,
    dtypes: Optional[Dict[str, Dict[str, str]]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.DataFrame]:
    """
    Loads OULAD dataset files with optimized memory usage and enhanced validation.
    Ensures proper interpretation of date fields relative to module start.

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
        ('studentAssessment.csv', 'student_assessments'),
        ('studentRegistration.csv', 'student_registration')
    ]

    datasets = {}
    chunk_size = chunk_size or DATA_PROCESSING['chunk_size']
    dtypes = dtypes or DATA_PROCESSING['dtypes']
    logger = logger or logging.getLogger('edupredict')

    for file_name, dataset_key in required_files:
        file_path = os.path.join(data_path, file_name)
        if not os.path.exists(file_path):
            if file_name == 'studentRegistration.csv':
                logger.warning(f"Optional file {file_name} not found in {data_path}, skipping...")
                continue
            else:
                raise FileNotFoundError(f"Required file {file_name} not found in {data_path}")

        logger.info(f"Loading {file_name}...")
        
        # Use chunked reading for large files
        if chunk_size and file_name in ['studentVle.csv', 'studentRegistration.csv']:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=dtypes.get(file_name)):
                chunks.append(chunk)
            datasets[dataset_key] = pd.concat(chunks, ignore_index=True)
        else:
            datasets[dataset_key] = pd.read_csv(file_path, dtype=dtypes.get(file_name))

        logger.info(f"Loaded {dataset_key} with shape {datasets[dataset_key].shape}")

    # Validate dates across datasets
    is_valid, date_issues = validate_date_fields(datasets, logger)
    if not is_valid:
        for dataset, issues in date_issues.items():
            for issue in issues:
                logger.warning(f"{dataset}: {issue}")

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
        
        # Check registration data consistency
        if 'student_registration' in datasets:
            registration_student_ids = set(datasets['student_registration']['id_student'])
            if not registration_student_ids.issubset(student_ids):
                warnings.append("Registration data contain unknown student IDs")
            
            # Check if there are duplicate registrations for same student/module/presentation
            if all(col in datasets['student_registration'].columns 
                   for col in ['id_student', 'code_module', 'code_presentation']):
                registration_df = datasets['student_registration']
                registration_counts = registration_df.groupby(
                    ['id_student', 'code_module', 'code_presentation']
                ).size()
                
                duplicate_registrations = registration_counts[registration_counts > 1]
                if not duplicate_registrations.empty:
                    warnings.append(
                        f"Found {len(duplicate_registrations)} students with duplicate "
                        "registrations for the same module/presentation"
                    )

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

def create_fairness_aware_sample(
    datasets: Dict[str, pd.DataFrame],
    sample_size: int = 5000,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.DataFrame]:
    """
    Creates a fairness-aware sample of the data that preserves demographic distributions
    across all protected attributes defined in the configuration.
    
    Args:
        datasets: Dictionary of dataframes to sample from
        sample_size: Target size for student sample (default 5000)
        random_state: Random seed for reproducibility
        logger: Logger instance
    
    Returns:
        Dictionary of sampled dataframes
    """
    from config import FAIRNESS, PROTECTED_ATTRIBUTES
    
    logger = logger or logging.getLogger('edupredict')
    
    try:
        # Get student info dataframe
        student_info = datasets['student_info']
        
        # Get protected attributes from configuration
        protected_attrs = FAIRNESS.get('protected_attributes', [])
        
        if not protected_attrs:
            logger.warning("No protected attributes defined in configuration. Using default attributes.")
            protected_attrs = ['gender', 'age_band', 'imd_band']
        
        # Filter to only include protected attributes that exist in the dataset
        available_attrs = [attr for attr in protected_attrs if attr in student_info.columns]
        
        if not available_attrs:
            logger.warning("None of the configured protected attributes found in dataset. Using stratified random sampling.")
            # Fall back to simple random sampling if no protected attributes are available
            sampled_student_info = student_info.sample(n=min(sample_size, len(student_info)), random_state=random_state)
            sampled_ids = sampled_student_info['id_student'].unique()
        else:
            # Calculate original demographic proportions for logging purposes
            demo_props = {}
            for attr in available_attrs:
                demo_props[attr] = student_info[attr].value_counts(normalize=True)
                logger.info(f"Original {attr} distribution:\n{demo_props[attr]}")
            
            # Create stratified sample using all available protected attributes
            # Start by creating a composite strata column
            student_info_copy = student_info.copy()
            student_info_copy['strata'] = ''
            for attr in available_attrs:
                student_info_copy['strata'] += student_info_copy[attr].astype(str) + '_'
            
            strata_counts = student_info_copy['strata'].value_counts()
            
            # Calculate sample size for each stratum proportionally
            sample_sizes = {}
            min_size_per_stratum = 1  # Ensure at least one sample per stratum
            
            # Apply balanced thresholds from config if available
            for stratum, count in strata_counts.items():
                # Calculate proportional sample size
                prop_size = int((count / len(student_info)) * sample_size)
                
                # Apply minimum thresholds from config for protected attributes
                stratum_attrs = stratum.split('_')[:-1]  # Last element is empty due to trailing '_'
                
                # Check if any protected attribute has a minimum threshold
                for i, attr_value in enumerate(stratum_attrs):
                    attr_name = available_attrs[i]
                    if attr_name in PROTECTED_ATTRIBUTES:
                        balanced_threshold = PROTECTED_ATTRIBUTES[attr_name].get('balanced_threshold')
                        if balanced_threshold:
                            # Ensure this group gets at least the minimum specified percentage
                            min_threshold_size = int(balanced_threshold * sample_size)
                            prop_size = max(prop_size, min_threshold_size)
                
                # Ensure at least minimum size
                sample_sizes[stratum] = max(prop_size, min_size_per_stratum)
            
            # Adjust to match target sample size
            total_allocated = sum(sample_sizes.values())
            if total_allocated > sample_size:
                # Scale down proportionally
                scaling_factor = sample_size / total_allocated
                sample_sizes = {k: max(1, int(v * scaling_factor)) for k, v in sample_sizes.items()}
            
            # Sample from each stratum
            sampled_students = []
            for stratum, size in sample_sizes.items():
                stratum_df = student_info_copy[student_info_copy['strata'] == stratum]
                if len(stratum_df) > 0:
                    if len(stratum_df) >= size:
                        sampled = stratum_df.sample(n=size, random_state=random_state)
                    else:
                        # If not enough samples, take all available and note the shortage
                        logger.warning(f"Stratum {stratum} has only {len(stratum_df)} samples, needed {size}")
                        sampled = stratum_df
                    sampled_students.append(sampled)
            
            # Combine all sampled strata
            sampled_student_info = pd.concat(sampled_students).drop(columns=['strata'])
            sampled_ids = sampled_student_info['id_student'].unique()
        
        # Create samples for other datasets based on sampled student IDs
        sampled_datasets = {
            'student_info': sampled_student_info
        }
        
        # Sample other datasets based on sampled student IDs
        for name, df in datasets.items():
            if name != 'student_info':
                if 'id_student' in df.columns:
                    sampled_df = df[df['id_student'].isin(sampled_ids)]
                    sampled_datasets[name] = sampled_df
                else:
                    sampled_datasets[name] = df
        
        # Log sampling results
        logger.info(f"Created fairness-aware sample with {len(sampled_ids)} students")
        for name, df in sampled_datasets.items():
            logger.info(f"Sampled {name}: {len(df)} rows")
            
        # Verify demographic distributions
        if available_attrs:
            new_props = {}
            dist_changes = {}
            
            for attr in available_attrs:
                new_props[attr] = sampled_student_info[attr].value_counts(normalize=True)
                # Calculate absolute difference in distributions
                attr_diffs = {}
                for category in set(demo_props[attr].index) | set(new_props[attr].index):
                    orig_val = demo_props[attr].get(category, 0)
                    new_val = new_props[attr].get(category, 0)
                    attr_diffs[category] = abs(new_val - orig_val)
                
                dist_changes[attr] = attr_diffs
                
                logger.info(f"Final {attr} distribution:\n{new_props[attr]}")
                max_diff = max(attr_diffs.values()) if attr_diffs else 0
                logger.info(f"Maximum distribution difference for {attr}: {max_diff:.4f}")
                
                # Check if distribution difference exceeds fairness threshold
                if max_diff > FAIRNESS.get('threshold', 0.1):
                    logger.warning(f"Distribution difference for {attr} exceeds fairness threshold")
        
        return sampled_datasets
        
    except Exception as e:
        logger.error(f"Error creating fairness-aware sample: {str(e)}")
        raise