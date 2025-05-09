import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
import os
from sklearn.model_selection import StratifiedKFold
from config import FEATURE_ENGINEERING, EVALUATION, VERSION_CONTROL, DIRS, FAIRNESS, PROTECTED_ATTRIBUTES

logger = logging.getLogger('edupredict')

def create_stratified_splits(
    data: pd.DataFrame,
    target_col: str,
    strat_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.DataFrame]:
    """
    Creates stratified train/test/validation splits preserving demographic distributions.
    """
    logger = logger or logging.getLogger('edupredict')
    strat_cols = strat_cols or FAIRNESS['protected_attributes']

    try:
        # Log initial distributions of protected attributes
        logger.info("Initial protected attribute distributions:")
        for col in strat_cols:
            if col in data.columns:
                dist = data[col].value_counts(normalize=True)
                logger.info(f"{col} distribution:\n{dist}")

        # Create combined stratification label using protected attributes
        data['strat_label'] = data[strat_cols].astype(str).apply('_'.join, axis=1)
        
        # Check for small groups before splitting
        group_counts = data['strat_label'].value_counts()
        small_groups = group_counts[group_counts < FAIRNESS['min_group_size']]
        if not small_groups.empty:
            logger.warning(
                f"Found intersectional groups below minimum size ({FAIRNESS['min_group_size']}):\n"
                f"{small_groups}"
            )

        # First split: train+val vs test
        train_val_idx, test_idx = next(StratifiedKFold(
            n_splits=int(1/test_size),
            shuffle=True,
            random_state=random_state
        ).split(data, data['strat_label']))
        
        # Create train+val and test sets
        train_val_data = data.iloc[train_val_idx].copy()
        test_data = data.iloc[test_idx].copy()
        
        # Second split: train vs validation
        train_val_strat = train_val_data['strat_label']
        val_size_adjusted = validation_size / (1 - test_size)
        
        train_idx, val_idx = next(StratifiedKFold(
            n_splits=int(1/val_size_adjusted),
            shuffle=True,
            random_state=random_state
        ).split(train_val_data, train_val_strat))
        
        # Create final splits
        train_data = train_val_data.iloc[train_idx].copy()
        val_data = train_val_data.iloc[val_idx].copy()
        
        # Clean up temporary stratification column
        for df in [train_data, val_data, test_data]:
            df.drop('strat_label', axis=1, inplace=True)
        
        # Log split sizes
        logger.info(f"Train set: {len(train_data)} samples")
        logger.info(f"Validation set: {len(val_data)} samples")
        logger.info(f"Test set: {len(test_data)} samples")
        
        # Verify demographic balance
        splits = {
            'Train': train_data,
            'Validation': val_data,
            'Test': test_data
        }
        
        for col in strat_cols:
            logger.info(f"\nDistribution of {col}:")
            train_dist = train_data[col].value_counts(normalize=True).to_dict()
            val_dist = val_data[col].value_counts(normalize=True).to_dict()
            test_dist = test_data[col].value_counts(normalize=True).to_dict()
            
            logger.info(f"Train: {train_dist}")
            logger.info(f"Validation: {val_dist}")
            logger.info(f"Test: {test_dist}")
            
            # Validate distributions
            if not validate_demographic_balance(train_dist, val_dist, test_dist, FAIRNESS['threshold']):
                logger.warning(f"Demographic imbalance detected in {col}")
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }

    except Exception as e:
        logger.error(f"Error creating stratified splits: {str(e)}")
        raise

def perform_student_level_split(
    data: pd.DataFrame,
    student_col: str,
    protected_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.DataFrame]:
    """
    Splits data at student level while maintaining demographic balance.
    
    Args:
        data: DataFrame to split
        student_col: Column containing student IDs
        protected_cols: Protected attribute columns to balance
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        logger: Logger instance
    
    Returns:
        Dictionary with train, validation, and test splits
    """
    logger = logger or logging.getLogger('edupredict')
    protected_cols = protected_cols or FAIRNESS['protected_attributes']
    
    try:
        # Create stratification label using protected attributes
        student_data = data.groupby(student_col)[protected_cols].agg(lambda x: x.iloc[0])
        student_data['strat_label'] = student_data[protected_cols].astype(str).apply('_'.join, axis=1)
        
        # Get unique student IDs
        unique_students = student_data.index.unique()
        
        # Perform stratified split of student IDs
        splitter = StratifiedKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
        train_val_idx, test_idx = next(splitter.split(
            unique_students,
            student_data.loc[unique_students, 'strat_label']
        ))
        
        # Split student IDs
        test_ids = unique_students[test_idx]
        train_val_ids = unique_students[train_val_idx]
        
        # Create test set
        test_data = data[data[student_col].isin(test_ids)]
        remaining_data = data[data[student_col].isin(train_val_ids)]
        
        if val_size:
            # Calculate adjusted validation size
            adjusted_val_size = val_size / (1 - test_size)
            
            # Split remaining data into train and validation
            train_idx, val_idx = next(StratifiedKFold(
                n_splits=int(1/adjusted_val_size),
                shuffle=True,
                random_state=random_state
            ).split(
                train_val_ids,
                student_data.loc[train_val_ids, 'strat_label']
            ))
            
            val_ids = train_val_ids[val_idx]
            train_ids = train_val_ids[train_idx]
            
            val_data = remaining_data[remaining_data[student_col].isin(val_ids)]
            train_data = remaining_data[remaining_data[student_col].isin(train_ids)]
        else:
            val_data = pd.DataFrame()
            train_data = remaining_data
        
        # Log split information
        logger.info(f"Train set: {len(train_data)} rows, {train_data[student_col].nunique()} students")
        if len(val_data) > 0:
            logger.info(f"Validation set: {len(val_data)} rows, {val_data[student_col].nunique()} students")
        logger.info(f"Test set: {len(test_data)} rows, {test_data[student_col].nunique()} students")
        
        # Check demographic balance
        splits = {
            'Train': train_data,
            'Test': test_data
        }
        if len(val_data) > 0:
            splits['Validation'] = val_data
        
        for col in protected_cols:
            logger.info(f"\nDistribution of {col} across splits:")
            reference_dist = train_data.groupby(student_col)[col].first().value_counts(normalize=True)
            
            for name, split in splits.items():
                split_dist = split.groupby(student_col)[col].first().value_counts(normalize=True)
                logger.info(f"{name}: {split_dist.to_dict()}")
                
                max_diff = max(abs(reference_dist - split_dist))
                if max_diff > FAIRNESS['threshold']:
                    logger.warning(
                        f"Large distribution difference in {col} for {name} split: {max_diff:.3f}"
                    )
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
    except Exception as e:
        logger.error(f"Error performing student-level split: {str(e)}")
        raise

def save_data_splits(
    data_splits: Dict[str, pd.DataFrame],
    output_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, str]]:
    """
    Saves split datasets with demographic information and metadata.
    
    Args:
        data_splits: Dictionary of split DataFrames
        output_dir: Directory to save the splits
        metadata: Additional metadata to include
        logger: Logger instance
        
    Returns:
        Dictionary with paths and sizes of saved splits"""
    logger = logger or logging.getLogger('edupredict')
    protected_cols = FAIRNESS['protected_attributes']
    
    try:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        version_dir = os.path.join(output_dir, f'split_v{timestamp}')
        os.makedirs(version_dir, exist_ok=True)
        
        saved_paths = {}
        split_demographics = {}
        
        for split_name, split_data in data_splits.items():
            if len(split_data) > 0:
                # Calculate demographic statistics
                demo_stats = {}
                for col in protected_cols:
                    if col in split_data.columns:
                        demo_stats[col] = split_data[col].value_counts().to_dict()
                
                # Save split data
                file_path = os.path.join(version_dir, f'{split_name}.parquet')
                split_data.to_parquet(file_path, index=False)
                
                saved_paths[split_name] = {
                    'path': file_path,
                    'n_rows': len(split_data)
                }
                split_demographics[split_name] = demo_stats
                logger.info(f"Saved {split_name} split to {file_path}")
        
        # Prepare extended metadata
        full_metadata = {
            'timestamp': timestamp,
            'split_sizes': {name: info['n_rows'] for name, info in saved_paths.items()},
            'demographic_distributions': split_demographics,
            'fairness_config': {
                'threshold': FAIRNESS['threshold'],
                'min_group_size': FAIRNESS['min_group_size'],
                'protected_attributes': protected_cols
            }
        }
        
        if metadata:
            full_metadata.update(metadata)
        
        # Save metadata
        metadata_path = os.path.join(version_dir, 'split_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        logger.info(f"Saved split metadata to {metadata_path}")
        
        return saved_paths

    except Exception as e:
        logger.error(f"Error saving data splits: {str(e)}")
        raise

def validate_demographic_balance(
    train_dist: Dict[str, float],
    val_dist: Optional[Dict[str, float]] = None,
    test_dist: Optional[Dict[str, float]] = None,
    threshold: float = 0.05
) -> bool:
    """
    Validates that demographic distributions are similar across splits.
    """
    try:
        if not train_dist:
            return False

        # Get all unique categories across all distributions
        all_categories = set(train_dist.keys())
        if val_dist:
            all_categories.update(val_dist.keys())
        if test_dist:
            all_categories.update(test_dist.keys())

        # Normalize distributions by adding missing categories with 0 frequency
        train_dist = {cat: train_dist.get(cat, 0.0) for cat in all_categories}
        if val_dist:
            val_dist = {cat: val_dist.get(cat, 0.0) for cat in all_categories}
        if test_dist:
            test_dist = {cat: test_dist.get(cat, 0.0) for cat in all_categories}

        # Re-normalize proportions to ensure they sum to 1
        train_sum = sum(train_dist.values())
        train_dist = {k: v/train_sum for k, v in train_dist.items()}

        if val_dist:
            val_sum = sum(val_dist.values())
            val_dist = {k: v/val_sum for k, v in val_dist.items()}
            # Check validation set distribution
            for category in all_categories:
                if abs(train_dist[category] - val_dist[category]) > threshold:
                    return False

        if test_dist:
            test_sum = sum(test_dist.values())
            test_dist = {k: v/test_sum for k, v in test_dist.items()}
            # Check test set distribution
            for category in all_categories:
                if abs(train_dist[category] - test_dist[category]) > threshold:
                    return False

        return True

    except Exception as e:
        logging.error(f"Error in demographic balance validation: {str(e)}")
        return False