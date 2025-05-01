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
    
    Args:
        data: DataFrame to split
        target_col: Target column for stratification
        strat_cols: List of columns to use for stratification
        test_size: Proportion of data to include in the test split
        validation_size: Proportion of data to include in the validation split
        random_state: Random seed for reproducibility
        logger: Logger instance

    Returns:
        Dictionary with train, validation, and test splits"""
    logger = logger or logging.getLogger('edupredict')
    strat_cols = strat_cols or FAIRNESS['protected_attributes']

    try:
        # Log initial distributions of protected attributes
        logger.info("Initial protected attribute distributions:")
        for col in strat_cols:
            if col in data.columns:
                logger.info(f"{col} distribution:\n{data[col].value_counts(normalize=True)}")

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
        train_val_data = data.iloc[train_val_idx]
        test_data = data.iloc[test_idx]
        
        # Second split: train vs validation
        train_val_strat = train_val_data['strat_label']
        val_size_adjusted = validation_size / (1 - test_size)
        
        train_idx, val_idx = next(StratifiedKFold(
            n_splits=int(1/val_size_adjusted),
            shuffle=True,
            random_state=random_state
        ).split(train_val_data, train_val_strat))
        
        # Create final splits
        train_data = train_val_data.iloc[train_idx]
        val_data = train_val_data.iloc[val_idx]
        
        # Clean up temporary stratification column
        for df in [train_data, val_data, test_data]:
            if 'strat_label' in df.columns:
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
        
        imbalances_found = False
        for col in strat_cols:
            logger.info(f"\nDistribution of {col}:")
            reference_dist = train_data[col].value_counts(normalize=True)
            
            for name, split in splits.items():
                split_dist = split[col].value_counts(normalize=True)
                logger.info(f"{name}: {split_dist.to_dict()}")
                
                # Check distribution difference
                max_diff = max(abs(reference_dist - split_dist))
                if max_diff > FAIRNESS['threshold']:
                    imbalances_found = True
                    logger.warning(
                        f"Large distribution difference in {col} for {name} split: {max_diff:.3f}"
                    )

        if imbalances_found:
            logger.warning(
                "Some splits show significant demographic imbalances. Consider adjusting "
                "splitting strategy or using bias mitigation techniques."
            )
        
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
    data_splits: Dict[str, pd.DataFrame],
    demographic_cols: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, float]]:
    """
    Validates demographic balance across data splits.
    
    Args:
        data_splits: Dictionary of split DataFrames
        demographic_cols: Demographic columns to check
        logger: Logger instance
    
    Returns:
        Dictionary with distribution differences for each demographic column
    """
    logger = logger or logging.getLogger('edupredict')
    demographic_cols = demographic_cols or FAIRNESS['protected_attributes']
    threshold = FAIRNESS['threshold']
    
    try:
        validation_results = {
            'is_balanced': True,
            'differences': {},
            'details': {}
        }
        
        # Get distribution of each demographic column in each split
        distributions = {}
        for split_name, split_data in data_splits.items():
            if len(split_data) > 0:
                distributions[split_name] = {
                    col: split_data[col].value_counts(normalize=True).to_dict()
                    for col in demographic_cols if col in split_data.columns
                }
        
        # Compare distributions between splits
        for col in demographic_cols:
            col_diffs = {}
            max_diff = 0
            
            # Use train set as reference
            if 'train' in distributions:
                train_dist = distributions['train'][col]
                
                for split_name, split_dist in distributions.items():
                    if split_name != 'train':
                        # Calculate maximum absolute difference in proportions
                        split_col_dist = split_dist[col]
                        diffs = {
                            cat: abs(train_dist.get(cat, 0) - split_col_dist.get(cat, 0))
                            for cat in set(train_dist.keys()) | set(split_col_dist.keys())
                        }
                        max_cat_diff = max(diffs.values())
                        col_diffs[split_name] = max_cat_diff
                        max_diff = max(max_diff, max_cat_diff)
                        
                        # Log large differences
                        if max_cat_diff > threshold:
                            logger.warning(
                                f"Large distribution difference in {col} between train and {split_name} "
                                f"splits: {max_cat_diff:.3f} (threshold: {threshold})"
                            )
                            
                            # Log specific category differences
                            for cat, diff in diffs.items():
                                if diff > threshold:
                                    logger.warning(
                                        f"  Category '{cat}' shows significant difference: "
                                        f"train={train_dist.get(cat, 0):.3f}, "
                                        f"{split_name}={split_col_dist.get(cat, 0):.3f}"
                                    )
            
            validation_results['differences'][col] = col_diffs
            validation_results['details'][col] = {
                'max_difference': max_diff,
                'is_balanced': max_diff <= threshold,
                'threshold': threshold
            }
            
            if max_diff > threshold:
                validation_results['is_balanced'] = False
                
                # Add specific recommendations
                if 'recommendations' not in validation_results:
                    validation_results['recommendations'] = []
                validation_results['recommendations'].append(
                    f"Consider rebalancing {col} using stratified sampling or bias mitigation "
                    f"techniques to reduce maximum difference ({max_diff:.3f}) below threshold ({threshold})"
                )
        
        return validation_results

    except Exception as e:
        logger.error(f"Error validating demographic balance: {str(e)}")
        raise