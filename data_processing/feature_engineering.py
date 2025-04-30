import pandas as pd # type: ignore
import numpy as np # type: ignore
from typing import Dict, List, Optional
import logging
from config import PROTECTED_ATTRIBUTES
from config import RANDOM_SEED
from sklearn.model_selection import train_test_split # type: ignore

logger = logging.getLogger('edupredict')

def create_demographic_features(cleaned_demographics: pd.DataFrame) -> pd.DataFrame:
    """Transforms cleaned demographic data into model features."""

    features = cleaned_demographics.copy()

    # preserve key identifier columns
    id_columns = ['id_student', 'code_module', 'code_presentation']
    preserved_ids = features[id_columns].copy()

    # create encoded categorical variables
    categorical_columns = ['gender', 'region', 'highest_education',
                         'imd_band', 'age_band']

    for col in categorical_columns:
        # label encoding
        features[f"{col}_encoded"] = pd.Categorical(features[col]).codes

        # one-hot encoding for tree-based models
        one_hot = pd.get_dummies(features[col],
                                prefix=col,
                                drop_first=True)
        features = pd.concat([features, one_hot], axis=1)

    # create educational background features
    features['is_first_attempt'] = (features['num_of_prev_attempts'] == 0)
    features['credit_density'] = features['studied_credits'] / features['num_of_prev_attempts'].clip(1)

    # use preserved key column copies
    for col in id_columns:
        features[col] = preserved_ids[col]

    return features


def create_temporal_features(cleaned_vle_data: pd.DataFrame,
                           window_sizes: List[int]) -> Dict[str, pd.DataFrame]:
    """Creates time-based engagement features using multiple window sizes."""

    temporal_features = {}

    for window_size in window_sizes:
        # create time windows
        cleaned_vle_data['window'] = cleaned_vle_data['date'] // window_size

        # store grouping columns
        group_cols = ['id_student', 'code_module', 'code_presentation', 'window']

        # handle numeric aggregations
        numeric_metrics = cleaned_vle_data.groupby(group_cols).agg({
            'sum_click': ['sum', 'mean', 'std'],
            'id_site': 'nunique'
        })

        # flatten column names
        numeric_metrics.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
            for col in numeric_metrics.columns
        ]
        numeric_metrics = numeric_metrics.reset_index()

        # handle activity type aggregation
        activity_counts = pd.pivot_table(
            cleaned_vle_data,
            index=group_cols,
            columns='activity_type',
            values='sum_click',
            aggfunc='count',
            fill_value=0
        )

        # rename activity columns and reset index
        activity_counts.columns = [f'activity_{col}' for col in activity_counts.columns]
        activity_counts = activity_counts.reset_index()

        # merge dataframes
        temporal_features[f"window_{window_size}"] = numeric_metrics.merge(
            activity_counts,
            on=group_cols,
            validate='one_to_one'  # validates 1-1 merge
        )

    return temporal_features


def create_assessment_features(cleaned_assessment_data: pd.DataFrame) -> pd.DataFrame:
    """Creates assessment-based features."""

    # preserve key identifier structure
    group_cols = ['id_student', 'code_module', 'code_presentation']

    # calculate submission delay and weighted components
    cleaned_assessment_data['submission_delay'] = (
        cleaned_assessment_data['date_submitted'] - cleaned_assessment_data['date']
    )
    cleaned_assessment_data['score_weight_product'] = (
        cleaned_assessment_data['score'] * cleaned_assessment_data['weight']
    )

    # group and aggregate all metrics
    performance_metrics = cleaned_assessment_data.groupby(group_cols).agg({
        'score': ['mean', 'std', 'min', 'max', 'count'],
        'submission_delay': ['mean', 'std'],
        'weight': 'sum',
        'is_banked': 'sum',
        'score_weight_product': 'sum'
    }).reset_index()  # preserves original column names

    # flatten column names except identifier columns unchanged
    performance_metrics.columns = [
        col[0] if col[0] in group_cols else f"{col[0]}_{col[1]}"
        for col in performance_metrics.columns
    ]

    # calculate final metrics
    performance_metrics['weighted_score'] = (
        performance_metrics['score_weight_product_sum'] /
        performance_metrics['weight_sum'].replace(0, np.nan)
    ).fillna(0)

    performance_metrics['submission_consistency'] = (
        performance_metrics['submission_delay_std'] /
        performance_metrics['submission_delay_mean'].replace(0, np.nan)
    ).fillna(0)

    # drop intermediate calculation columns
    performance_metrics = performance_metrics.drop(
        ['score_weight_product_sum', 'weight_sum'],
        axis=1
    )

    return performance_metrics


def create_sequential_features(cleaned_vle_data, chunk_size=50000):
    """Creates sequential features with optimized chunked processing."""
    logger.info("Creating sequential features...")
    
    # Pre-sort the entire dataset
    cleaned_vle_data = cleaned_vle_data.sort_values(['id_student', 'date'])
    
    # Get unique students
    unique_students = cleaned_vle_data['id_student'].unique()
    
    # Process in chunks of students rather than individual students
    for i in range(0, len(unique_students), chunk_size):
        chunk_students = unique_students[i:i + chunk_size]
        chunk_data = cleaned_vle_data[cleaned_vle_data['id_student'].isin(chunk_students)]
        
        # Create features for this chunk of students
        features = pd.DataFrame({
            'id_student': chunk_data['id_student'],
            'code_module': chunk_data['code_module'],
            'code_presentation': chunk_data['code_presentation'],
            'time_since_last': chunk_data.groupby('id_student')['date'].diff(),
            'cumulative_clicks': chunk_data.groupby('id_student')['sum_click'].cumsum(),
            'activity_type': chunk_data['activity_type'],
            'sum_click': chunk_data['sum_click'],
            'date': chunk_data['date'],
            'prev_activity': chunk_data.groupby('id_student')['activity_type'].shift()
        })
        
        yield features.fillna(0)


def prepare_dual_path_features(demographic_features, temporal_features,
                             assessment_features, sequential_features,
                             chunk_size: int = 50000):
    """Prepares dual-path features with chunked processing and categorical encoding."""
    
    # Define categorical columns for encoding
    categorical_cols = ['code_module', 'code_presentation', 'gender', 
                       'region', 'highest_education', 'imd_band', 'age_band']
    
    # Process static path in chunks
    static_chunks = []
    for chunk_start in range(0, len(demographic_features), chunk_size):
        chunk_end = chunk_start + chunk_size
        demo_chunk = demographic_features.iloc[chunk_start:chunk_end]
        
        # Encode categorical variables for each chunk
        encoded_chunk = demo_chunk.copy()
        for col in categorical_cols:
            if col in encoded_chunk.columns:
                # Label encoding
                encoded_chunk[f"{col}_encoded"] = pd.Categorical(encoded_chunk[col]).codes
                
                # One-hot encoding
                one_hot = pd.get_dummies(encoded_chunk[col], 
                                       prefix=col,
                                       drop_first=True)
                encoded_chunk = pd.concat([encoded_chunk, one_hot], axis=1)
        
        # Merge chunk with assessment features
        static_chunk = encoded_chunk.merge(
            assessment_features,
            on=['id_student', 'code_module', 'code_presentation'],
            how='inner'
        )
        static_chunks.append(static_chunk)
    
    # Combine static path chunks
    static_path = pd.concat(static_chunks, ignore_index=True)
    
    # Process sequential path
    # Add window column first
    window_size = 7  # Using 7-day window as specified
    sequential_features['window'] = sequential_features['date'] // window_size
    
    # Encode categorical variables in sequential features
    seq_categorical_cols = ['code_module', 'code_presentation', 'activity_type']
    for col in seq_categorical_cols:
        if col in sequential_features.columns:
            # Label encoding
            sequential_features[f"{col}_encoded"] = pd.Categorical(sequential_features[col]).codes
            
            # One-hot encoding
            one_hot = pd.get_dummies(sequential_features[col],
                                   prefix=col,
                                   drop_first=True)
            sequential_features = pd.concat([sequential_features, one_hot], axis=1)
    
    # Merge with temporal features
    sequential_path = sequential_features.merge(
        temporal_features['window_7'],
        on=['id_student', 'code_module', 'code_presentation', 'window'],
        how='inner'
    )
    
    # Log feature encoding info
    logger.info("Feature encoding summary:")
    logger.info(f"Static path categorical columns encoded: {categorical_cols}")
    logger.info(f"Sequential path categorical columns encoded: {seq_categorical_cols}")
    logger.info(f"Final static path shape: {static_path.shape}")
    logger.info(f"Final sequential path shape: {sequential_path.shape}")
    
    return {
        'static_path': static_path,
        'sequential_path': sequential_path
    }


def prepare_target_variable(data):
    """Prepare binary target variable from final_result."""
    # 0: Pass/Distinction (not at-risk)
    # 1: Fail/Withdrawn (at-risk)
    target_map = {
        'pass': 0,
        'distinction': 0,
        'fail': 1,
        'withdrawn': 1
    }
    
    return data['final_result'].str.lower().map(target_map)


def create_stratified_splits(dual_path_features, test_size=0.2, random_state=0):
    """Creates stratified train/test splits preserving demographic distributions."""
    
    # Extract static path features for stratification
    static_features = dual_path_features['static_path']
    
    # Create stratification columns
    static_features['strat_gender'] = static_features['gender']
    static_features['strat_age'] = static_features['age_band']
    static_features['strat_imd'] = static_features['imd_band'].apply(
        lambda x: x if pd.notna(x) else 'unknown'
    )
    
    # Create combined stratification column
    static_features['stratify_col'] = static_features['strat_gender'] + '_' + \
                                     static_features['strat_age'] + '_' + \
                                     static_features['strat_imd'].astype(str)
    
    # Get student-level data for stratification
    student_df = static_features[['id_student', 'stratify_col']].drop_duplicates()
    
    try:
        # Attempt stratified split
        train_ids, test_ids = train_test_split(
            student_df['id_student'],
            test_size=test_size,
            random_state=random_state,
            stratify=student_df['stratify_col']
        )
    except ValueError as e:
        logger.warning("Stratification failed due to insufficient samples. Falling back to random split.")
        # Fallback to random split if stratification fails
        train_ids, test_ids = train_test_split(
            student_df['id_student'],
            test_size=test_size,
            random_state=random_state
        )
    
    # Split static and sequential features
    static_train = static_features[static_features['id_student'].isin(train_ids)]
    static_test = static_features[static_features['id_student'].isin(test_ids)]
    
    # Print verification of split distribution
    logger.info("\nVerifying demographic distribution in train/test splits:")
    for col in ['gender', 'age_band', 'imd_band']:
        logger.info(f"\nDistribution of {col} in training set:")
        logger.info(static_train[col].value_counts(normalize=True))
        
        logger.info(f"\nDistribution of {col} in test set:")
        logger.info(static_test[col].value_counts(normalize=True))
    
    return {
        'static_train': static_train,
        'static_test': static_test,
        'train_ids': train_ids,
        'test_ids': test_ids
    }