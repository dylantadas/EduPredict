import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split

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


def create_sequential_features(cleaned_vle_data: pd.DataFrame) -> pd.DataFrame:
    """Creates sequential features for the gru/lstm path, maintaining temporal ordering."""

    # sort by student and time
    sequential_data = cleaned_vle_data.sort_values(['id_student', 'date'])

    # create time-based features
    sequential_data['time_since_last'] = sequential_data.groupby('id_student')['date'].diff()
    sequential_data['cumulative_clicks'] = sequential_data.groupby('id_student')['sum_click'].cumsum()

    # create activity transition features
    sequential_data['prev_activity'] = sequential_data.groupby('id_student')['activity_type'].shift()

    return sequential_data


def prepare_dual_path_features(demographic_features, temporal_features,
                             assessment_features, sequential_features,
                             chunk_size: int = 50000):
    """Prepares dual-path features with chunked processing."""

    # process static path in chunks
    static_chunks = []
    for chunk_start in range(0, len(demographic_features), chunk_size):
        chunk_end = chunk_start + chunk_size
        demo_chunk = demographic_features.iloc[chunk_start:chunk_end]

        # merge chunk with assessment features
        static_chunk = demo_chunk.merge(
            assessment_features,
            on=['id_student', 'code_module', 'code_presentation'],
            how='inner'
        )
        static_chunks.append(static_chunk)

    # combine static path chunks
    static_path = pd.concat(static_chunks, ignore_index=True)

    # process sequential path similarly
    sequential_path = sequential_features.merge(
        temporal_features['window_7'],
        on=['id_student', 'code_module', 'code_presentation', 'window'],
        how='inner'
    )

    return {
        'static_path': static_path,
        'sequential_path': sequential_path
    }


def prepare_target_variable(data):
    """Prepare binary target variable from final_result."""
    # Convert final_result to binary (Pass/Fail)
    # Distinction and Pass are considered positive outcomes (1)
    # Fail and Withdrawn are considered negative outcomes (0)
    return (data['final_result'].isin(['Pass', 'Distinction'])).astype(int)


def create_stratified_splits(data, test_size=0.2, random_state=42):
    """Create stratified train/test splits maintaining demographic balance."""
    from sklearn.model_selection import train_test_split
    
    # Define stratification columns (from config)
    strat_cols = ['gender', 'age_band', 'imd_band', 'final_result']
    
    # Create a combined stratification column
    strat = data[strat_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    
    # Split the data
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )
    
    # Create the splits
    split_data = {
        'static_train': data.iloc[train_idx],
        'static_test': data.iloc[test_idx],
        'train_ids': data.iloc[train_idx]['id_student'].values,
        'test_ids': data.iloc[test_idx]['id_student'].values
    }
    
    return split_data