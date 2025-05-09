import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import defaultdict

class SequencePreprocessor:
    """
    Preprocessor for transforming temporal/sequential student data into
    format suitable for GRU model training.
    
    This class handles:
    1. Grouping data by student
    2. Sorting by timestamp/date
    3. Handling variable-length sequences
    4. Encoding categorical features
    5. Scaling numerical features
    6. Creating padded sequences with masks
    """
    
    def __init__(
        self,
        max_seq_length: int = 100,
        mask_value: float = 0.0,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize SequencePreprocessor.
        
        Args:
            max_seq_length: Maximum length of sequences (will be padded/truncated)
            mask_value: Value to use for padding
            logger: Optional logger for messages
        """
        self.max_seq_length = max_seq_length
        self.mask_value = mask_value
        self.logger = logger or logging.getLogger('edupredict')
        
        # Initialize encoders/scalers
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.fitted = False
        
        # Track column names
        self.categorical_cols = []
        self.numerical_cols = []
        
        self.logger.info(f"Initialized SequencePreprocessor with max_seq_length={max_seq_length}")
    
    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str] = None,
        numerical_cols: List[str] = None
    ) -> 'SequencePreprocessor':
        """
        Fit preprocessor on a dataframe, creating encoders and scalers.
        
        Args:
            df: DataFrame containing student data
            categorical_cols: List of categorical column names to encode
            numerical_cols: List of numerical column names to scale
        
        Returns:
            Self for method chaining
        """
        try:
            # Store column names
            self.categorical_cols = categorical_cols or []
            self.numerical_cols = numerical_cols or []
            
            # Fit categorical encoders
            for col in self.categorical_cols:
                if col in df.columns:
                    unique_values = df[col].astype(str).unique()
                    self.logger.info(f"Fitting OneHotEncoder for '{col}' with {len(unique_values)} unique values")
                    
                    # Create and fit encoder
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoder.fit(df[[col]].astype(str))
                    
                    # Store encoder
                    self.categorical_encoders[col] = encoder
                else:
                    self.logger.warning(f"Column '{col}' not found in data")
            
            # Fit numerical scalers
            for col in self.numerical_cols:
                if col in df.columns:
                    self.logger.info(f"Fitting StandardScaler for '{col}'")
                    
                    # Create and fit scaler
                    scaler = StandardScaler()
                    # Reshape for sklearn compatibility
                    values = df[col].values.reshape(-1, 1)
                    scaler.fit(values)
                    
                    # Store scaler
                    self.numerical_scalers[col] = scaler
                else:
                    self.logger.warning(f"Column '{col}' not found in data")
            
            self.fitted = True
            self.logger.info("Sequence preprocessor fit completed")
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting sequence preprocessor: {str(e)}")
            raise
    
    def transform_sequences(
        self,
        df: pd.DataFrame,
        student_col: str = 'id_student',
        time_col: str = 'date',
        static_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """Transforms raw data into sequence features.
        
        Args:
            df: DataFrame with temporal/sequential data
            student_col: Column name for student IDs
            time_col: Column name for timestamps
            static_features: Optional DataFrame with static features
            
        Returns:
            Dictionary containing:
            - sequence_lengths: Length of each student's sequence
            - mask: Binary mask indicating valid sequence positions
            - categorical features: One-hot encoded categorical features
            - numerical features: Scaled numerical features
            - temporal_features: Time-based features
            - static_features: Optional static features per student
        """
        try:
            # Process temporal features first
            df_temporal = self.prepare_temporal_features(
                df,
                student_col=student_col,
                time_col=time_col
            )
            
            # Sort by student and time
            df_sorted = df_temporal.sort_values([student_col, time_col])
            
            # Get unique students
            student_ids = df_sorted[student_col].unique()
            student_groups = df_sorted.groupby(student_col)
            num_students = len(student_ids)
            
            self.logger.info(f"Processing sequences for {num_students} students")
            
            # Initialize result arrays
            result = {
                'student_ids': np.array(student_ids),
                'sequence_lengths': np.zeros(num_students, dtype=int)
            }
            
            # Initialize arrays for categorical features
            for col in self.categorical_cols:
                if col in df_sorted.columns and col in self.categorical_encoders:
                    encoder = self.categorical_encoders[col]
                    encoded_dim = len(encoder.categories_[0])
                    result[f'cat_{col}'] = np.zeros(
                        (num_students, self.max_seq_length, encoded_dim),
                        dtype=np.float32
                    )
            
            # Initialize arrays for numerical and temporal features
            numerical_cols = [col for col in df_sorted.columns if col in self.numerical_scalers]
            temporal_cols = ['time_since_last', 'time_since_start', 'day_of_week', 
                           'is_weekend', 'event_count']
            
            if numerical_cols:
                for col in numerical_cols:
                    result[f'num_{col}'] = np.zeros(
                        (num_students, self.max_seq_length),
                        dtype=np.float32
                    ) + self.mask_value
                    
            if temporal_cols:
                result['temporal_features'] = np.zeros(
                    (num_students, self.max_seq_length, len(temporal_cols)),
                    dtype=np.float32
                )
            
            # Create mask array
            result['mask'] = np.zeros((num_students, self.max_seq_length), dtype=np.float32)
            
            # Process each student group
            for i, student_id in enumerate(student_ids):
                # Get student data and sort by time
                student_data = student_groups.get_group(student_id)
                
                # Handle sequence length
                seq_length = min(len(student_data), self.max_seq_length)
                result['sequence_lengths'][i] = seq_length
                
                # Set mask for valid sequence positions
                result['mask'][i, :seq_length] = 1.0
                
                # Truncate if needed
                if len(student_data) > self.max_seq_length:
                    student_data = student_data.iloc[-self.max_seq_length:]
                
                # Process categorical features
                for col in self.categorical_cols:
                    if col in student_data.columns and col in self.categorical_encoders:
                        encoder = self.categorical_encoders[col]
                        encoded_values = encoder.transform(
                            student_data[col].astype(str).values.reshape(-1, 1)
                        ).reshape(-1, len(encoder.categories_[0]))
                        result[f'cat_{col}'][i, :seq_length] = encoded_values
                
                # Process numerical features
                for col in numerical_cols:
                    if col in self.numerical_scalers:
                        scaler = self.numerical_scalers[col]
                        scaled_values = scaler.transform(
                            student_data[col].values.reshape(-1, 1)
                        ).flatten()
                        result[f'num_{col}'][i, :seq_length] = scaled_values
                
                # Process temporal features
                if temporal_cols:
                    temporal_values = student_data[temporal_cols].values
                    result['temporal_features'][i, :seq_length] = temporal_values
            
            # Add static features if provided
            if static_features is not None:
                if student_col in static_features.columns:
                    static_df = static_features.set_index(student_col).loc[student_ids].reset_index()
                    static_cols = [col for col in static_df.columns if col != student_col]
                    if static_cols:
                        static_array = static_df[static_cols].values
                        result['static_features'] = static_array
                        self.logger.info(f"Added {static_array.shape[1]} static features")
                else:
                    self.logger.warning(f"Student column '{student_col}' not found in static_features")
            
            self.logger.info("Sequence transformation complete")
            return result
            
        except Exception as e:
            self.logger.error(f"Error transforming sequences: {str(e)}")
            raise
    
    def fit_transform_sequences(
        self,
        df: pd.DataFrame,
        student_col: str = 'id_student',
        time_col: str = 'date',
        categorical_cols: List[str] = None,
        numerical_cols: List[str] = None,
        static_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            df: DataFrame containing temporal/sequential data
            student_col: Column name containing student identifiers
            time_col: Column name containing timestamps/dates
            categorical_cols: List of categorical column names to encode
            numerical_cols: List of numerical column names to scale
            static_features: Optional DataFrame of static features for each student
        
        Returns:
            Dictionary of arrays for model training
        """
        self.fit(df, categorical_cols, numerical_cols)
        return self.transform_sequences(df, student_col, time_col, static_features)
    
    def prepare_temporal_features(
        self,
        df: pd.DataFrame,
        student_col: str = 'id_student',
        time_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Prepares temporal features like time between events.
        
        Args:
            df: DataFrame containing temporal data
            student_col: Column name containing student identifiers
            time_col: Column name containing timestamps/dates
        
        Returns:
            DataFrame with added temporal features
        """
        try:
            # Sort data by student and time
            result_df = df.sort_values([student_col, time_col]).copy()
            
            # Add time_since_last feature
            result_df['time_since_last'] = result_df.groupby(student_col)[time_col].diff().fillna(0)
            
            # Add time_since_start feature
            result_df['time_since_start'] = result_df.groupby(student_col)[time_col].transform(
                lambda x: x - x.iloc[0]
            ).fillna(0)
            
            # Add weekday feature if time_col represents a date
            if pd.api.types.is_datetime64_dtype(result_df[time_col]):
                result_df['day_of_week'] = result_df[time_col].dt.dayofweek
                result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add event_count feature (cumulative count of events per student)
            result_df['event_count'] = result_df.groupby(student_col).cumcount() + 1
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error preparing temporal features: {str(e)}")
            raise
    
    def add_interaction_features(
        self, 
        df: pd.DataFrame,
        student_col: str = 'id_student',
        interaction_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Adds interaction frequency features like counts per activity type.
        
        Args:
            df: DataFrame containing student interaction data
            student_col: Column name containing student identifiers
            interaction_cols: List of columns to create interaction features from
        
        Returns:
            DataFrame with added interaction features
        """
        try:
            result_df = df.copy()
            
            if not interaction_cols:
                # Use default columns if none specified
                potential_cols = ['activity_type', 'id_site', 'sum_click']
                interaction_cols = [col for col in potential_cols if col in df.columns]
            
            # Process each interaction column
            for col in interaction_cols:
                if col in df.columns:
                    # For categorical columns, create counts per category
                    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                        # Get counts per student and category
                        counts = df.groupby([student_col, col]).size().unstack(fill_value=0)
                        
                        # Rename columns to avoid conflicts
                        counts = counts.add_prefix(f'count_{col}_')
                        
                        # Merge back to result_df
                        result_df = pd.merge(
                            result_df,
                            counts.reset_index(),
                            on=student_col,
                            how='left'
                        )
                    
                    # For numerical columns, create aggregated stats
                    elif np.issubdtype(df[col].dtype, np.number):
                        # Calculate stats per student
                        stats = df.groupby(student_col)[col].agg(
                            ['sum', 'mean', 'std', 'max']
                        ).fillna(0)
                        
                        # Rename columns
                        stats.columns = [f'{col}_{stat}' for stat in stats.columns]
                        
                        # Merge back to result_df
                        result_df = pd.merge(
                            result_df,
                            stats.reset_index(),
                            on=student_col,
                            how='left'
                        )
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error adding interaction features: {str(e)}")
            raise
    
    def extract_event_patterns(
        self, 
        df: pd.DataFrame,
        student_col: str = 'id_student',
        time_col: str = 'date',
        event_col: str = 'activity_type',
        n_gram_size: int = 3
    ) -> pd.DataFrame:
        """
        Extracts event sequence patterns (n-grams) for each student.
        
        Args:
            df: DataFrame containing sequential data
            student_col: Column name containing student identifiers
            time_col: Column name containing timestamps/dates
            event_col: Column containing event types
            n_gram_size: Size of event sequence patterns to extract
        
        Returns:
            DataFrame with added pattern features
        """
        try:
            # Sort data by student and time
            sorted_df = df.sort_values([student_col, time_col])
            
            # Group by student
            student_groups = sorted_df.groupby(student_col)
            
            # Function to extract n-grams
            def get_ngrams(series, n=n_gram_size):
                values = series.astype(str).tolist()
                return ['_'.join(values[i:i+n]) for i in range(len(values)-n+1)]
            
            # Extract n-grams for each student
            patterns = {}
            pattern_counts = defaultdict(int)
            
            for student_id, group in student_groups:
                if len(group) >= n_gram_size:
                    student_patterns = get_ngrams(group[event_col])
                    patterns[student_id] = student_patterns
                    
                    # Count pattern occurrences
                    for pattern in student_patterns:
                        pattern_counts[pattern] += 1
            
            # Filter to top k most common patterns
            top_k = 20
            top_patterns = sorted(
                pattern_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k]
            
            # Create feature dataframe
            pattern_features = pd.DataFrame({student_col: sorted_df[student_col].unique()})
            
            # Add pattern features (binary indicators)
            for pattern, _ in top_patterns:
                pattern_features[f'pattern_{pattern}'] = pattern_features[student_col].map(
                    lambda sid: 1 if sid in patterns and pattern in patterns[sid] else 0
                )
            
            return pattern_features
            
        except Exception as e:
            self.logger.error(f"Error extracting event patterns: {str(e)}")
            raise
    
    def save(self, filepath: str) -> None:
        """
        Save preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save using pickle
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'categorical_encoders': self.categorical_encoders,
                    'numerical_scalers': self.numerical_scalers,
                    'categorical_cols': self.categorical_cols,
                    'numerical_cols': self.numerical_cols,
                    'fitted': self.fitted,
                    'max_seq_length': self.max_seq_length,
                    'mask_value': self.mask_value
                }, f)
            
            # Save metadata in JSON format for easier inspection
            metadata_path = filepath.replace('.pkl', '.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'max_seq_length': self.max_seq_length,
                    'mask_value': float(self.mask_value),
                    'fitted': self.fitted,
                    'categorical_cols': self.categorical_cols,
                    'numerical_cols': self.numerical_cols,
                    'categorical_encoders': {
                        col: {
                            'type': type(encoder).__name__,
                            'n_features': len(encoder.categories_[0]),
                            'categories': [str(c) for c in encoder.categories_[0]]
                        } for col, encoder in self.categorical_encoders.items()
                    },
                    'numerical_scalers': {
                        col: {
                            'type': type(scaler).__name__,
                            'mean': float(scaler.mean_[0]),
                            'scale': float(scaler.scale_[0])
                        } for col, scaler in self.numerical_scalers.items()
                    }
                }, f, indent=2)
            
            self.logger.info(f"Saved sequence preprocessor to {filepath}")
            self.logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str, logger: Optional[logging.Logger] = None) -> 'SequencePreprocessor':
        """
        Load preprocessor from disk.
        
        Args:
            filepath: Path to load the preprocessor from
            logger: Optional logger
        
        Returns:
            Loaded SequencePreprocessor instance
        """
        try:
            logger = logger or logging.getLogger('edupredict')
            logger.info(f"Loading sequence preprocessor from {filepath}")
            
            # Load using pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Create instance
            instance = cls(
                max_seq_length=data['max_seq_length'],
                mask_value=data['mask_value'],
                logger=logger
            )
            
            # Restore attributes
            instance.categorical_encoders = data['categorical_encoders']
            instance.numerical_scalers = data['numerical_scalers']
            instance.categorical_cols = data['categorical_cols']
            instance.numerical_cols = data['numerical_cols']
            instance.fitted = data['fitted']
            
            logger.info(f"Successfully loaded preprocessor with {len(instance.categorical_cols)} categorical and {len(instance.numerical_cols)} numerical features")
            
            return instance
            
        except Exception as e:
            if logger:
                logger.error(f"Error loading preprocessor: {str(e)}")
            raise