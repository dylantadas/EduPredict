import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class SequencePreprocessor:
    """
    Base sequence preprocessor for transforming variable-length sequences into fixed-length
    tensor representations suitable for deep learning models.
    
    This class handles:
    1. Sequence padding and truncation
    2. Feature encoding and scaling
    3. Masking for variable length sequences
    """
    
    def __init__(
        self,
        max_seq_length: int = 100,
        mask_value: float = 0.0,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize sequence preprocessor.
        
        Args:
            max_seq_length: Maximum sequence length for padding/truncation
            mask_value: Value to use for padding
            logger: Optional logger instance
        """
        self.max_seq_length = max_seq_length
        self.mask_value = mask_value
        self.logger = logger or logging.getLogger('edupredict')
        
        # Initialize preprocessors
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.fitted = False
        
        # Track columns
        self.categorical_cols = []
        self.numerical_cols = []
    
    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str] = None,
        numerical_cols: List[str] = None
    ) -> 'SequencePreprocessor':
        """Fit preprocessor on training data.
        
        Args:
            df: DataFrame containing sequences
            categorical_cols: Categorical column names to encode
            numerical_cols: Numerical column names to scale
        """
        try:
            self.categorical_cols = categorical_cols or []
            self.numerical_cols = numerical_cols or []
            
            # Fit categorical encoders
            for col in self.categorical_cols:
                if col in df.columns:
                    self.logger.info(f"Fitting OneHotEncoder for '{col}'")
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoder.fit(df[[col]].astype(str))
                    self.categorical_encoders[col] = encoder
            
            # Fit numerical scalers  
            for col in self.numerical_cols:
                if col in df.columns:
                    self.logger.info(f"Fitting StandardScaler for '{col}'")
                    scaler = StandardScaler()
                    values = df[col].values.reshape(-1, 1)
                    scaler.fit(values)
                    self.numerical_scalers[col] = scaler
            
            self.fitted = True
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting preprocessor: {str(e)}")
            raise
    
    def transform_sequences(
        self,
        df: pd.DataFrame,
        student_col: str = 'id_student',
        time_col: str = 'date'
    ) -> Dict[str, np.ndarray]:
        """Transform sequences into model-ready format.
        
        Args:
            df: DataFrame containing sequences
            student_col: Column containing student IDs 
            time_col: Column containing timestamps
            
        Returns:
            Dictionary containing:
            - sequence_data: (num_students, max_seq_length, num_features) array
            - sequence_lengths: Length of each sequence before padding
            - mask: Binary mask indicating valid positions
        """
        try:
            if not self.fitted:
                raise ValueError("Preprocessor must be fitted before transforming")
            
            # Sort by student and time
            df_sorted = df.sort_values([student_col, time_col])
            student_ids = df_sorted[student_col].unique()
            student_groups = df_sorted.groupby(student_col)
            num_students = len(student_ids)
            
            # Initialize output
            result = {
                'student_ids': student_ids,
                'sequence_lengths': np.zeros(num_students, dtype=int),
                'mask': np.zeros((num_students, self.max_seq_length), dtype=np.float32)
            }
            
            # Initialize feature arrays
            feature_dim = 0
            for col in self.categorical_cols:
                if col in self.categorical_encoders:
                    encoder = self.categorical_encoders[col]
                    dim = len(encoder.categories_[0])
                    result[f'cat_{col}'] = np.zeros(
                        (num_students, self.max_seq_length, dim),
                        dtype=np.float32
                    )
                    feature_dim += dim
            
            for col in self.numerical_cols:
                if col in self.numerical_scalers:
                    result[f'num_{col}'] = np.zeros(
                        (num_students, self.max_seq_length),
                        dtype=np.float32
                    )
                    feature_dim += 1
            
            # Process each student's sequence
            for i, student_id in enumerate(student_ids):
                student_data = student_groups.get_group(student_id)
                seq_length = min(len(student_data), self.max_seq_length)
                result['sequence_lengths'][i] = seq_length
                
                # Set mask for valid positions
                result['mask'][i, :seq_length] = 1.0
                
                # Truncate if needed
                if len(student_data) > self.max_seq_length:
                    student_data = student_data.iloc[-self.max_seq_length:]
                
                # Transform categorical features
                for col in self.categorical_cols:
                    if col in student_data.columns and col in self.categorical_encoders:
                        encoder = self.categorical_encoders[col]
                        encoded = encoder.transform(
                            student_data[col].astype(str).values.reshape(-1, 1)
                        )
                        result[f'cat_{col}'][i, :seq_length] = encoded
                
                # Transform numerical features
                for col in self.numerical_cols:
                    if col in student_data.columns and col in self.numerical_scalers:
                        scaler = self.numerical_scalers[col]
                        scaled = scaler.transform(
                            student_data[col].values.reshape(-1, 1)
                        ).flatten()
                        result[f'num_{col}'][i, :seq_length] = scaled
            
            # Combine all features into single array
            all_features = []
            for key in result:
                if key.startswith(('cat_', 'num_')):
                    if len(result[key].shape) == 3:
                        reshaped = result[key].reshape(num_students, self.max_seq_length, -1)
                    else:
                        reshaped = result[key].reshape(num_students, self.max_seq_length, 1)
                    all_features.append(reshaped)
            
            if all_features:
                result['sequence_data'] = np.concatenate(all_features, axis=2)
            
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
        numerical_cols: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """Fit preprocessor and transform sequences in one step."""
        self.fit(df, categorical_cols, numerical_cols)
        return self.transform_sequences(df, student_col, time_col)
    
    def save(self, filepath: str) -> None:
        """Save preprocessor state to disk."""
        try:
            # Save preprocessor objects
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'max_seq_length': self.max_seq_length,
                    'mask_value': self.mask_value,
                    'categorical_encoders': self.categorical_encoders,
                    'numerical_scalers': self.numerical_scalers,
                    'categorical_cols': self.categorical_cols,
                    'numerical_cols': self.numerical_cols,
                    'fitted': self.fitted
                }, f)
            
            # Save readable metadata
            metadata_path = filepath.replace('.pkl', '.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'max_seq_length': self.max_seq_length,
                    'mask_value': float(self.mask_value),
                    'categorical_cols': self.categorical_cols,
                    'numerical_cols': self.numerical_cols,
                    'fitted': self.fitted
                }, f, indent=2)
                
            self.logger.info(f"Saved preprocessor to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str, logger: Optional[logging.Logger] = None) -> 'SequencePreprocessor':
        """Load preprocessor from disk."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            instance = cls(
                max_seq_length=data['max_seq_length'],
                mask_value=data['mask_value'],
                logger=logger
            )
            instance.categorical_encoders = data['categorical_encoders']
            instance.numerical_scalers = data['numerical_scalers']
            instance.categorical_cols = data['categorical_cols']
            instance.numerical_cols = data['numerical_cols']
            instance.fitted = data['fitted']
            
            return instance
            
        except Exception as e:
            if logger:
                logger.error(f"Error loading preprocessor: {str(e)}")
            raise