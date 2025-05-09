from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import logging
import json
from pathlib import Path
from config import FEATURE_ENGINEERING, DIRS
from utils.monitoring_utils import monitor_memory_usage

logger = logging.getLogger('edupredict')

class CategoricalEncoder:
    """
    Handles encoding of categorical variables with memory optimization
    and fairness considerations.
    """
    
    def __init__(
        self,
        encoding_type: str = 'hybrid',
        min_frequency: float = 0.01,
        handle_unknown: str = 'ignore',
        sparse: bool = False
    ):
        """
        Initialize the encoder.
        
        Args:
            encoding_type: Type of encoding ('label', 'onehot', or 'hybrid')
            min_frequency: Minimum frequency for a category to be treated as separate
            handle_unknown: How to handle unknown categories ('ignore' or 'error')
            sparse: Whether to return sparse matrix for one-hot encoding
        """
        self.encoding_type = encoding_type
        self.min_frequency = min_frequency
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}
        self.feature_names_: Dict[str, List[str]] = {}
        self.category_maps_: Dict[str, Dict] = {}
        
    @monitor_memory_usage
    def fit(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> 'CategoricalEncoder':
        """
        Fit the encoder on the data.
        
        Args:
            X: Input DataFrame
            columns: List of columns to encode. If None, encodes all object/category columns.
        """
        try:
            if columns is None:
                columns = X.select_dtypes(include=['object', 'category']).columns
                
            for col in columns:
                # Calculate value frequencies
                value_counts = X[col].value_counts(normalize=True)
                frequent_categories = value_counts[value_counts >= self.min_frequency].index
                
                # Map rare categories to 'Other'
                category_map = {cat: cat if cat in frequent_categories else 'Other' 
                              for cat in value_counts.index}
                self.category_maps_[col] = category_map
                
                # Apply mapping and fit encoders
                mapped_series = X[col].map(category_map)
                
                if self.encoding_type in ['label', 'hybrid']:
                    self.label_encoders[col] = LabelEncoder().fit(mapped_series.unique())
                
                if self.encoding_type in ['onehot', 'hybrid']:
                    self.onehot_encoders[col] = OneHotEncoder(
                        sparse=self.sparse,
                        handle_unknown=self.handle_unknown
                    ).fit(mapped_series.values.reshape(-1, 1))
                    
                    # Store feature names
                    categories = self.onehot_encoders[col].categories_[0]
                    self.feature_names_[col] = [f"{col}_{cat}" for cat in categories]
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting categorical encoder: {str(e)}")
            raise
            
    @monitor_memory_usage
    def transform(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Transform the data using the fitted encoder.
        
        Args:
            X: Input DataFrame
            columns: List of columns to encode. If None, uses columns from fit.
        """
        try:
            result = X.copy()
            
            if columns is None:
                columns = list(self.category_maps_.keys())
            
            for col in columns:
                if col not in self.category_maps_:
                    logger.warning(f"Column {col} was not fitted, skipping.")
                    continue
                
                # Map categories using fitted mapping
                mapped_series = X[col].map(self.category_maps_[col])
                mapped_series = mapped_series.fillna('Other')
                
                # Apply label encoding if needed
                if self.encoding_type in ['label', 'hybrid']:
                    label_encoded = pd.Series(
                        self.label_encoders[col].transform(mapped_series),
                        name=f"{col}_encoded"
                    )
                    result[f"{col}_encoded"] = label_encoded
                
                # Apply one-hot encoding if needed
                if self.encoding_type in ['onehot', 'hybrid']:
                    onehot_encoded = self.onehot_encoders[col].transform(
                        mapped_series.values.reshape(-1, 1)
                    )
                    
                    if self.sparse:
                        onehot_encoded = onehot_encoded.toarray()
                    
                    onehot_df = pd.DataFrame(
                        onehot_encoded,
                        columns=self.feature_names_[col],
                        index=X.index
                    )
                    
                    # Optimize memory by converting to more efficient dtypes
                    onehot_df = onehot_df.astype(np.int8)
                    
                    result = pd.concat([result, onehot_df], axis=1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
            
    def fit_transform(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit the encoder and transform the data in one step.
        """
        return self.fit(X, columns).transform(X, columns)
        
    def save_encoding_maps(self, output_dir: Union[str, Path]) -> None:
        """
        Save the encoding mappings to disk for later reference.
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                'encoding_type': self.encoding_type,
                'min_frequency': self.min_frequency,
                'category_maps': self.category_maps_,
                'feature_names': self.feature_names_
            }
            
            output_path = output_dir / 'categorical_encoding_maps.json'
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved encoding maps to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving encoding maps: {str(e)}")
            raise
            
    @classmethod
    def load_encoding_maps(cls, input_path: Union[str, Path]) -> Dict:
        """
        Load previously saved encoding mappings.
        """
        try:
            with open(input_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading encoding maps: {str(e)}")
            raise