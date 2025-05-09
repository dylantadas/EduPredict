from typing import Dict, List, Optional, Generator, Union, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from config import FEATURE_ENGINEERING, DATA_PROCESSING, DIRS, TEMPORAL_CONFIG
from utils.monitoring_utils import monitor_memory_usage, track_progress, track_execution_time
from feature_engineering.feature_selector import NumpyJSONEncoder
from feature_engineering.demographic_features import load_features, save_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger('edupredict')

class SequentialFeatureProcessor:
    """
    Processes sequential student interaction data into features suitable for ML models.
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        padding: str = 'pre',
        truncating: str = 'pre',
        normalize: bool = True
    ):
        """
        Initialize sequential feature processor.
        
        Args:
            sequence_length: Maximum length of sequences
            padding: Strategy for padding ('pre' or 'post')
            truncating: Strategy for truncating ('pre' or 'post')
            normalize: Whether to normalize sequences
        """
        self.sequence_length = sequence_length
        self.padding = padding
        self.truncating = truncating
        self.normalize = normalize
        
        self.feature_mapping_: Dict[str, int] = {}
        self.scaler_: Optional[MinMaxScaler] = None
        self.sequence_stats_: Dict = {}
        self.session_gap = TEMPORAL_CONFIG['session_gap_days']
        
    @monitor_memory_usage
    def fit_transform(
        self,
        sequences: List[List[Dict]],
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Transform sequence data into model-ready features.
        
        Args:
            sequences: List of interaction sequences per student
            feature_columns: List of features to extract from sequences
            
        Returns:
            Tuple of (padded sequences array, auxiliary features dict)
        """
        try:
            # Create feature mapping
            self._create_feature_mapping(sequences, feature_columns)
            
            # Transform sequences to numeric
            numeric_sequences = self._sequences_to_numeric(sequences)
            
            # Pad sequences
            padded_sequences = self._pad_sequences(numeric_sequences)
            
            # Extract auxiliary features
            auxiliary_features = self._extract_auxiliary_features(sequences)
            
            # Normalize if enabled
            if self.normalize:
                padded_sequences = self._normalize_sequences(padded_sequences)
                
            # Calculate sequence statistics
            self._calculate_sequence_stats(padded_sequences)
            
            return padded_sequences, auxiliary_features
            
        except Exception as e:
            logger.error(f"Error processing sequential features: {str(e)}")
            raise
            
    def _create_feature_mapping(
        self,
        sequences: List[List[Dict]],
        feature_columns: List[str]
    ) -> None:
        """
        Create mapping of features to numeric indices.
        """
        try:
            unique_features = set()
            
            for sequence in sequences:
                for interaction in sequence:
                    for col in feature_columns:
                        if col in interaction:
                            unique_features.add(f"{col}_{interaction[col]}")
                            
            self.feature_mapping_ = {
                feature: idx 
                for idx, feature in enumerate(sorted(unique_features))
            }
            
        except Exception as e:
            logger.error(f"Error creating feature mapping: {str(e)}")
            raise
            
    def _sequences_to_numeric(
        self,
        sequences: List[List[Dict]]
    ) -> List[np.ndarray]:
        """
        Convert sequences to numeric arrays using feature mapping.
        """
        try:
            numeric_sequences = []
            
            for sequence in sequences:
                # Split into sessions based on time gaps
                sessions = []
                current_session = []
                last_time = None
                
                for interaction in sequence:
                    current_time = interaction.get('date', None)
                    
                    if current_time is not None:
                        if last_time is not None and (current_time - last_time) > self.session_gap:
                            if current_session:
                                sessions.append(current_session)
                            current_session = []
                        current_session.append(interaction)
                        last_time = current_time
                
                if current_session:
                    sessions.append(current_session)
                
                # Process each session
                session_vectors = []
                for session in sessions:
                    session_vector = np.zeros(len(self.feature_mapping_))
                    for interaction in session:
                        for col, val in interaction.items():
                            feature_key = f"{col}_{val}"
                            if feature_key in self.feature_mapping_:
                                idx = self.feature_mapping_[feature_key]
                                session_vector[idx] = 1
                    session_vectors.append(session_vector)
                
                numeric_sequences.append(np.array(session_vectors))
            
            return numeric_sequences
            
        except Exception as e:
            logger.error(f"Error converting sequences to numeric: {str(e)}")
            raise
            
    def _pad_sequences(
        self,
        sequences: List[np.ndarray]
    ) -> np.ndarray:
        """
        Pad sequences to uniform length.
        """
        try:
            padded = np.zeros((
                len(sequences),
                self.sequence_length,
                len(self.feature_mapping_)
            ))
            
            for i, seq in enumerate(sequences):
                if len(seq) > self.sequence_length:
                    if self.truncating == 'pre':
                        seq = seq[-self.sequence_length:]
                    else:
                        seq = seq[:self.sequence_length]
                        
                if self.padding == 'post':
                    padded[i, :len(seq)] = seq
                else:
                    padded[i, -len(seq):] = seq
                    
            return padded
            
        except Exception as e:
            logger.error(f"Error padding sequences: {str(e)}")
            raise
            
    def _extract_auxiliary_features(
        self,
        sequences: List[List[Dict]]
    ) -> Dict[str, np.ndarray]:
        """
        Extract auxiliary features from sequences.
        """
        try:
            sequence_lengths = np.array([len(seq) for seq in sequences])
            interaction_counts = np.array([
                sum(1 for interaction in seq 
                    for val in interaction.values() 
                    if val is not None)
                for seq in sequences
            ])
            
            return {
                'sequence_length': sequence_lengths,
                'interaction_count': interaction_counts,
                'avg_interactions_per_step': interaction_counts / sequence_lengths
            }
            
        except Exception as e:
            logger.error(f"Error extracting auxiliary features: {str(e)}")
            raise
            
    def _normalize_sequences(
        self,
        sequences: np.ndarray
    ) -> np.ndarray:
        """
        Normalize sequence values.
        """
        try:
            if self.scaler_ is None:
                self.scaler_ = MinMaxScaler()
                
            # Reshape for scaling
            original_shape = sequences.shape
            flattened = sequences.reshape(-1, sequences.shape[-1])
            
            # Scale and reshape back
            normalized = self.scaler_.fit_transform(flattened)
            return normalized.reshape(original_shape)
            
        except Exception as e:
            logger.error(f"Error normalizing sequences: {str(e)}")
            raise
            
    def _calculate_sequence_stats(
        self,
        sequences: np.ndarray
    ) -> None:
        """
        Calculate statistics about sequences.
        """
        try:
            stats = {
                'mean_activation': np.mean(sequences, axis=(0, 1)),
                'std_activation': np.std(sequences, axis=(0, 1)),
                'max_activation': np.max(sequences, axis=(0, 1)),
                'sparsity': np.mean(sequences == 0)
            }
            
            self.sequence_stats_ = {
                str(idx): {
                    'mean': float(stats['mean_activation'][idx]),
                    'std': float(stats['std_activation'][idx]),
                    'max': float(stats['max_activation'][idx])
                }
                for idx in range(len(self.feature_mapping_))
            }
            
            self.sequence_stats_['global'] = {
                'sparsity': float(stats['sparsity'])
            }
            
        except Exception as e:
            logger.error(f"Error calculating sequence statistics: {str(e)}")
            raise
            
    def export_feature_metadata(
        self,
        output_dir: Union[str, Path]
    ) -> None:
        """
        Export metadata about sequential features.
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                'sequence_length': self.sequence_length,
                'feature_mapping': self.feature_mapping_,
                'sequence_stats': self.sequence_stats_,
                'timestamp': datetime.now().isoformat()
            }
            
            output_path = output_dir / 'sequential_features_metadata.json'
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Exported sequential feature metadata to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting sequential metadata: {str(e)}")
            raise

def calculate_sequence_similarity(
    seq1: np.ndarray,
    seq2: np.ndarray,
    method: str = 'cosine'
) -> float:
    """
    Calculate similarity between two sequences.
    
    Args:
        seq1: First sequence array
        seq2: Second sequence array
        method: Similarity metric ('cosine' or 'euclidean')
        
    Returns:
        Similarity score
    """
    try:
        if method == 'cosine':
            dot_product = np.sum(seq1 * seq2)
            norm_1 = np.sqrt(np.sum(seq1 ** 2))
            norm_2 = np.sqrt(np.sum(seq2 ** 2))
            
            if norm_1 == 0 or norm_2 == 0:
                return 0
                
            return dot_product / (norm_1 * norm_2)
            
        elif method == 'euclidean':
            return 1 / (1 + np.sqrt(np.sum((seq1 - seq2) ** 2)))
            
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
            
    except Exception as e:
        logger.error(f"Error calculating sequence similarity: {str(e)}")
        raise

def create_sequential_features(
    vle_data: pd.DataFrame,
    submission_data: pd.DataFrame,
    student_ids: Dict[str, List[str]],
    max_seq_length: Optional[int] = None,
    categorical_cols: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Create sequential features from VLE interactions and submissions.
    
    Args:
        vle_data: DataFrame with VLE interaction data
        submission_data: DataFrame with assessment submission data
        student_ids: Dictionary mapping split names to lists of student IDs
        max_seq_length: Maximum sequence length (defaults to config value)
        categorical_cols: List of categorical columns to encode
        logger: Logger instance
        
    Returns:
        Dictionary containing sequential features for each data split
    """
    try:
        if logger is None:
            logger = logging.getLogger('edupredict')
            
        if max_seq_length is None:
            max_seq_length = TEMPORAL_CONFIG['max_sequence_length']
            
        # Initialize sequence processor
        processor = SequentialFeatureProcessor(
            sequence_length=max_seq_length,
            padding=TEMPORAL_CONFIG['padding_strategy'],
            truncating=TEMPORAL_CONFIG['truncating_strategy'],
            normalize=TEMPORAL_CONFIG['normalize_sequences']
        )
        
        # Process each split
        results = {}
        for split_name, split_ids in student_ids.items():
            logger.info(f"Processing {split_name} split sequences...")
            
            # Filter data for this split
            split_vle = vle_data[vle_data['id_student'].isin(split_ids)]
            split_submissions = submission_data[submission_data['id_student'].isin(split_ids)]
            
            # Create sequences for each student
            sequences = []
            sequence_lengths = []
            processed_ids = []
            targets = []
            
            for student_id in split_ids:
                # Get student's interactions
                student_vle = split_vle[split_vle['id_student'] == student_id]
                student_submissions = split_submissions[split_submissions['id_student'] == student_id]
                
                if len(student_vle) == 0 and len(student_submissions) == 0:
                    continue
                    
                # Create sequence
                sequence = []
                
                # Add VLE interactions
                for _, row in student_vle.sort_values('date').iterrows():
                    interaction = {
                        'type': 'vle',
                        'activity': row['activity_type'],
                        'duration': row.get('length', 0)
                    }
                    sequence.append(interaction)
                    
                # Add submissions
                for _, row in student_submissions.sort_values('date_submitted').iterrows():
                    submission = {
                        'type': 'submission',
                        'assessment': row['id_assessment'],
                        'score': row.get('score', 0)
                    }
                    sequence.append(submission)
                
                if sequence:
                    sequences.append(sequence)
                    sequence_lengths.append(len(sequence))
                    processed_ids.append(student_id)
                    targets.append(student_submissions['is_banked'].max())
            
            # Transform sequences
            if sequences:
                padded_sequences, auxiliary_features = processor.fit_transform(
                    sequences=sequences,
                    feature_columns=categorical_cols or ['type', 'activity', 'assessment']
                )
                
                results[split_name] = {
                    'sequence_data': padded_sequences,
                    'sequence_lengths': sequence_lengths,
                    'student_ids': processed_ids,
                    'targets': targets,
                    'auxiliary_features': auxiliary_features,
                    'metadata': {
                        'feature_mapping': processor.feature_mapping_,
                        'sequence_stats': processor.sequence_stats_
                    }
                }
                
                logger.info(f"Created {len(sequences)} sequences for {split_name} split")
            else:
                logger.warning(f"No sequences created for {split_name} split")
                
        # Export metadata if output directory is configured
        if DIRS.get('feature_data'):
            processor.export_feature_metadata(DIRS['feature_data'])
            
        return results
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating sequential features: {str(e)}")
        raise