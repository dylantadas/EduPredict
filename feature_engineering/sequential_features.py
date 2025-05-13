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
    Handles OULAD-specific sequence processing and feature engineering.
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        padding: str = 'pre',
        truncating: str = 'pre',
        normalize: bool = True
    ):
        """Initialize sequential feature processor.
        
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
        
        # Module metadata
        self.module_lengths: Dict[str, int] = {}  # Track module durations
        self.module_metadata: Dict[str, Any] = {} # Additional module info
        
        # Activity weighting from OULAD documentation
        self.vle_activity_weights = {
            'resource': 1.0,
            'url': 1.0,
            'quiz': 2.0,     # Weight interactive activities higher
            'forum': 1.5,
            'oucontent': 1.0,
            'subpage': 1.0,
            'homepage': 0.5,
            'page': 1.0,
            'questionnaire': 1.5,
            'ouelluminate': 2.0,
            'sharedsubpage': 1.0,
            'externalquiz': 2.0,
            'dataplus': 1.5,
            'glossary': 1.0,
            'htmlactivity': 1.5,
            'oucollaborate': 2.0,
            'dualpane': 1.0
        }
    
    def validate_sequences(
        self,
        sequences: Dict[str, np.ndarray],
        module_col: str = 'code_module'
    ) -> Dict[str, np.ndarray]:
        """Validates sequences against module duration constraints.
        
        Args:
            sequences: Dictionary containing sequence data and metadata
            module_col: Column containing module codes
            
        Returns:
            Dictionary with validated sequences
        """
        try:
            if not self.module_lengths:
                logger.warning("No module metadata set - skipping validation")
                return sequences
                
            result = sequences.copy()
            
            # Get module for each sequence
            if 'metadata' in sequences:
                module_data = sequences['metadata'].get(module_col)
                if module_data is not None:
                    for i, module in enumerate(module_data):
                        if module in self.module_lengths:
                            max_length = self.module_lengths[module]
                            
                            # Trim sequence lengths if needed
                            if result['sequence_lengths'][i] > max_length:
                                result['sequence_lengths'][i] = max_length
                                
                                # Update masks
                                if 'mask' in result:
                                    result['mask'][i, max_length:] = 0
                                    
                                # Update feature arrays
                                for key in result:
                                    if isinstance(result[key], np.ndarray) and \
                                       len(result[key].shape) == 3 and \
                                       result[key].shape[0] == len(module_data):
                                        result[key][i, max_length:] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating sequences: {str(e)}")
            raise
    
    def set_module_metadata(self, courses_df: pd.DataFrame) -> None:
        """Set module metadata for sequence validation.
        
        Args:
            courses_df: DataFrame containing course/module information
        """
        if 'code_module' not in courses_df.columns or 'length' not in courses_df.columns:
            raise ValueError("courses_df must contain 'code_module' and 'length' columns")
            
        self.module_lengths = dict(zip(courses_df['code_module'], courses_df['length']))
        self.module_metadata = courses_df.set_index('code_module').to_dict('index')
    
    def integrate_assessment_data(
        self,
        vle_data: pd.DataFrame,
        assessment_data: pd.DataFrame,
        student_col: str = 'id_student',
        time_col: str = 'date'
    ) -> pd.DataFrame:
        """Integrate assessment data with VLE interactions.
        
        Args:
            vle_data: DataFrame with VLE interactions
            assessment_data: DataFrame with assessment submissions
            student_col: Column containing student IDs
            time_col: Column for timestamps
            
        Returns:
            Combined DataFrame with integrated assessment and VLE data
        """
        try:
            required_cols = {
                'vle': [student_col, time_col, 'activity_type', 'sum_click'],
                'assessment': [student_col, 'date_submitted', 'score', 'is_banked', 'assessment_type']
            }
            
            for df_name, cols in required_cols.items():
                df = vle_data if df_name == 'vle' else assessment_data
                missing = [col for col in cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing required columns in {df_name} data: {missing}")
            
            # Process assessment features
            assessment_features = assessment_data.copy()
            assessment_features['effective_score'] = np.where(
                assessment_features['is_banked'],
                assessment_features['score'] * 0.8,  # Reduce weight of banked assessments
                assessment_features['score']
            )
            assessment_features[time_col] = assessment_features['date_submitted']
            assessment_features['activity_type'] = assessment_features['assessment_type']
            assessment_features['sum_click'] = 1  # Each submission counts as one interaction
            assessment_features['is_assessment'] = True
            
            # Mark VLE interactions
            vle_data = vle_data.copy()
            vle_data['is_assessment'] = False
            vle_data['effective_score'] = 0
            
            # Combine and sort
            common_cols = [col for col in vle_data.columns if col in assessment_features.columns]
            combined = pd.concat(
                [vle_data[common_cols], assessment_features[common_cols]], 
                ignore_index=True
            )
            return combined.sort_values([student_col, time_col])
            
        except Exception as e:
            logger.error(f"Error integrating assessment data: {str(e)}")
            raise
            
    def process_vle_materials(
        self,
        interactions: pd.DataFrame,
        materials: pd.DataFrame,
        student_col: str = 'id_student'
    ) -> pd.DataFrame:
        """Enrich VLE interactions with material context.
        
        Args:
            interactions: DataFrame with student VLE interactions
            materials: DataFrame with VLE material information
            student_col: Column containing student IDs
            
        Returns:
            Enriched DataFrame with material context and weighted interactions
        """
        try:
            required_cols = {
                'interactions': [student_col, 'id_site', 'date', 'sum_click'],
                'materials': ['id_site', 'activity_type', 'week_from', 'week_to']
            }
            
            for df_name, cols in required_cols.items():
                df = interactions if df_name == 'interactions' else materials
                missing = [col for col in cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing required columns in {df_name} data: {missing}")
            
            # Merge with material info
            enriched = interactions.merge(
                materials,
                on=['id_site'],
                how='left'
            )
            
            # Calculate planned windows
            enriched['in_planned_window'] = (
                (enriched['date'] >= enriched['week_from'] * 7) &
                (enriched['date'] <= enriched['week_to'] * 7)
            )
            
            # Apply weights
            enriched['timing_weight'] = np.where(
                enriched['in_planned_window'],
                1.0,  # Normal weight for on-time interactions
                0.7   # Reduced weight for out-of-window interactions
            )
            enriched['activity_weight'] = enriched['activity_type'].map(
                self.vle_activity_weights
            ).fillna(1.0)
            
            # Calculate effective interactions
            if 'sum_click' in enriched.columns:
                enriched['effective_clicks'] = (
                    enriched['sum_click'] * 
                    enriched['timing_weight'] * 
                    enriched['activity_weight']
                )
            
            return enriched
            
        except Exception as e:
            logger.error(f"Error processing VLE materials: {str(e)}")
            raise

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
    logger: Optional[logging.Logger] = None,
    courses_df: Optional[pd.DataFrame] = None,
    vle_materials_df: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, Any]]:
    """Create sequential features from VLE interactions and submissions.
    
    Args:
        vle_data: DataFrame with VLE interaction data
        submission_data: DataFrame with assessment submission data
        student_ids: Dictionary mapping split names to lists of student IDs
        max_seq_length: Maximum sequence length (defaults to config value)
        categorical_cols: List of categorical columns to encode
        logger: Logger instance
        courses_df: DataFrame containing course metadata
        vle_materials_df: DataFrame containing VLE material information
        
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
        
        # Set module metadata if available
        if courses_df is not None:
            processor.set_module_metadata(courses_df)
            logger.info("Loaded module metadata for sequence validation")
        
        # Process each split
        results = {}
        for split_name, split_ids in student_ids.items():
            logger.info(f"Processing {split_name} split sequences...")
            
            # Filter data for this split
            split_vle = vle_data[vle_data['id_student'].isin(split_ids)]
            split_submissions = submission_data[submission_data['id_student'].isin(split_ids)]
            
            # Process VLE materials if available
            if vle_materials_df is not None:
                split_vle = processor.process_vle_materials(
                    split_vle, 
                    vle_materials_df,
                    student_col='id_student'
                )
                logger.info("Enriched VLE interactions with material context")
            
            # Integrate assessment data
            combined_data = processor.integrate_assessment_data(
                split_vle,
                split_submissions,
                student_col='id_student',
                time_col='date'
            )
            logger.info("Integrated assessment data with VLE interactions")
            
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
                
                # Create chronological sequence of all events (VLE + submissions)
                sequence = []
                
                # Process VLE interactions - make sure to include date
                if len(student_vle) > 0:
                    for _, row in student_vle.iterrows():
                        interaction = {
                            'type': 'vle',
                            'activity': row.get('activity_type', 'unknown'),
                            'date': row['date'],  # Days relative to module start
                            'sum_click': row.get('sum_click', 1),
                            'is_pre_module': row['date'] < 0,
                            'week_number': row['date'] // 7,
                            'day_of_week': row['date'] % 7
                        }
                        
                        # Add site ID if available
                        if 'id_site' in row:
                            interaction['site_id'] = row['id_site']
                            
                        sequence.append(interaction)
                
                # Process assessment submissions - make sure to include date
                if len(student_submissions) > 0:
                    for _, row in student_submissions.iterrows():
                        submission = {
                            'type': 'submission',
                            'assessment': row.get('id_assessment', -1),
                            'date': row['date_submitted'],  # Days relative to module start
                            'score': row.get('score', 0),
                            'is_banked': row.get('is_banked', 0),
                            'week_number': row['date_submitted'] // 7,
                            'day_of_week': row['date_submitted'] % 7
                        }
                        sequence.append(submission)
                
                if sequence:
                    # Sort sequence by date (chronologically including pre-module events)
                    sequence.sort(key=lambda x: x['date'])
                    
                    # Calculate temporal intervals between events
                    for i in range(1, len(sequence)):
                        sequence[i]['time_since_last'] = sequence[i]['date'] - sequence[i-1]['date']
                        
                    # Group events by time windows to create meaningful sessions
                    session_gap = TEMPORAL_CONFIG.get('session_gap_days', 1)
                    sessions = []
                    current_session = []
                    
                    for event in sequence:
                        # Start new session if this is first event or gap exceeds threshold
                        if not current_session or event.get('time_since_last', 0) > session_gap:
                            if current_session:
                                sessions.append(current_session)
                            current_session = [event]
                        else:
                            current_session.append(event)
                    
                    if current_session:
                        sessions.append(current_session)
                    
                    # Create features for each session to prepare for sequence modeling
                    processed_sequence = []
                    for session in sessions:
                        session_features = {
                            'type': 'session',
                            'date': session[0]['date'],  # Use first event date as session date
                            'is_pre_module': session[0]['date'] < 0,
                            'week_number': session[0]['date'] // 7,
                            'day_of_week': session[0]['date'] % 7,
                            'event_count': len(session),
                            'vle_count': sum(1 for e in session if e['type'] == 'vle'),
                            'submission_count': sum(1 for e in session if e['type'] == 'submission'),
                            'total_clicks': sum(e.get('sum_click', 0) for e in session if e['type'] == 'vle'),
                            'avg_score': np.mean([e.get('score', 0) for e in session if e['type'] == 'submission' and e.get('score') is not None]) if any(e['type'] == 'submission' for e in session) else 0,
                            'duration': session[-1]['date'] - session[0]['date'],
                            'activity_types': '_'.join(sorted(set(e.get('activity', 'unknown') for e in session if e['type'] == 'vle')))
                        }
                        processed_sequence.append(session_features)
                    
                    # Capture the sequence
                    sequences.append(processed_sequence)
                    sequence_lengths.append(len(processed_sequence))
                    processed_ids.append(student_id)
                    
                    # Determine target for this sequence
                    # Use final result if available (aggregating through submissions data)
                    if len(student_submissions) > 0:
                        # For now, use if the student banked any assessments as a proxy for success
                        # This should be replaced with actual final_result if available
                        targets.append(int(any(s['is_banked'] == 1 for s in student_submissions.to_dict('records'))))
                    else:
                        targets.append(0)  # Default target if no submissions available
            
            # Transform and validate sequences
            if sequences:
                cat_cols = categorical_cols or ['type', 'activity_types', 'is_pre_module']
                
                padded_sequences, auxiliary_features = processor.fit_transform(
                    sequences=sequences,
                    feature_columns=cat_cols
                )
                
                # Validate sequences against module constraints
                sequence_metadata = {
                    'code_module': [seq[0].get('code_module') for seq in sequences if seq]
                }
                validated_sequences = processor.validate_sequences(
                    {
                        'sequence_data': padded_sequences,
                        'sequence_lengths': sequence_lengths,
                        'metadata': sequence_metadata
                    }
                )
                
                results[split_name] = {
                    'sequence_data': validated_sequences['sequence_data'],
                    'sequence_lengths': validated_sequences['sequence_lengths'],
                    'student_ids': processed_ids,
                    'targets': targets,
                    'auxiliary_features': auxiliary_features,
                    'metadata': {
                        'feature_mapping': processor.feature_mapping_,
                        'sequence_stats': processor.sequence_stats_
                    }
                }
                
                logger.info(f"Created {len(sequences)} sequences for {split_name} split")
                logger.info(f"Average sequence length: {np.mean(sequence_lengths):.2f}")
                logger.info(f"Max sequence length: {np.max(sequence_lengths)}")
                logger.info(f"Min sequence length: {np.min(sequence_lengths)}")
                
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