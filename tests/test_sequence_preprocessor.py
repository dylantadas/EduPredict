import os
import sys
import unittest
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from model_implementation.sequence_preprocessor import SequencePreprocessor
from config import DIRS, LOGGING

class TestSequencePreprocessor(unittest.TestCase):
    """Test cases for the SequencePreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Configure logging
        logging.basicConfig(
            level=LOGGING.get('level', logging.INFO),
            format=LOGGING.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger = logging.getLogger('test_sequence_preprocessor')
        
        # Create test data with temporal features
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        student_ids = np.repeat([1001, 1002, 1003, 1004, 1005], 20)
        
        # Create sample activity types
        activity_types = np.random.choice(
            ['resource', 'forum', 'quiz', 'assignment', 'video'],
            size=100
        )
        
        # Create numerical features
        click_counts = np.random.randint(1, 50, size=100)
        duration = np.random.randint(10, 3600, size=100)  # seconds
        
        # Create test dataframe
        self.test_df = pd.DataFrame({
            'id_student': student_ids,
            'date': dates,
            'activity_type': activity_types,
            'click_count': click_counts,
            'duration': duration
        })
        
        # Create static features dataframe
        self.static_features = pd.DataFrame({
            'id_student': [1001, 1002, 1003, 1004, 1005],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'age': [21, 19, 23, 20, 22],
            'previous_gpa': [3.2, 3.8, 2.9, 3.5, 3.3]
        })
        
        # Define column types
        self.categorical_cols = ['activity_type']
        self.numerical_cols = ['click_count', 'duration']
        
        # Create instance of SequencePreprocessor
        self.preprocessor = SequencePreprocessor(
            max_seq_length=10,
            mask_value=0.0,
            logger=self.logger
        )
    
    def test_initialization(self):
        """Test initialization of SequencePreprocessor"""
        self.assertEqual(self.preprocessor.max_seq_length, 10)
        self.assertEqual(self.preprocessor.mask_value, 0.0)
        self.assertFalse(self.preprocessor.fitted)
        self.assertEqual(self.preprocessor.categorical_cols, [])
        self.assertEqual(self.preprocessor.numerical_cols, [])
    
    def test_fit(self):
        """Test fit method"""
        # Fit the preprocessor
        self.preprocessor.fit(
            self.test_df,
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols
        )
        
        # Check that it's been fitted
        self.assertTrue(self.preprocessor.fitted)
        
        # Check that encoders and scalers have been created
        self.assertEqual(len(self.preprocessor.categorical_encoders), len(self.categorical_cols))
        self.assertEqual(len(self.preprocessor.numerical_scalers), len(self.numerical_cols))
        
        # Check that categorical encoder has correct structure
        self.assertIn('activity_type', self.preprocessor.categorical_encoders)
        
        # Check that numerical scalers have correct structure
        self.assertIn('click_count', self.preprocessor.numerical_scalers)
        self.assertIn('duration', self.preprocessor.numerical_scalers)
    
    def test_transform_sequences(self):
        """Test transform_sequences method"""
        # First fit the preprocessor
        self.preprocessor.fit(
            self.test_df,
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols
        )
        
        # Transform the sequences
        sequence_data = self.preprocessor.transform_sequences(
            self.test_df,
            student_col='id_student',
            time_col='date',
            static_features=self.static_features
        )
        
        # Check that the output has the expected structure
        self.assertIn('student_ids', sequence_data)
        self.assertIn('sequence_lengths', sequence_data)
        self.assertIn('mask', sequence_data)
        self.assertIn('cat_activity_type', sequence_data)
        self.assertIn('num_click_count', sequence_data)
        self.assertIn('num_duration', sequence_data)
        self.assertIn('static_features', sequence_data)
        
        # Check dimensions
        num_students = 5  # Number of unique students in test data
        max_seq_length = 10  # As defined in setUp
        
        self.assertEqual(len(sequence_data['student_ids']), num_students)
        self.assertEqual(sequence_data['mask'].shape, (num_students, max_seq_length))
        self.assertEqual(sequence_data['num_click_count'].shape, (num_students, max_seq_length))
        
        # Check that masks are correct (should be all 1s up to sequence_length, then 0s)
        for i in range(num_students):
            seq_length = sequence_data['sequence_lengths'][i]
            self.assertTrue(np.all(sequence_data['mask'][i, :seq_length] == 1.0))
            if seq_length < max_seq_length:
                self.assertTrue(np.all(sequence_data['mask'][i, seq_length:] == 0.0))
        
        # Check static features dimension
        self.assertEqual(sequence_data['static_features'].shape, (num_students, 3))  # 3 static features
    
    def test_prepare_temporal_features(self):
        """Test prepare_temporal_features method"""
        # Apply temporal feature creation
        temporal_df = self.preprocessor.prepare_temporal_features(
            self.test_df,
            student_col='id_student',
            time_col='date'
        )
        
        # Check that new columns have been added
        self.assertIn('time_since_last', temporal_df.columns)
        self.assertIn('time_since_start', temporal_df.columns)
        self.assertIn('day_of_week', temporal_df.columns)
        self.assertIn('is_weekend', temporal_df.columns)
        self.assertIn('event_count', temporal_df.columns)
        
        # Check that time features are computed correctly
        # For first entry of each student, time_since_last should be 0
        for student_id in self.test_df['id_student'].unique():
            student_rows = temporal_df[temporal_df['id_student'] == student_id].sort_values('date')
            self.assertEqual(student_rows['time_since_last'].iloc[0], pd.Timedelta(0))
            self.assertEqual(student_rows['time_since_start'].iloc[0], pd.Timedelta(0))
            self.assertEqual(student_rows['event_count'].iloc[0], 1)
    
    def test_add_interaction_features(self):
        """Test add_interaction_features method"""
        # Apply interaction feature creation
        interaction_df = self.preprocessor.add_interaction_features(
            self.test_df,
            student_col='id_student',
            interaction_cols=['activity_type', 'click_count']
        )
        
        # Check that new columns have been added (categorical counts)
        activity_types = self.test_df['activity_type'].unique()
        for activity in activity_types:
            self.assertIn(f'count_activity_type_{activity}', interaction_df.columns)
        
        # Check that numerical aggregations have been added
        self.assertIn('click_count_sum', interaction_df.columns)
        self.assertIn('click_count_mean', interaction_df.columns)
        self.assertIn('click_count_std', interaction_df.columns)
        self.assertIn('click_count_max', interaction_df.columns)
        
        # Verify counts for one student
        student_id = self.test_df['id_student'].unique()[0]
        student_data = self.test_df[self.test_df['id_student'] == student_id]
        
        for activity in activity_types:
            expected_count = len(student_data[student_data['activity_type'] == activity])
            actual_count = interaction_df[interaction_df['id_student'] == student_id][f'count_activity_type_{activity}'].iloc[0]
            self.assertEqual(expected_count, actual_count)
    
    def test_extract_event_patterns(self):
        """Test extract_event_patterns method"""
        # Apply pattern extraction
        patterns_df = self.preprocessor.extract_event_patterns(
            self.test_df,
            student_col='id_student',
            time_col='date',
            event_col='activity_type',
            n_gram_size=2
        )
        
        # Check that result is a dataframe with student IDs
        self.assertIsInstance(patterns_df, pd.DataFrame)
        self.assertIn('id_student', patterns_df.columns)
        self.assertEqual(len(patterns_df), len(self.test_df['id_student'].unique()))
        
        # Check that some pattern columns exist
        pattern_cols = [col for col in patterns_df.columns if col.startswith('pattern_')]
        self.assertTrue(len(pattern_cols) > 0)
    
    def test_save_load(self):
        """Test saving and loading preprocessor"""
        # First fit the preprocessor
        self.preprocessor.fit(
            self.test_df,
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols
        )
        
        # Save the preprocessor
        test_dir = Path(DIRS['models']) / 'test'
        test_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(test_dir / 'test_preprocessor.pkl')
        
        self.preprocessor.save(save_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(save_path.replace('.pkl', '.json')))
        
        # Load the preprocessor
        loaded_preprocessor = SequencePreprocessor.load(save_path, logger=self.logger)
        
        # Check that loaded preprocessor has same attributes
        self.assertEqual(loaded_preprocessor.max_seq_length, self.preprocessor.max_seq_length)
        self.assertEqual(loaded_preprocessor.categorical_cols, self.preprocessor.categorical_cols)
        self.assertEqual(loaded_preprocessor.numerical_cols, self.preprocessor.numerical_cols)
        self.assertTrue(loaded_preprocessor.fitted)
        
        # Check that encoders and scalers are working
        self.assertEqual(
            len(loaded_preprocessor.categorical_encoders['activity_type'].categories_[0]),
            len(self.preprocessor.categorical_encoders['activity_type'].categories_[0])
        )
        
        # Clean up
        os.remove(save_path)
        os.remove(save_path.replace('.pkl', '.json'))

if __name__ == '__main__':
    unittest.main()