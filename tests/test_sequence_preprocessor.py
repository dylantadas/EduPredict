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
from feature_engineering.sequential_features import SequentialFeatureProcessor
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
        
        # Create test module data
        self.courses_df = pd.DataFrame({
            'code_module': ['AAA', 'BBB', 'CCC'],
            'length': [180, 240, 120],
            'code_presentation': ['2013J', '2014B', '2013B']
        })
        
        # Create test VLE material data
        self.materials_df = pd.DataFrame({
            'id_site': range(1, 6),
            'code_module': ['AAA'] * 5,
            'code_presentation': ['2013J'] * 5,
            'activity_type': ['resource', 'quiz', 'forum', 'url', 'oucontent'],
            'week_from': [1, 1, 2, 2, 3],
            'week_to': [4, 6, 8, 5, 7]
        })
        
        # Create test interaction data
        dates = pd.date_range(start='2013-01-01', periods=100, freq='D')
        student_ids = np.repeat([1001, 1002, 1003, 1004, 1005], 20)
        activity_types = np.random.choice(['resource', 'quiz', 'forum', 'url', 'oucontent'], size=100)
        site_ids = np.random.choice(range(1, 6), size=100)
        
        self.test_df = pd.DataFrame({
            'id_student': student_ids,
            'code_module': 'AAA',
            'code_presentation': '2013J',
            'date': (dates - dates[0]).days,  # Convert to days from start
            'id_site': site_ids,
            'activity_type': activity_types,
            'click_count': np.random.randint(1, 50, size=100),
            'sum_click': np.random.randint(1, 50, size=100)
        })
        
        # Create test assessment data
        self.assessment_df = pd.DataFrame({
            'id_student': student_ids[:20],
            'code_module': 'AAA',
            'code_presentation': '2013J',
            'id_assessment': np.random.randint(1, 5, size=20),
            'date_submitted': np.random.randint(1, 180, size=20),
            'is_banked': np.random.choice([0, 1], size=20, p=[0.9, 0.1]),
            'score': np.random.uniform(0, 100, size=20),
            'assessment_type': np.random.choice(['TMA', 'CMA'], size=20)
        })
        
        # Create instance of SequencePreprocessor
        self.processor = SequentialFeatureProcessor(
            sequence_length=10,
            padding='pre',
            truncating='pre',
            normalize=True
        )

    def test_module_metadata(self):
        """Test module metadata handling"""
        self.processor.set_module_metadata(self.courses_df)
        self.assertEqual(len(self.processor.module_lengths), 3)
        self.assertEqual(self.processor.module_lengths['AAA'], 180)
        self.assertTrue('AAA' in self.processor.module_metadata)

    def test_process_vle_materials(self):
        """Test VLE material processing"""
        enriched = self.processor.process_vle_materials(
            self.test_df,
            self.materials_df,
            student_col='id_student'
        )
        self.assertIn('in_planned_window', enriched.columns)
        self.assertIn('timing_weight', enriched.columns)
        self.assertIn('activity_weight', enriched.columns)
        self.assertIn('effective_clicks', enriched.columns)
        
        # Verify weights are correctly applied
        quiz_rows = enriched[enriched['activity_type'] == 'quiz']
        self.assertTrue(all(quiz_rows['activity_weight'] == 2.0))

    def test_integrate_assessment_data(self):
        """Test assessment data integration"""
        combined = self.processor.integrate_assessment_data(
            self.test_df,
            self.assessment_df,
            student_col='id_student',
            time_col='date'
        )
        self.assertIn('is_assessment', combined.columns)
        self.assertIn('effective_score', combined.columns)
        
        # Verify banked assessments are weighted correctly
        banked_scores = combined[
            (combined['is_assessment']) & 
            (combined['is_banked'] == 1)
        ]['effective_score']
        original_scores = self.assessment_df[
            self.assessment_df['is_banked'] == 1
        ]['score']
        self.assertTrue(all(banked_scores == original_scores * 0.8))

    def test_sequence_validation(self):
        """Test sequence length validation against module constraints"""
        self.processor.set_module_metadata(self.courses_df)
        
        # Create test sequences that exceed module length
        test_sequences = {
            'sequence_data': np.random.rand(5, 200, 10),  # Sequences longer than AAA module
            'sequence_lengths': np.array([200] * 5),
            'metadata': {
                'code_module': ['AAA'] * 5
            }
        }
        
        validated = self.processor.validate_sequences(test_sequences)
        self.assertTrue(all(validated['sequence_lengths'] <= 180))  # AAA module length

    def test_full_pipeline(self):
        """Test complete sequence processing pipeline"""
        sequences = []
        for student_id in self.test_df['id_student'].unique():
            student_data = self.test_df[self.test_df['id_student'] == student_id]
            sequence = [{
                'type': 'vle',
                'activity': row['activity_type'],
                'date': row['date'],
                'sum_click': row['sum_click'],
                'code_module': row['code_module']
            } for _, row in student_data.iterrows()]
            if sequence:
                sequences.append(sequence)
        
        padded_sequences, auxiliary = self.processor.fit_transform(
            sequences=sequences,
            feature_columns=['type', 'activity']
        )
        
        # Verify sequence dimensions
        self.assertEqual(padded_sequences.shape[1], self.processor.sequence_length)
        self.assertIn('sequence_length', auxiliary)
        self.assertIn('interaction_count', auxiliary)

if __name__ == '__main__':
    unittest.main()