import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# import functions to test
from data_processing.data_processing import (
    load_raw_datasets,
    clean_demographic_data,
    clean_vle_data,
    clean_assessment_data,
    validate_data_consistency
)

from data_processing.feature_engineering import (
    create_demographic_features,
    create_temporal_features,
    create_assessment_features,
    create_sequential_features,
    prepare_dual_path_features,
    create_stratified_splits,
    prepare_target_variable
)


class TestDataLoading(unittest.TestCase):
    """Tests for data loading functions."""

    def setUp(self):
        """Create sample data for testing."""
        # create temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = self.temp_dir.name
        
        # create sample csv files
        self.student_info = pd.DataFrame({
            'id_student': [1, 2, 3, 4, 5],
            'code_module': ['AAA', 'BBB', 'AAA', 'CCC', 'BBB'],
            'code_presentation': ['2020J', '2020J', '2020B', '2020B', '2020B'],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'region': ['East Anglian', 'London', 'Scotland', 'Wales', 'North'],
            'highest_education': ['A Level', 'HE Qualification', 'A Level', 'No Formal', 'HE Qualification'],
            'imd_band': ['0-10%', '20-30%', '30-40%', '50-60%', '90-100%'],
            'age_band': ['0-35', '0-35', '35-55', '35-55', '55<='],
            'num_of_prev_attempts': [0, 1, 0, 2, 1],
            'studied_credits': [60, 60, 30, 60, 30],
            'disability': ['N', 'Y', 'N', 'N', 'Y'],
            'final_result': ['Pass', 'Fail', 'Distinction', 'Withdrawn', 'Pass']
        })
        
        self.vle_interactions = pd.DataFrame({
            'id_student': [1, 1, 2, 2, 3, 4, 5],
            'code_module': ['AAA', 'AAA', 'BBB', 'BBB', 'AAA', 'CCC', 'BBB'],
            'code_presentation': ['2020J', '2020J', '2020J', '2020J', '2020B', '2020B', '2020B'],
            'id_site': [1, 2, 1, 3, 2, 3, 1],
            'date': [10, 15, 5, 20, 30, 25, 40],
            'sum_click': [5, 10, 8, 3, 12, 7, 9]
        })
        
        self.vle_materials = pd.DataFrame({
            'id_site': [1, 2, 3],
            'code_module': ['AAA', 'BBB', 'CCC'],
            'code_presentation': ['2020J', '2020J', '2020B'],
            'activity_type': ['resource', 'quiz', 'forum'],
            'week_from': [1, 2, 3],
            'week_to': [2, 3, 4]
        })
        
        self.assessments = pd.DataFrame({
            'code_module': ['AAA', 'BBB', 'CCC'],
            'code_presentation': ['2020J', '2020J', '2020B'],
            'id_assessment': [1, 2, 3],
            'assessment_type': ['TMA', 'CMA', 'Exam'],
            'date': [30, 60, 90],
            'weight': [25, 25, 50]
        })
        
        self.student_assessments = pd.DataFrame({
            'id_assessment': [1, 1, 2, 2, 3],
            'id_student': [1, 2, 2, 3, 4],
            'date_submitted': [25, 28, 55, 58, 85],
            'is_banked': [0, 0, 0, 0, 0],
            'score': [85, 65, 75, 90, 45]
        })
        
        self.courses = pd.DataFrame({
            'code_module': ['AAA', 'BBB', 'CCC'],
            'code_presentation': ['2020J', '2020J', '2020B'],
            'module_presentation_length': [240, 240, 180]
        })
        
        # write sample data to csv files
        self.student_info.to_csv(os.path.join(self.data_path, 'studentInfo.csv'), index=False)
        self.vle_interactions.to_csv(os.path.join(self.data_path, 'studentVle.csv'), index=False)
        self.vle_materials.to_csv(os.path.join(self.data_path, 'vle.csv'), index=False)
        self.assessments.to_csv(os.path.join(self.data_path, 'assessments.csv'), index=False)
        self.student_assessments.to_csv(os.path.join(self.data_path, 'studentAssessment.csv'), index=False)
        self.courses.to_csv(os.path.join(self.data_path, 'courses.csv'), index=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_load_raw_datasets(self):
        """Tests loading raw datasets from csv files."""
        # test loading datasets
        datasets = load_raw_datasets(self.data_path)
        
        # check if all datasets were loaded
        expected_keys = ['student_info', 'vle_interactions', 'vle_materials', 
                        'assessments', 'student_assessments', 'courses']
        self.assertEqual(set(datasets.keys()), set(expected_keys))
        
        # check if datasets have correct number of rows
        self.assertEqual(len(datasets['student_info']), 5)
        self.assertEqual(len(datasets['vle_interactions']), 7)
        self.assertEqual(len(datasets['vle_materials']), 3)
        self.assertEqual(len(datasets['assessments']), 3)
        self.assertEqual(len(datasets['student_assessments']), 5)
        self.assertEqual(len(datasets['courses']), 3)
    
    def test_clean_demographic_data(self):
        """Tests cleaning demographic data."""
        # create test data with missing values
        test_data = self.student_info.copy()
        test_data.loc[2, 'imd_band'] = np.nan
        test_data.loc[3, 'disability'] = np.nan
        
        # clean data
        cleaned_data = clean_demographic_data(test_data)
        
        # check if missing values were handled correctly
        self.assertEqual(cleaned_data.loc[2, 'imd_band'], 'unknown')
        self.assertEqual(cleaned_data.loc[3, 'disability'], 'N')
        
        # check if string columns were standardized
        self.assertTrue(all(cleaned_data['gender'].str.islower()))
        self.assertTrue(all(cleaned_data['region'].str.islower()))
        self.assertTrue(all(cleaned_data['highest_education'].str.islower()))
        self.assertTrue(all(cleaned_data['imd_band'].str.islower()))
        self.assertTrue(all(cleaned_data['age_band'].str.islower()))
    
    def test_clean_vle_data(self):
        """Tests cleaning VLE interaction data."""
        # create test data with invalid values
        test_interactions = self.vle_interactions.copy()
        test_interactions.loc[3, 'sum_click'] = 0  # invalid click count
        
        # create incomplete materials data
        test_materials = self.vle_materials.copy()
        test_materials.loc[1, 'activity_type'] = np.nan  # missing activity type
        
        # clean data
        cleaned_data = clean_vle_data(test_interactions, test_materials)
        
        # check if invalid click counts were removed
        self.assertEqual(len(cleaned_data), 6)  # one row removed
        
        # check if all rows have activity types
        self.assertFalse(cleaned_data['activity_type'].isna().any())
        
        # check if merged correctly
        self.assertTrue('activity_type' in cleaned_data.columns)
        self.assertTrue('week_from' in cleaned_data.columns)
        self.assertTrue('week_to' in cleaned_data.columns)
    
    def test_clean_assessment_data(self):
        """Tests cleaning assessment data."""
        # create test data with invalid scores
        test_assessments = self.student_assessments.copy()
        test_assessments.loc[1, 'score'] = 101  # invalid score
        test_assessments.loc[3, 'score'] = -5   # invalid score
        
        # clean data
        cleaned_data = clean_assessment_data(self.assessments, test_assessments)
        
        # check if invalid scores were removed
        self.assertEqual(len(cleaned_data), 3)  # two rows removed
        
        # check if merged correctly
        self.assertTrue('assessment_type' in cleaned_data.columns)
        self.assertTrue('weight' in cleaned_data.columns)
        self.assertTrue('date' in cleaned_data.columns)
    
    def test_validate_data_consistency(self):
        """Tests data consistency validation."""
        # create consistent datasets
        datasets = {
            'student_info': self.student_info,
            'vle_interactions': self.vle_interactions,
            'student_assessments': self.student_assessments,
            'courses': self.courses
        }
        
        # validate data consistency
        result = validate_data_consistency(datasets)
        self.assertTrue(result)
        
        # create inconsistent datasets (student ID not in student_info)
        invalid_datasets = datasets.copy()
        invalid_vle = self.vle_interactions.copy()
        invalid_vle.loc[0, 'id_student'] = 999  # non-existent student ID
        invalid_datasets['vle_interactions'] = invalid_vle
        
        # validation should still pass but print warning
        with patch('builtins.print') as mock_print:
            result = validate_data_consistency(invalid_datasets)
            self.assertTrue(result)  # should still return True
            mock_print.assert_called_with("Warning: Some VLE or assessment records have unknown student IDs")


class TestFeatureEngineering(unittest.TestCase):
    """Tests for feature engineering functions."""
    
    def setUp(self):
        """Create sample data for testing."""
        # create sample data for feature engineering
        self.demographics = pd.DataFrame({
            'id_student': [1, 2, 3, 4, 5],
            'code_module': ['AAA', 'BBB', 'AAA', 'CCC', 'BBB'],
            'code_presentation': ['2020J', '2020J', '2020B', '2020B', '2020B'],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'region': ['East Anglian', 'London', 'Scotland', 'Wales', 'North'],
            'highest_education': ['A Level', 'HE Qualification', 'A Level', 'No Formal', 'HE Qualification'],
            'imd_band': ['0-10%', '20-30%', '30-40%', '50-60%', '90-100%'],
            'age_band': ['0-35', '0-35', '35-55', '35-55', '55<='],
            'num_of_prev_attempts': [0, 1, 0, 2, 1],
            'studied_credits': [60, 60, 30, 60, 30],
            'disability': ['N', 'Y', 'N', 'N', 'Y'],
            'final_result': ['Pass', 'Fail', 'Distinction', 'Withdrawn', 'Pass']
        })
        
        self.vle_data = pd.DataFrame({
            'id_student': [1, 1, 2, 2, 3, 4, 5],
            'code_module': ['AAA', 'AAA', 'BBB', 'BBB', 'AAA', 'CCC', 'BBB'],
            'code_presentation': ['2020J', '2020J', '2020J', '2020J', '2020B', '2020B', '2020B'],
            'id_site': [1, 2, 1, 3, 2, 3, 1],
            'date': [10, 15, 5, 20, 30, 25, 40],
            'sum_click': [5, 10, 8, 3, 12, 7, 9],
            'activity_type': ['resource', 'quiz', 'resource', 'forum', 'quiz', 'forum', 'resource']
        })
        
        self.assessment_data = pd.DataFrame({
            'id_student': [1, 2, 2, 3, 4],
            'code_module': ['AAA', 'BBB', 'BBB', 'AAA', 'CCC'],
            'code_presentation': ['2020J', '2020J', '2020J', '2020B', '2020B'],
            'id_assessment': [1, 1, 2, 2, 3],
            'assessment_type': ['TMA', 'TMA', 'CMA', 'CMA', 'Exam'],
            'date': [30, 30, 60, 60, 90],
            'date_submitted': [25, 28, 55, 58, 85],
            'weight': [25, 25, 25, 25, 50],
            'score': [85, 65, 75, 90, 45]
        })
    
    def test_create_demographic_features(self):
        """Tests creating demographic features."""
        # create demographic features
        features = create_demographic_features(self.demographics)
        
        # check if encoded categorical variables were created
        for col in ['gender', 'region', 'highest_education', 'imd_band', 'age_band']:
            encoded_col = f"{col}_encoded"
            self.assertTrue(encoded_col in features.columns)
            
            # check one-hot encoding
            one_hot_cols = [c for c in features.columns if c.startswith(f"{col}_")]
            self.assertTrue(len(one_hot_cols) > 0)
        
        # check if educational background features were created
        self.assertTrue('is_first_attempt' in features.columns)
        self.assertTrue('credit_density' in features.columns)
        
        # check if key columns were preserved
        for col in ['id_student', 'code_module', 'code_presentation']:
            self.assertTrue(col in features.columns)
    
    def test_create_temporal_features(self):
        """Tests creating temporal features with multiple window sizes."""
        # create temporal features
        window_sizes = [7, 14]
        temporal_features = create_temporal_features(self.vle_data, window_sizes)
        
        # check if features were created for each window size
        self.assertEqual(len(temporal_features), len(window_sizes))
        
        for window_size in window_sizes:
            window_key = f"window_{window_size}"
            self.assertTrue(window_key in temporal_features)
            
            # check if numeric metrics were calculated
            features = temporal_features[window_key]
            self.assertTrue('sum_click_sum' in features.columns)
            self.assertTrue('sum_click_mean' in features.columns)
            self.assertTrue('id_site_nunique' in features.columns)
            
            # check if activity type counts were created
            activity_cols = [c for c in features.columns if c.startswith('activity_')]
            self.assertTrue(len(activity_cols) > 0)
    
    def test_create_assessment_features(self):
        """Tests creating assessment features."""
        # create assessment features
        features = create_assessment_features(self.assessment_data)
        
        # check if aggregate metrics were calculated
        self.assertTrue('score_mean' in features.columns)
        self.assertTrue('score_std' in features.columns)
        self.assertTrue('score_min' in features.columns)
        self.assertTrue('score_max' in features.columns)
        self.assertTrue('score_count' in features.columns)
        
        # check if submission delay metrics were calculated
        self.assertTrue('submission_delay_mean' in features.columns)
        self.assertTrue('submission_delay_std' in features.columns)
        
        # check if final metrics were calculated
        self.assertTrue('weighted_score' in features.columns)
        self.assertTrue('submission_consistency' in features.columns)
        
        # check if key columns were preserved
        for col in ['id_student', 'code_module', 'code_presentation']:
            self.assertTrue(col in features.columns)
    
    def test_create_sequential_features(self):
        """Tests creating sequential features."""
        # create sequential features
        features = create_sequential_features(self.vle_data)
        
        # check if time-based features were created
        self.assertTrue('time_since_last' in features.columns)
        self.assertTrue('cumulative_clicks' in features.columns)
        
        # check if activity transition feature was created
        self.assertTrue('prev_activity' in features.columns)
        
        # check if data was sorted correctly
        for student_id in features['id_student'].unique():
            student_data = features[features['id_student'] == student_id]
            
            # check if dates are in ascending order
            self.assertTrue(student_data['date'].is_monotonic_increasing)
            
            # check if cumulative clicks are increasing
            self.assertTrue(student_data['cumulative_clicks'].is_monotonic_increasing)
    
    def test_prepare_target_variable(self):
        """Tests preparing binary target variable."""
        # prepare target variable
        target = prepare_target_variable(self.demographics)
        
        # check if binary target was created
        self.assertEqual(set(target.unique()), {0, 1})
        
        # check specific mappings
        self.assertEqual(target[self.demographics['final_result'] == 'Pass'].iloc[0], 0)
        self.assertEqual(target[self.demographics['final_result'] == 'Distinction'].iloc[0], 0)
        self.assertEqual(target[self.demographics['final_result'] == 'Fail'].iloc[0], 1)
        self.assertEqual(target[self.demographics['final_result'] == 'Withdrawn'].iloc[0], 1)
        
        # check number of at-risk students
        self.assertEqual(sum(target), 2)  # 1 Fail + 1 Withdrawn


if __name__ == '__main__':
    unittest.main()