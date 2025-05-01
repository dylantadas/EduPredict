import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_processing.data_loader import load_raw_datasets
from data_processing.data_cleaner import (
    clean_demographic_data,
    clean_vle_data,
    clean_assessment_data,
    clean_registration_data
)
from data_processing.data_splitter import create_stratified_splits
from data_processing.data_monitor import detect_data_quality_issues

from feature_engineering.feature_selector import (
    analyze_feature_importance,
    analyze_feature_correlations,
    analyze_demographic_impact
)

from utils.validation_utils import (
    validate_directories,
    validate_data_consistency,
    validate_feature_engineering_inputs
)

class TestPipeline(unittest.TestCase):
    """Tests for the complete data processing and feature engineering pipeline."""

    def setUp(self):
        """Create sample data and temporary directories."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.temp_dir.name)
        self.output_path = self.data_path / "output"
        self.output_path.mkdir()

        # Create sample demographic data with enough samples for stratification
        self.demographics = pd.DataFrame({
            'id_student': range(1, 21),  # 20 students
            'code_module': ['AAA']*5 + ['BBB']*10 + ['CCC']*5,
            'code_presentation': ['2020J']*10 + ['2020B']*10,
            'gender': ['M', 'F']*10,  # Even gender distribution
            'region': ['East', 'London', 'Scotland', 'Wales', 'North']*4,
            'highest_education': ['A Level', 'HE', 'A Level', 'None', 'HE']*4,
            'imd_band': ['0-10%', '20-30%', '30-40%', '50-60%', '90-100%']*4,
            'age_band': ['0-35']*7 + ['35-55']*7 + ['55<=']*6,  # More balanced age distribution
            'disability': ['N', 'Y']*10,
            'final_result': ['Pass', 'Fail', 'Distinction', 'Withdrawn']*5
        })

        # Create sample VLE data
        self.vle_data = pd.DataFrame({
            'id_student': [1, 1, 2, 2, 3],
            'code_module': ['AAA', 'AAA', 'BBB', 'BBB', 'AAA'],
            'code_presentation': ['2020J', '2020J', '2020J', '2020J', '2020B'],
            'id_site': [1, 2, 1, 3, 2],
            'date': [10, 15, 5, 20, 30],
            'sum_click': [5, 10, 8, 3, 12],
            'activity_type': ['resource', 'quiz', 'resource', 'forum', 'quiz']
        })

        # Create sample VLE materials data
        self.vle_materials = pd.DataFrame({
            'id_site': [1, 2, 3],
            'code_module': ['AAA', 'BBB', 'CCC'],
            'code_presentation': ['2020J', '2020J', '2020B'],
            'activity_type': ['resource', 'quiz', 'forum']
        })

        # Create sample assessments data
        self.assessments = pd.DataFrame({
            'id_assessment': [1, 2, 3, 4],
            'code_module': ['AAA', 'BBB', 'AAA', 'CCC'],
            'code_presentation': ['2020J', '2020J', '2020B', '2020B'],
            'assessment_type': ['TMA', 'CMA', 'TMA', 'Exam'],
            'date': [30, 60, 90, 180],
            'weight': [20, 30, 20, 30]
        })

        # Create sample assessment submission data
        self.assessment_data = pd.DataFrame({
            'id_student': [1, 2, 2, 3, 4],
            'id_assessment': [1, 1, 2, 3, 4],
            'code_module': ['AAA', 'BBB', 'BBB', 'AAA', 'CCC'],
            'code_presentation': ['2020J', '2020J', '2020J', '2020B', '2020B'],
            'date_submitted': [25, 28, 55, 58, 85],
            'score': [85, 65, 75, 90, 45]
        })

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_data_processing_workflow(self):
        """Tests the complete data processing workflow."""
        # Save sample data
        self.demographics.to_csv(self.data_path / "studentInfo.csv", index=False)
        self.vle_data.to_csv(self.data_path / "studentVle.csv", index=False)
        self.vle_materials.to_csv(self.data_path / "vle.csv", index=False)
        self.assessment_data.to_csv(self.data_path / "studentAssessment.csv", index=False)
        self.assessments.to_csv(self.data_path / "assessments.csv", index=False)

        # Test data loading
        datasets = load_raw_datasets(str(self.data_path))
        self.assertTrue('student_info' in datasets)
        self.assertTrue('vle_interactions' in datasets)
        self.assertTrue('student_assessments' in datasets)

        # Test data validation
        validation_result = validate_data_consistency(datasets)
        self.assertTrue(validation_result)

        # Test data quality monitoring
        quality_report = detect_data_quality_issues(datasets)
        self.assertTrue('quality_metrics' in quality_report)
        self.assertTrue('recommendations' in quality_report)

        # Test data cleaning
        clean_data = {
            'demographics': clean_demographic_data(datasets['student_info']),
            'vle': clean_vle_data(self.vle_data, pd.DataFrame()),
            'assessments': clean_assessment_data(
                datasets['assessments'],  # Use loaded assessment data
                datasets['student_assessments']
            )
        }
        self.assertTrue(all(len(df) > 0 for df in clean_data.values()))

        # Test data splitting
        splits = create_stratified_splits(
            clean_data['demographics'],
            target_col='final_result',
            strat_cols=['gender', 'age_band'],
            test_size=0.2,
            validation_size=0.2
        )
        self.assertTrue(all(k in splits for k in ['train', 'validation', 'test']))

    def test_feature_engineering_workflow(self):
        """Tests the complete feature engineering workflow."""
        # Test input validation
        required_columns = {
            'demographics': ['id_student', 'gender', 'age_band', 'final_result'],
            'vle': ['id_student', 'date', 'sum_click', 'activity_type'],
            'assessments': ['id_student', 'score', 'date_submitted']
        }
        
        data = {
            'demographics': self.demographics,
            'vle': self.vle_data,
            'assessments': self.assessment_data
        }
        
        valid_inputs = validate_feature_engineering_inputs(data, required_columns)
        self.assertTrue(valid_inputs)

        # Test demographic feature analysis
        demographic_impact = analyze_demographic_impact(
            self.demographics,
            protected_attributes=['gender', 'age_band', 'disability']
        )
        self.assertTrue(all(attr in demographic_impact for attr in ['gender', 'age_band', 'disability']))

        # Test feature importance analysis
        importance_df, _ = analyze_feature_importance(
            self.demographics,
            self.demographics['final_result']
        )
        self.assertTrue(isinstance(importance_df, pd.DataFrame))
        self.assertTrue('feature' in importance_df.columns)
        self.assertTrue('importance' in importance_df.columns)

        # Test correlation analysis
        correlations = analyze_feature_correlations(self.demographics)
        self.assertTrue(isinstance(correlations, pd.DataFrame))

    def test_registration_data_cleaning(self):
        """Tests the cleaning of student registration data."""
        # Create test data
        test_data = pd.DataFrame({
            'code_module': ['AAA', 'BBB', 'CCC', 'AAA'],
            'code_presentation': ['2013J', '2013J', '2014B', '2014B'],
            'id_student': [1, 2, 3, 4],
            'date_registration': [-30, -5, 10, 35],  # Mix of early, on-time, and late
            'date_unregistration': [None, 45, None, 190]  # Mix of completed and dropped
        })
        
        # Clean the data
        cleaned_data = clean_registration_data(test_data)
        
        # Validate cleaning results
        assert 'completed_module' in cleaned_data.columns
        assert 'registration_type' in cleaned_data.columns
        
        # Check registration type categorization
        assert cleaned_data.loc[0, 'registration_type'] == 'early'
        assert cleaned_data.loc[1, 'registration_type'] == 'on_time'
        assert cleaned_data.loc[2, 'registration_type'] == 'late'
        assert cleaned_data.loc[3, 'registration_type'] == 'late'
        
        # Check completion status
        assert cleaned_data.loc[0, 'completed_module']  # No unregistration date = completed
        assert not cleaned_data.loc[1, 'completed_module']  # Has unregistration = dropped
        assert cleaned_data.loc[2, 'completed_module']  # No unregistration date = completed
        assert not cleaned_data.loc[3, 'completed_module']  # Has unregistration = dropped
        
        # Validate module codes
        assert all(code in ['AAA', 'BBB', 'CCC'] for code in cleaned_data['code_module'])
        assert all(code in ['2013J', '2013B', '2014J', '2014B'] for code in cleaned_data['code_presentation'])

if __name__ == '__main__':
    unittest.main()