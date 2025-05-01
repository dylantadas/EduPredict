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
    clean_assessment_data
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

        # Create sample demographic data
        self.demographics = pd.DataFrame({
            'id_student': [1, 2, 3, 4, 5],
            'code_module': ['AAA', 'BBB', 'AAA', 'CCC', 'BBB'],
            'code_presentation': ['2020J', '2020J', '2020B', '2020B', '2020B'],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'region': ['East', 'London', 'Scotland', 'Wales', 'North'],
            'highest_education': ['A Level', 'HE', 'A Level', 'None', 'HE'],
            'imd_band': ['0-10%', '20-30%', '30-40%', '50-60%', '90-100%'],
            'age_band': ['0-35', '0-35', '35-55', '35-55', '55<='],
            'disability': ['N', 'Y', 'N', 'N', 'Y'],
            'final_result': ['Pass', 'Fail', 'Distinction', 'Withdrawn', 'Pass']
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

        # Create sample assessment data
        self.assessment_data = pd.DataFrame({
            'id_student': [1, 2, 2, 3, 4],
            'code_module': ['AAA', 'BBB', 'BBB', 'AAA', 'CCC'],
            'code_presentation': ['2020J', '2020J', '2020J', '2020B', '2020B'],
            'assessment_type': ['TMA', 'TMA', 'CMA', 'CMA', 'Exam'],
            'date': [30, 30, 60, 60, 90],
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
        self.assessment_data.to_csv(self.data_path / "studentAssessment.csv", index=False)

        # Test data loading
        datasets = load_raw_datasets(str(self.data_path))
        self.assertTrue('student_info' in datasets)
        self.assertTrue('vle_interactions' in datasets)
        self.assertTrue('student_assessments' in datasets)

        # Test data validation
        validation_result = validate_data_consistency(datasets)
        self.assertTrue(validation_result)

        # Test data quality monitoring
        quality_report = detect_data_quality_issues(
            datasets['student_info'],
            protected_cols=['gender', 'age_band', 'disability']
        )
        self.assertTrue('quality_metrics' in quality_report)
        self.assertTrue('recommendations' in quality_report)

        # Test data cleaning
        clean_data = {
            'demographics': clean_demographic_data(datasets['student_info']),
            'vle': clean_vle_data(self.vle_data, pd.DataFrame()),
            'assessments': clean_assessment_data(
                pd.DataFrame(), datasets['student_assessments']
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
        importance = analyze_feature_importance(
            self.demographics,
            self.demographics['final_result'],
            feature_type='demographic',
            save_path=str(self.output_path / 'importance.png')
        )
        self.assertTrue(isinstance(importance[0], pd.DataFrame))

        # Test correlation analysis
        correlations = analyze_feature_correlations(
            self.demographics,
            feature_type='demographic'
        )
        self.assertTrue(isinstance(correlations, pd.DataFrame))

if __name__ == '__main__':
    unittest.main()