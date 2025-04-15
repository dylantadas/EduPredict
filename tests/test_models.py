import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import pickle
import tensorflow as tf
from unittest.mock import patch, MagicMock

# add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# import model functions to test
from model_training.random_forest_model import RandomForestModel, find_optimal_threshold
from model_training.gru_model import GRUModel, SequencePreprocessor
from ensemble.ensemble import EnsembleModel
from evaluation.performance_metrics import calculate_model_metrics
from evaluation.fairness_analysis import (
    calculate_group_metrics, 
    calculate_fairness_metrics,
    evaluate_model_fairness
)


class TestRandomForestModel(unittest.TestCase):
    """Tests for Random Forest model."""
    
    def setUp(self):
        """Create sample data for model testing."""
        # create sample features and target
        np.random.seed(0)
        n_samples = 100
        n_features = 10
        
        # create binary classification problem
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, size=n_samples)
        
        # convert to dataframe
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.X_train = pd.DataFrame(X[:80], columns=feature_names)
        self.y_train = pd.Series(y[:80])
        self.X_test = pd.DataFrame(X[80:], columns=feature_names)
        self.y_test = pd.Series(y[80:])
        
        # create temporary directory for model saving/loading
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'rf_model.pkl')
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_model_init(self):
        """Tests model initialization."""
        # initialize model with default parameters
        model = RandomForestModel()
        
        # check if model attributes were set correctly
        self.assertFalse(model.trained)
        self.assertIsNone(model.feature_names)
        
        # initialize model with custom parameters
        model = RandomForestModel(
            n_estimators=50,
            max_depth=5,
            min_samples_split=5,
            class_weight=None
        )
        
        # check if custom parameters were set
        self.assertEqual(model.model.n_estimators, 50)
        self.assertEqual(model.model.max_depth, 5)
        self.assertEqual(model.model.min_samples_split, 5)
        self.assertIsNone(model.model.class_weight)
    
    def test_model_fit(self):
        """Tests model fitting."""
        # initialize and fit model
        model = RandomForestModel(n_estimators=10, random_state=0)
        model.fit(self.X_train, self.y_train)
        
        # check if model was trained
        self.assertTrue(model.trained)
        self.assertEqual(model.feature_names, self.X_train.columns.tolist())
    
    def test_model_predict(self):
        """Tests model prediction."""
        # initialize and fit model
        model = RandomForestModel(n_estimators=10, random_state=0)
        model.fit(self.X_train, self.y_train)
        
        # test predict_proba
        probs = model.predict_proba(self.X_test)
        self.assertEqual(len(probs), len(self.X_test))
        self.assertTrue(all(0 <= p <= 1 for p in probs))
        
        # test predict with default threshold
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(all(p in [0, 1] for p in preds))
        
        # test predict with custom threshold
        preds_custom = model.predict(self.X_test, threshold=0.7)
        self.assertEqual(len(preds_custom), len(self.X_test))
        self.assertTrue(all(p in [0, 1] for p in preds_custom))
    
    def test_model_evaluate(self):
        """Tests model evaluation."""
        # initialize and fit model
        model = RandomForestModel(n_estimators=10, random_state=0)
        model.fit(self.X_train, self.y_train)
        
        # evaluate model
        metrics = model.evaluate(self.X_test, self.y_test)
        
        # check if metrics were calculated
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('f1_score' in metrics)
        self.assertTrue('auc_roc' in metrics)
        self.assertTrue('confusion_matrix' in metrics)
        self.assertTrue('threshold' in metrics)
        
        # check metric values are in expected range
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['f1_score'] <= 1)
        self.assertTrue(0 <= metrics['auc_roc'] <= 1)
    
    def test_feature_importance(self):
        """Tests feature importance extraction."""
        # initialize and fit model
        model = RandomForestModel(n_estimators=10, random_state=0)
        model.fit(self.X_train, self.y_train)
        
        # get feature importance without plotting
        importance_df = model.get_feature_importance(plot=False)
        
        # check if feature importance dataframe was created
        self.assertEqual(len(importance_df), len(self.X_train.columns))
        self.assertTrue('Feature' in importance_df.columns)
        self.assertTrue('Importance' in importance_df.columns)
        
        # check if importance values sum to 1
        self.assertAlmostEqual(importance_df['Importance'].sum(), 1.0, places=5)
    
    def test_model_save_load(self):
        """Tests model saving and loading."""
        # initialize and fit model
        model = RandomForestModel(n_estimators=10, random_state=0)
        model.fit(self.X_train, self.y_train)
        
        # save model
        model.save_model(self.model_path)
        
        # check if model file was created
        self.assertTrue(os.path.exists(self.model_path))
        
        # load model
        loaded_model = RandomForestModel.load_model(self.model_path)
        
        # check if loaded model has same attributes
        self.assertTrue(loaded_model.trained)
        self.assertEqual(loaded_model.feature_names, model.feature_names)
        
        # check if loaded model makes same predictions
        original_preds = model.predict(self.X_test)
        loaded_preds = loaded_model.predict(self.X_test)
        self.assertTrue(all(original_preds == loaded_preds))
    
    def test_find_optimal_threshold(self):
        """Tests finding optimal threshold for binary classification."""
        # initialize and fit model
        model = RandomForestModel(n_estimators=10, random_state=0)
        model.fit(self.X_train, self.y_train)
        
        # find optimal threshold
        threshold = find_optimal_threshold(model, self.X_test, self.y_test, metric='f1')
        
        # check if threshold is in valid range
        self.assertTrue(0 <= threshold <= 1)


class TestGRUModel(unittest.TestCase):
    """Tests for GRU model."""
    
    def setUp(self):
        """Create sample data for sequence model testing."""
        # create sample sequential data
        np.random.seed(0)
        n_students = 20
        max_seq_length = 10
        categorical_dim = 3
        numerical_dim = 2
        
        # create categorical sequences (one-hot encoded activities)
        categorical_data = np.random.randint(0, 3, size=(n_students, max_seq_length, categorical_dim))
        
        # create numerical sequences
        numerical_data = np.random.randn(n_students, max_seq_length, numerical_dim)
        
        # create binary targets
        y = np.random.randint(0, 2, size=n_students)
        
        # split train/test
        self.X_train = {
            'categorical': categorical_data[:16],
            'numerical': numerical_data[:16],
            'students': np.arange(16),
            'student_index_map': {i: i for i in range(16)}
        }
        self.y_train = y[:16]
        
        self.X_test = {
            'categorical': categorical_data[16:],
            'numerical': numerical_data[16:],
            'students': np.arange(16, 20),
            'student_index_map': {i+16: i for i in range(4)}
        }
        self.y_test = y[16:]
        
        # create sample raw data for preprocessor
        self.raw_data = pd.DataFrame({
            'id_student': np.repeat(np.arange(5), 4),
            'date': np.tile(np.arange(4), 5),
            'activity_type': np.random.choice(['resource', 'quiz', 'forum'], 20),
            'sum_click': np.random.randint(1, 10, 20),
            'time_since_last': np.random.randint(0, 5, 20)
        })
        
        # create temporary directory for model saving/loading
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'gru_model.keras')
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    @unittest.skipIf(not tf.test.is_gpu_available(cuda_only=True), "Skip GRU tests if GPU not available")
    def test_model_init_and_build(self):
        """Tests model initialization and architecture building."""
        # initialize model
        model = GRUModel(
            gru_units=32,
            dense_units=[16],
            dropout_rate=0.2,
            learning_rate=0.001,
            max_seq_length=10
        )
        
        # check if model attributes were set correctly
        self.assertEqual(model.gru_units, 32)
        self.assertEqual(model.dense_units, [16])
        self.assertEqual(model.dropout_rate, 0.2)
        self.assertEqual(model.learning_rate, 0.001)
        self.assertEqual(model.max_seq_length, 10)
        self.assertFalse(model.trained)
        
        # build model architecture
        categorical_dim = self.X_train['categorical'].shape[2]
        numerical_dim = self.X_train['numerical'].shape[2]
        
        model.build_model(categorical_dim, numerical_dim)
        
        # check if model was built
        self.assertIsNotNone(model.model)
        
        # check if model has correct input shape
        self.assertEqual(len(model.model.inputs), 2)  # categorical and numerical inputs
    
    @unittest.skipIf(not tf.test.is_gpu_available(cuda_only=True), "Skip GRU tests if GPU not available")
    def test_model_fit_and_predict(self):
        """Tests model fitting and prediction."""
        # initialize and build model with small size for quick testing
        model = GRUModel(
            gru_units=8,
            dense_units=[4],
            dropout_rate=0.2,
            learning_rate=0.001,
            max_seq_length=10
        )
        
        categorical_dim = self.X_train['categorical'].shape[2]
        numerical_dim = self.X_train['numerical'].shape[2]
        model.build_model(categorical_dim, numerical_dim)
        
        # fit model for just 2 epochs for testing
        model.fit(
            self.X_train,
            self.y_train,
            epochs=2,
            batch_size=4
        )
        
        # check if model was trained
        self.assertTrue(model.trained)
        self.assertIsNotNone(model.history)
        
        # test predict_proba
        probs = model.predict_proba(self.X_test)
        self.assertEqual(len(probs), len(self.X_test['students']))
        self.assertTrue(all(0 <= p <= 1 for p in probs))
        
        # test predict
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test['students']))
        self.assertTrue(all(p in [0, 1] for p in preds))
    
    def test_sequence_preprocessor(self):
        """Tests sequence preprocessor functionality."""
        # initialize preprocessor
        preprocessor = SequencePreprocessor(max_seq_length=5)
        
        # fit preprocessor
        categorical_cols = ['activity_type']
        numerical_cols = ['sum_click', 'time_since_last']
        preprocessor.fit(self.raw_data, categorical_cols, numerical_cols)
        
        # check if preprocessor was fitted
        self.assertTrue(preprocessor.fitted)
        self.assertIsNotNone(preprocessor.activity_encoder)
        self.assertEqual(len(preprocessor.feature_scalers), len(numerical_cols))
        
        # transform data
        sequences = preprocessor.transform_sequences(
            self.raw_data,
            student_col='id_student',
            time_col='date',
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols
        )
        
        # check if sequences were created
        self.assertTrue('categorical' in sequences)
        self.assertTrue('numerical' in sequences)
        self.assertTrue('students' in sequences)
        self.assertTrue('student_index_map' in sequences)
        
        # check sequence dimensions
        n_students = len(self.raw_data['id_student'].unique())
        self.assertEqual(len(sequences['students']), n_students)
        self.assertEqual(sequences['categorical'].shape, (n_students, 5, 1))  # 5 is max_seq_length
        self.assertEqual(sequences['numerical'].shape, (n_students, 5, 2))  # 2 numerical features
        
        # check if student index mapping was created
        self.assertEqual(len(sequences['student_index_map']), n_students)
    
    @unittest.skipIf(not tf.test.is_gpu_available(cuda_only=True), "Skip GRU tests if GPU not available")
    def test_model_save_load(self):
        """Tests model saving and loading with preprocessor."""
        # skip test if TensorFlow can't run
        try:
            # initialize small model
            model = GRUModel(
                gru_units=8,
                dense_units=[4],
                dropout_rate=0.2,
                learning_rate=0.001,
                max_seq_length=10,
                categorical_dim=3,
                numerical_dim=2
            )
            
            # build and fit model for just 1 epoch
            model.build_model()
            
            # prepare inputs
            inputs = {
                'categorical_input': self.X_train['categorical'],
                'numerical_input': self.X_train['numerical']
            }
            
            # simulate training
            model.model.fit(inputs, self.y_train, epochs=1, verbose=0)
            model.trained = True  # mark as trained
            
            # initialize preprocessor
            preprocessor = SequencePreprocessor(max_seq_length=5)
            preprocessor.fit(self.raw_data, ['activity_type'], ['sum_click', 'time_since_last'])
            
            # save model and preprocessor
            model.save_model(self.model_path, preprocessor)
            
            # check if files were created
            self.assertTrue(os.path.exists(self.model_path))
            self.assertTrue(os.path.exists(self.model_path.replace('.keras', '_preprocessor.pkl')))
            self.assertTrue(os.path.exists(self.model_path.replace('.keras', '_config.pkl')))
            
            # load model and preprocessor
            loaded_model, loaded_preprocessor = GRUModel.load_model(self.model_path)
            
            # check if loaded model has same attributes
            self.assertTrue(loaded_model.trained)
            self.assertEqual(loaded_model.gru_units, model.gru_units)
            self.assertEqual(loaded_model.dense_units, model.dense_units)
            self.assertEqual(loaded_model.dropout_rate, model.dropout_rate)
            
            # check if loaded preprocessor has same attributes
            self.assertTrue(loaded_preprocessor.fitted)
            self.assertEqual(loaded_preprocessor.max_seq_length, preprocessor.max_seq_length)
            
        except tf.errors.InvalidArgumentError:
            # If TensorFlow fails to run on this system, skip the test
            self.skipTest("TensorFlow failed to run this test")


class TestEnsembleModel(unittest.TestCase):
    """Tests for Ensemble model."""
    
    def setUp(self):
        """Create sample models and data for ensemble testing."""
        # create mock models
        self.static_model = MagicMock()
        self.sequential_model = MagicMock()
        
        # setup mock predict_proba methods
        np.random.seed(0)
        n_samples = 20
        
        self.static_probs = np.random.rand(n_samples)
        self.sequential_probs = np.random.rand(n_samples)
        
        self.static_model.predict_proba.return_value = self.static_probs
        self.sequential_model.predict_proba.return_value = self.sequential_probs
        
        # create sample features
        self.static_features = pd.DataFrame({
            'id_student': list(range(n_samples)),
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        
        self.sequential_features = {
            'categorical': np.random.randn(n_samples, 10, 3),
            'numerical': np.random.randn(n_samples, 10, 2)
        }
        
        # create student ID mapping
        self.student_id_map = {i: i for i in range(n_samples)}
        
        # create binary targets
        self.y_true = np.random.randint(0, 2, size=n_samples)
        
        # create temporary directory for model saving/loading
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'ensemble_model.pkl')
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_ensemble_init(self):
        """Tests ensemble initialization."""
        # initialize ensemble with default weights
        ensemble = EnsembleModel()
        
        # check if weights sum to 1
        self.assertAlmostEqual(ensemble.static_weight + ensemble.sequential_weight, 1.0)
        self.assertEqual(ensemble.threshold, 0.5)
        self.assertFalse(ensemble.optimized)
        
        # initialize with custom weights
        ensemble = EnsembleModel(static_weight=0.7, sequential_weight=0.3, threshold=0.6)
        self.assertEqual(ensemble.static_weight, 0.7)
        self.assertEqual(ensemble.sequential_weight, 0.3)
        self.assertEqual(ensemble.threshold, 0.6)
        
        # check validation of weights
        with self.assertRaises(ValueError):
            EnsembleModel(static_weight=0.7, sequential_weight=0.4)  # sum > 1
    
    def test_set_models(self):
        """Tests setting component models."""
        ensemble = EnsembleModel()
        
        # set models
        ensemble.set_models(self.static_model, self.sequential_model)
        
        # check if models were set
        self.assertEqual(ensemble.static_model, self.static_model)
        self.assertEqual(ensemble.sequential_model, self.sequential_model)
    
    def test_predict_without_models(self):
        """Tests prediction fails without models."""
        ensemble = EnsembleModel()
        
        # predict without setting models should raise error
        with self.assertRaises(ValueError):
            ensemble.predict_proba(self.static_features, self.sequential_features)
    
    def test_predict_proba(self):
        """Tests probability prediction."""
        # initialize and set models
        ensemble = EnsembleModel(static_weight=0.6, sequential_weight=0.4)
        ensemble.set_models(self.static_model, self.sequential_model)
        
        # predict probabilities
        probs = ensemble.predict_proba(
            self.static_features, 
            self.sequential_features,
            self.student_id_map
        )
        
        # check result shape
        self.assertEqual(len(probs), len(self.y_true))
        
        # check if weighted average was calculated correctly
        expected_probs = 0.6 * self.static_probs + 0.4 * self.sequential_probs
        np.testing.assert_allclose(probs, expected_probs)
        
        # check if models were called with correct arguments
        self.static_model.predict_proba.assert_called_once_with(self.static_features)
        self.sequential_model.predict_proba.assert_called_once_with(self.sequential_features)
    
    def test_predict(self):
        """Tests binary prediction."""
        # initialize and set models
        ensemble = EnsembleModel(static_weight=0.6, sequential_weight=0.4, threshold=0.7)
        ensemble.set_models(self.static_model, self.sequential_model)
        
        # predict binary classes
        preds = ensemble.predict(
            self.static_features, 
            self.sequential_features,
            self.student_id_map
        )
        
        # check result shape
        self.assertEqual(len(preds), len(self.y_true))
        
        # check if threshold was applied correctly
        expected_probs = 0.6 * self.static_probs + 0.4 * self.sequential_probs
        expected_preds = (expected_probs >= 0.7).astype(int)
        np.testing.assert_array_equal(preds, expected_preds)
    
    def test_optimize_weights(self):
        """Tests optimizing ensemble weights."""
        # initialize and set models
        ensemble = EnsembleModel()
        ensemble.set_models(self.static_model, self.sequential_model)
        
        # create a new instance of mock models for optimization
        # to avoid interference with the predict_proba mock returns
        static_model_opt = MagicMock()
        sequential_model_opt = MagicMock()
        
        # setup mock predict_proba for optimization
        static_model_opt.predict_proba.return_value = self.static_probs
        sequential_model_opt.predict_proba.return_value = self.sequential_probs
        
        ensemble.static_model = static_model_opt
        ensemble.sequential_model = sequential_model_opt
        
        # optimize weights
        ensemble.optimize_weights(
            self.static_features,
            self.sequential_features,
            self.y_true,
            self.student_id_map,
            metric='f1',
            weight_grid=5
        )
        
        # check if optimization was performed
        self.assertTrue(ensemble.optimized)
        
        # check if weights are in valid range
        self.assertTrue(0 <= ensemble.static_weight <= 1)
        self.assertTrue(0 <= ensemble.sequential_weight <= 1)
        self.assertAlmostEqual(ensemble.static_weight + ensemble.sequential_weight, 1.0)
        self.assertTrue(0 <= ensemble.threshold <= 1)
    
    def test_evaluate(self):
        """Tests ensemble evaluation."""
        # initialize and set models
        ensemble = EnsembleModel(static_weight=0.6, sequential_weight=0.4, threshold=0.5)
        ensemble.set_models(self.static_model, self.sequential_model)
        
        # evaluate ensemble
        metrics = ensemble.evaluate(
            self.static_features,
            self.sequential_features,
            self.y_true,
            self.student_id_map
        )
        
        # check if metrics were calculated
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('f1_score' in metrics)
        self.assertTrue('auc_roc' in metrics)
        self.assertTrue('confusion_matrix' in metrics)
        self.assertTrue('threshold' in metrics)
        self.assertTrue('static_weight' in metrics)
        self.assertTrue('sequential_weight' in metrics)
        
        # check metric values are in expected range
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['f1_score'] <= 1)
        self.assertTrue(0 <= metrics['auc_roc'] <= 1)
    
    def test_evaluate_demographic_fairness(self):
        """Tests demographic fairness evaluation."""
        # initialize and set models
        ensemble = EnsembleModel(static_weight=0.6, sequential_weight=0.4, threshold=0.5)
        ensemble.set_models(self.static_model, self.sequential_model)
        
        # add demographic column to static features
        self.static_features['gender'] = np.random.choice(['M', 'F'], size=len(self.static_features))
        
        # evaluate fairness
        fairness_df = ensemble.evaluate_demographic_fairness(
            self.static_features,
            self.sequential_features,
            self.y_true,
            'gender',
            self.student_id_map
        )
        
        # check if fairness metrics were calculated
        self.assertTrue('gender' in fairness_df.index)
        self.assertTrue('count' in fairness_df.columns)
        self.assertTrue('accuracy' in fairness_df.columns)
        self.assertTrue('f1' in fairness_df.columns)
        
        # check if disparate impact ratio was calculated
        self.assertTrue('disparate_impact_ratio' in fairness_df.columns)
    
    def test_save_load_model(self):
        """Tests saving and loading ensemble model."""
        # initialize and set models
        ensemble = EnsembleModel(static_weight=0.7, sequential_weight=0.3, threshold=0.6)
        ensemble.set_models(self.static_model, self.sequential_model)
        ensemble.optimized = True
        
        # save ensemble
        ensemble.save_model(self.model_path)
        
        # check if file was created
        self.assertTrue(os.path.exists(self.model_path))
        
        # load ensemble
        loaded_ensemble = EnsembleModel.load_model(self.model_path)
        
        # check if attributes were loaded correctly
        self.assertEqual(loaded_ensemble.static_weight, ensemble.static_weight)
        self.assertEqual(loaded_ensemble.sequential_weight, ensemble.sequential_weight)
        self.assertEqual(loaded_ensemble.threshold, ensemble.threshold)
        self.assertEqual(loaded_ensemble.optimized, ensemble.optimized)


class TestFairnessAnalysis(unittest.TestCase):
    """Tests for fairness analysis functions."""
    
    def setUp(self):
        """Create sample data for fairness analysis."""
        np.random.seed(0)
        n_samples = 100
        
        # create binary predictions and ground truth
        self.y_true = np.random.randint(0, 2, size=n_samples)
        self.y_pred = np.random.randint(0, 2, size=n_samples)
        self.y_prob = np.random.rand(n_samples)
        
        # create protected attributes
        self.gender = np.random.choice(['M', 'F'], size=n_samples)
        self.age_group = np.random.choice(['18-25', '26-35', '36+'], size=n_samples)
        
        self.protected_attributes = {
            'gender': self.gender,
            'age_group': self.age_group
        }
    
    def test_calculate_group_metrics(self):
        """Tests calculation of group-level performance metrics."""
        # calculate metrics for gender groups
        group_metrics = calculate_group_metrics(
            self.y_true,
            self.y_pred,
            self.y_prob,
            self.gender
        )
        
        # check if metrics were calculated for each group
        self.assertEqual(len(group_metrics), 2)  # M and F
        self.assertTrue('group' in group_metrics.columns)
        self.assertTrue('count' in group_metrics.columns)
        self.assertTrue('accuracy' in group_metrics.columns)
        self.assertTrue('precision' in group_metrics.columns)
        self.assertTrue('recall' in group_metrics.columns)
        self.assertTrue('f1' in group_metrics.columns)
        self.assertTrue('auc' in group_metrics.columns)
        self.assertTrue('true_positive_rate' in group_metrics.columns)
        self.assertTrue('false_positive_rate' in group_metrics.columns)
        
        # check if metrics are in valid range
        self.assertTrue(all(0 <= v <= 1 for v in group_metrics['accuracy']))
        self.assertTrue(all(0 <= v <= 1 for v in group_metrics['precision']))
        self.assertTrue(all(0 <= v <= 1 for v in group_metrics['recall']))
        self.assertTrue(all(0 <= v <= 1 for v in group_metrics['f1']))
    
    def test_calculate_fairness_metrics(self):
        """Tests calculation of fairness metrics."""
        # calculate group metrics first
        group_metrics = calculate_group_metrics(
            self.y_true,
            self.y_pred,
            self.y_prob,
            self.gender
        )
        
        # calculate fairness metrics
        fairness_metrics = calculate_fairness_metrics(group_metrics)
        
        # check if key fairness metrics were calculated
        self.assertTrue('demographic_parity_difference' in fairness_metrics)
        self.assertTrue('disparate_impact_ratio' in fairness_metrics)
        self.assertTrue('equal_opportunity_difference' in fairness_metrics)
        
        # check if metrics are in valid range
        self.assertTrue(0 <= fairness_metrics['demographic_parity_difference'] <= 1)
        self.assertTrue(0 <= fairness_metrics['disparate_impact_ratio'] <= 1)
        self.assertTrue(0 <= fairness_metrics['equal_opportunity_difference'] <= 1)
    
    def test_evaluate_model_fairness(self):
        """Tests comprehensive model fairness evaluation."""
        # evaluate fairness across multiple protected attributes
        fairness_results = evaluate_model_fairness(
            self.y_true,
            self.y_pred,
            self.y_prob,
            self.protected_attributes
        )
        
        # check if results were calculated for each attribute
        self.assertTrue('gender' in fairness_results)
        self.assertTrue('age_group' in fairness_results)
        
        # check if each attribute has the expected components
        for attr, results in fairness_results.items():
            self.assertTrue('group_metrics' in results)
            self.assertTrue('fairness_metrics' in results)
            self.assertTrue('threshold_results' in results)
            self.assertTrue('passes_all_thresholds' in results)
            
            # check if fairness metrics were calculated
            fairness_metrics = results['fairness_metrics']
            self.assertTrue('demographic_parity_difference' in fairness_metrics)
            self.assertTrue('disparate_impact_ratio' in fairness_metrics)
            self.assertTrue('equal_opportunity_difference' in fairness_metrics)


if __name__ == '__main__':
    unittest.main()