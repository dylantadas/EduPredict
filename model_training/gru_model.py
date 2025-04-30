import numpy as np # type: ignore
import pandas as pd # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.layers import GRU, Dense, Input, Embedding, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder # type: ignore
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt # type: ignore
import pickle
import os

class SequencePreprocessor:
    """Preprocesses sequential data for gru model."""
    
    def __init__(self, max_seq_length: int = 100):
        """Initializes with sequence parameters."""
        self.max_seq_length = max_seq_length
        self.feature_scalers = {}
        self.activity_encoder = None
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, categorical_cols: List[str], numerical_cols: List[str]):
        """Fits preprocessors on training data."""
        
        # fit label encoders for categorical features
        if categorical_cols:
            self.activity_encoder = LabelEncoder()
            # Use a list comprehension to collect all unique values across DataFrame
            all_activities = df[categorical_cols[0]].dropna().unique().tolist()
            # Add an 'unknown' category to handle missing values
            all_activities.append('unknown')
            self.activity_encoder.fit(all_activities)
        
        # fit scalers for numerical features
        for col in numerical_cols:
            self.feature_scalers[col] = StandardScaler()
            # Reshape to handle single feature scaling
            self.feature_scalers[col].fit(df[col].values.reshape(-1, 1))
        
        self.fitted = True
        return self
    
    def transform_sequences(self, df: pd.DataFrame, 
                           student_col: str = 'id_student',
                           time_col: str = 'date',
                           categorical_cols: List[str] = ['activity_type'],
                           numerical_cols: List[str] = ['sum_click', 'time_since_last']):
        """Transforms raw data into padded sequences for each student."""
        
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        # sort by student and time
        sorted_df = df.sort_values([student_col, time_col])
        
        # get unique students
        students = sorted_df[student_col].unique()
        
        # create sequences
        X_categorical = []
        X_numerical = []
        
        for student in students:
            student_data = sorted_df[sorted_df[student_col] == student]
            
            # handle too long sequences by taking the last max_seq_length entries
            if len(student_data) > self.max_seq_length:
                student_data = student_data.iloc[-self.max_seq_length:]
            
            # create categorical sequence
            if categorical_cols:
                cat_seq = []
                for col in categorical_cols:
                    # replace NaN with 'unknown' and encode
                    activity_values = student_data[col].fillna('unknown').values
                    encoded_activities = self.activity_encoder.transform(activity_values)
                    cat_seq.append(encoded_activities)
                
                # stack encoded categorical features
                if len(categorical_cols) > 1:
                    cat_seq = np.column_stack(cat_seq)
                else:
                    cat_seq = cat_seq[0].reshape(-1, 1)
                
                # pad sequence if needed
                if len(cat_seq) < self.max_seq_length:
                    padding = np.zeros((self.max_seq_length - len(cat_seq), cat_seq.shape[1]))
                    cat_seq = np.vstack([padding, cat_seq])
                
                X_categorical.append(cat_seq)
            
            # create numerical sequence
            if numerical_cols:
                num_seq = []
                for col in numerical_cols:
                    # scale values
                    values = student_data[col].fillna(0).values.reshape(-1, 1)
                    scaled_values = self.feature_scalers[col].transform(values).flatten()
                    num_seq.append(scaled_values)
                
                # stack scaled numerical features
                num_seq = np.column_stack(num_seq)
                
                # pad sequence if needed
                if len(num_seq) < self.max_seq_length:
                    padding = np.zeros((self.max_seq_length - len(num_seq), num_seq.shape[1]))
                    num_seq = np.vstack([padding, num_seq])
                
                X_numerical.append(num_seq)
        
        # convert lists to arrays
        if X_categorical:
            X_categorical = np.array(X_categorical)
        if X_numerical:
            X_numerical = np.array(X_numerical)
        
        # create mapping from student ID to sequence index
        student_index_map = {student: i for i, student in enumerate(students)}
        
        return {
            'categorical': X_categorical if categorical_cols else None,
            'numerical': X_numerical if numerical_cols else None,
            'students': students,
            'student_index_map': student_index_map
        }
    
    def save(self, filepath: str):
        """Saves preprocessor to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'max_seq_length': self.max_seq_length,
                'feature_scalers': self.feature_scalers,
                'activity_encoder': self.activity_encoder,
                'fitted': self.fitted
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SequencePreprocessor':
        """Loads preprocessor from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(max_seq_length=data['max_seq_length'])
        preprocessor.feature_scalers = data['feature_scalers']
        preprocessor.activity_encoder = data['activity_encoder']
        preprocessor.fitted = data['fitted']
        
        return preprocessor


class GRUModel:
    """GRU model for sequential feature path."""
    
    def __init__(self, 
                gru_units: int = 64,
                dense_units: List[int] = [32],
                dropout_rate: float = 0.3,
                learning_rate: float = 0.001,
                max_seq_length: int = 100,
                categorical_dim: Optional[int] = None,
                numerical_dim: Optional[int] = None):
        """Initializes model architecture parameters."""
        
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.categorical_dim = categorical_dim
        self.numerical_dim = numerical_dim
        
        self.model = None
        self.preprocessor = None
        self.history = None
        self.trained = False
    
    def build_model(self, categorical_dim: Optional[int] = None, 
                   numerical_dim: Optional[int] = None):
        """Builds GRU model architecture."""
        
        # utilize dimensions or instance variables
        if categorical_dim is not None:
            self.categorical_dim = categorical_dim
        if numerical_dim is not None:
            self.numerical_dim = numerical_dim
        
        # validate dimensions
        if self.categorical_dim is None and self.numerical_dim is None:
            raise ValueError("At least one of categorical_dim or numerical_dim must be specified")
        
        # define inputs
        inputs = []
        features = []
        
        # categorical features input
        if self.categorical_dim is not None:
            cat_input = Input(shape=(self.max_seq_length, self.categorical_dim), name='categorical_input')
            inputs.append(cat_input)
            features.append(cat_input)
        
        # numerical features input
        if self.numerical_dim is not None:
            num_input = Input(shape=(self.max_seq_length, self.numerical_dim), name='numerical_input')
            inputs.append(num_input)
            features.append(num_input)
        
        # combine features if both types exist
        if len(features) > 1:
            x = tf.keras.layers.Concatenate()(features)
        else:
            x = features[0]
        
        # gru layers
        x = GRU(self.gru_units, return_sequences=True)(x)
        x = Dropout(self.dropout_rate)(x)
        x = GRU(self.gru_units)(x)
        x = Dropout(self.dropout_rate)(x)
        
        # dense layers
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # output layer
        output = Dense(1, activation='sigmoid')(x)
        
        # create model
        model = Model(inputs=inputs, outputs=output)
        
        # compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def fit(self, 
           X_train: Dict[str, np.ndarray], 
           y_train: np.ndarray,
           X_val: Optional[Dict[str, np.ndarray]] = None, 
           y_val: Optional[np.ndarray] = None,
           epochs: int = 20,
           batch_size: int = 32,
           callbacks: List = None,
           class_weights: Optional[Dict] = None):
        """Trains the model on sequential data."""
        
        # build model if doesn't exist
        if self.model is None:
            categorical_dim = X_train['categorical'].shape[2] if 'categorical' in X_train and X_train['categorical'] is not None else None
            numerical_dim = X_train['numerical'].shape[2] if 'numerical' in X_train and X_train['numerical'] is not None else None
            self.build_model(categorical_dim, numerical_dim)
        
        # prepare inputs
        train_inputs = {}
        if 'categorical' in X_train and X_train['categorical'] is not None:
            train_inputs['categorical_input'] = X_train['categorical']
        if 'numerical' in X_train and X_train['numerical'] is not None:
            train_inputs['numerical_input'] = X_train['numerical']
        
        # prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            val_inputs = {}
            if 'categorical' in X_val and X_val['categorical'] is not None:
                val_inputs['categorical_input'] = X_val['categorical']
            if 'numerical' in X_val and X_val['numerical'] is not None:
                val_inputs['numerical_input'] = X_val['numerical']
            validation_data = (val_inputs, y_val)
        
        # default callbacks if none provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                # add model checkpoint if necessary
            ]
        
        # train model
        history = self.model.fit(
            train_inputs,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        self.history = history
        self.trained = True
        
        return history
    
    def predict_proba(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Predicts risk probabilities for sequential data."""
        
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # prepare inputs
        inputs = {}
        if 'categorical' in X and X['categorical'] is not None:
            inputs['categorical_input'] = X['categorical']
        if 'numerical' in X and X['numerical'] is not None:
            inputs['numerical_input'] = X['numerical']
        
        # make predictions
        return self.model.predict(inputs).flatten()
    
    def predict(self, X: Dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Predicts binary risk class using threshold."""
        
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def evaluate(self, X: Dict[str, np.ndarray], y: np.ndarray) -> Dict:
        """Evaluates model performance."""
        
        # prepare inputs
        inputs = {}
        if 'categorical' in X and X['categorical'] is not None:
            inputs['categorical_input'] = X['categorical']
        if 'numerical' in X and X['numerical'] is not None:
            inputs['numerical_input'] = X['numerical']
        
        # evaluate model
        results = self.model.evaluate(inputs, y, verbose=1)
        
        # create metrics dictionary
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'auc': results[2]
        }
        
        print(f"Model Performance:")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        
        return metrics
    
    def plot_training_history(self):
        """Plots training history metrics."""
        
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
        
        # plot training & validation accuracy
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path: str, preprocessor: Optional[SequencePreprocessor] = None):
        """Saves model and preprocessor to disk."""
        
        if not self.trained or self.model is None:
            raise ValueError("Cannot save untrained model")
        
        # create directory if doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # save keras model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # save preprocessor if provided
        if preprocessor is not None:
            preprocessor_path = model_path.replace('.keras', '_preprocessor.pkl')
            preprocessor.save(preprocessor_path)
            print(f"Preprocessor saved to {preprocessor_path}")
        
        # save model configuration
        config_path = model_path.replace('.keras', '_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump({
                'gru_units': self.gru_units,
                'dense_units': self.dense_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'max_seq_length': self.max_seq_length,
                'categorical_dim': self.categorical_dim,
                'numerical_dim': self.numerical_dim
            }, f)
        print(f"Model configuration saved to {config_path}")
    
    @classmethod
    def load_model(cls, model_path: str, load_preprocessor: bool = True) -> Tuple['GRUModel', Optional[SequencePreprocessor]]:
        """Loads model and preprocessor from disk."""
        
        # load model configuration
        config_path = model_path.replace('.keras', '_config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # create model instance with loaded configuration
        gru_model = cls(
            gru_units=config['gru_units'],
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            max_seq_length=config['max_seq_length'],
            categorical_dim=config['categorical_dim'],
            numerical_dim=config['numerical_dim']
        )
        
        # load Keras model
        gru_model.model = load_model(model_path)
        gru_model.trained = True
        
        # load preprocessor if requested
        preprocessor = None
        if load_preprocessor:
            preprocessor_path = model_path.replace('.keras', '_preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                preprocessor = SequencePreprocessor.load(preprocessor_path)
        
        return gru_model, preprocessor


def prepare_gru_training_data(sequential_features, student_info, train_ids, test_ids):
    """Prepares and splits data for GRU model training."""
    
    # get target variable
    target_df = student_info[['id_student', 'final_result']].copy()
    target_df['is_at_risk'] = target_df['final_result'].str.lower().apply(
        lambda x: 1 if x in ['fail', 'withdrawal'] else 0
    )
    
    # create preprocessor
    preprocessor = SequencePreprocessor(max_seq_length=100)
    
    # define feature columns
    categorical_cols = ['activity_type']
    numerical_cols = ['sum_click', 'time_since_last', 'cumulative_clicks']
    
    # fit preprocessor on training data only
    train_features = sequential_features[sequential_features['id_student'].isin(train_ids)]
    preprocessor.fit(train_features, categorical_cols, numerical_cols)
    
    # transform training and test data
    train_sequences = preprocessor.transform_sequences(
        train_features, 
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols
    )
    
    test_features = sequential_features[sequential_features['id_student'].isin(test_ids)]
    test_sequences = preprocessor.transform_sequences(
        test_features,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols
    )
    
    # create target arrays
    y_train = np.array([
        target_df[target_df['id_student'] == student]['is_at_risk'].iloc[0]
        for student in train_sequences['students']
    ])
    
    y_test = np.array([
        target_df[target_df['id_student'] == student]['is_at_risk'].iloc[0]
        for student in test_sequences['students']
    ])
    
    return {
        'X_train': train_sequences,
        'y_train': y_train,
        'X_test': test_sequences,
        'y_test': y_test,
        'preprocessor': preprocessor
    }