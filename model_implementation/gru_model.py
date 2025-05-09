import logging
import numpy as np
import pandas as pd
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Dense, GRU, Input, Embedding, Dropout, Masking, 
    BatchNormalization, concatenate, Bidirectional
)
from tensorflow.keras.callbacks import ( # type: ignore
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard
)
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from tensorflow.keras.metrics import AUC, Precision, Recall # type: ignore
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Import project-specific modules
from config import DIRS, FEATURE_ENGINEERING, RANDOM_SEED, FAIRNESS
from evaluation.fairness_analysis import calculate_fairness_metrics, analyze_bias_patterns

# Set random seed for reproducibility
tf.random.set_seed(RANDOM_SEED)

class GRUModel:
    """
    GRU (Gated Recurrent Unit) model for EduPredict with sequential feature support
    and fairness-aware evaluation.
    """
    
    def __init__(
        self,
        seq_length: int = 100,
        gru_units: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        embedding_dims: int = 32,
        use_static_features: bool = True,
        bidirectional: bool = True,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes GRU model with hyperparameters.
        
        Args:
            seq_length: Maximum sequence length for input data
            gru_units: Number of units in GRU layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            embedding_dims: Dimensions for categorical feature embeddings
            use_static_features: Whether to use static features alongside sequences
            bidirectional: Whether to use bidirectional GRU
            logger: Logger for tracking model lifecycle
        """
        self.logger = logger or logging.getLogger('edupredict')
        
        # Store hyperparameters
        self.seq_length = seq_length
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.embedding_dims = embedding_dims
        self.use_static_features = use_static_features
        self.bidirectional = bidirectional
        
        # Initialize model attributes
        self.model = None
        self.history = None
        self.feature_config = None
        
        # Store hyperparameters for reference/metadata
        self.hyperparams = {
            'seq_length': seq_length,
            'gru_units': gru_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'embedding_dims': embedding_dims,
            'use_static_features': use_static_features,
            'bidirectional': bidirectional
        }
        
        # Initialize metadata container
        self.metadata = {
            'model_type': 'GRU',
            'hyperparameters': self.hyperparams,
            'training_history': {},
            'evaluation_metrics': {}
        }
        
        self.logger.info(f"Initialized GRU model with {gru_units} units")
    
    def build_model(
        self,
        categorical_dims: Dict[str, int],
        numerical_features: int = 0,
        static_features: int = 0,
        temporal_features: int = 0
    ) -> None:
        """
        Builds the GRU model architecture.
        
        Args:
            categorical_dims: Dictionary mapping categorical features to dimensions
            numerical_features: Number of numerical features in sequences
            static_features: Number of static features
            temporal_features: Number of temporal context features
        """
        try:
            # Store feature configuration
            self.feature_config = {
                'categorical_dims': categorical_dims,
                'numerical_features': numerical_features,
                'static_features': static_features,
                'temporal_features': temporal_features
            }
            
            # Create inputs for categorical features
            categorical_inputs = []
            categorical_embeddings = []
            for feature_name, dim in categorical_dims.items():
                categorical_input = Input(
                    shape=(self.seq_length, dim),
                    name=f"input_cat_{feature_name}"
                )
                categorical_inputs.append(categorical_input)
                categorical_embeddings.append(categorical_input)
            
            # Create input for numerical features
            numerical_input = None
            if numerical_features > 0:
                numerical_input = Input(
                    shape=(self.seq_length, numerical_features),
                    name="input_numerical"
                )
            
            # Create input for temporal features
            temporal_input = None
            if temporal_features > 0:
                temporal_input = Input(
                    shape=(self.seq_length, temporal_features),
                    name="input_temporal"
                )
            
            # Create input for sequence mask
            mask_input = Input(
                shape=(self.seq_length,),
                name="input_mask"
            )
            
            # Combine all sequential features
            sequence_features = []
            if len(categorical_embeddings) > 0:
                if len(categorical_embeddings) > 1:
                    sequence_features.append(concatenate(categorical_embeddings, axis=-1))
                else:
                    sequence_features.append(categorical_embeddings[0])
            
            if numerical_input is not None:
                sequence_features.append(numerical_input)
                
            if temporal_input is not None:
                sequence_features.append(temporal_input)
            
            # Combine sequence features if multiple types exist
            if len(sequence_features) > 1:
                combined_sequences = concatenate(sequence_features, axis=-1)
            elif len(sequence_features) == 1:
                combined_sequences = sequence_features[0]
            else:
                raise ValueError("No sequence features provided")
            
            # Apply mask
            expanded_mask = tf.expand_dims(mask_input, axis=-1)
            masked_sequences = combined_sequences * expanded_mask
            masked_layer = Masking(mask_value=0.0)(masked_sequences)
            
            # Apply GRU layer with optional bidirectional wrapper
            if self.bidirectional:
                gru_output = Bidirectional(
                    GRU(self.gru_units, return_sequences=False)
                )(masked_layer)
            else:
                gru_output = GRU(
                    self.gru_units, return_sequences=False
                )(masked_layer)
            
            gru_output = Dropout(self.dropout_rate)(gru_output)
            
            # Process static features if provided
            static_input = None
            if self.use_static_features and static_features > 0:
                static_input = Input(
                    shape=(static_features,),
                    name="input_static"
                )
                static_normalized = BatchNormalization()(static_input)
                combined_features = concatenate([gru_output, static_normalized])
            else:
                combined_features = gru_output
            
            # Dense layers
            x = Dense(32, activation='relu')(combined_features)
            x = Dropout(self.dropout_rate)(x)
            x = Dense(16, activation='relu')(x)
            
            # Output layer
            output = Dense(1, activation='sigmoid', name='output')(x)
            
            # Define model inputs
            model_inputs = categorical_inputs + [mask_input]
            if numerical_input is not None:
                model_inputs.append(numerical_input)
            if temporal_input is not None:
                model_inputs.append(temporal_input)
            if static_input is not None:
                model_inputs.append(static_input)
            
            # Create and compile model
            self.model = Model(inputs=model_inputs, outputs=output)
            
            # Compile with metrics
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    AUC(name='auc'),
                    Precision(name='precision'),
                    Recall(name='recall')
                ]
            )
            
            self.logger.info("GRU Model built successfully")
            self.logger.info(f"Model inputs: {[input.name for input in self.model.inputs]}")
            self.logger.info(f"Model has {self.model.count_params()} parameters")
            
            # Store architecture summary
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            self.metadata['model_architecture'] = '\n'.join(stringlist)
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise
    
    def prepare_inputs(self, sequence_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Prepares model inputs from sequence data.
        
        Args:
            sequence_data: Dictionary with sequence data from SequencePreprocessor
        
        Returns:
            Dictionary of model inputs
        """
        try:
            inputs = {}
            
            # Process categorical features
            for feature_name in self.feature_config['categorical_dims'].keys():
                if f'cat_{feature_name}' in sequence_data:
                    inputs[f'input_cat_{feature_name}'] = sequence_data[f'cat_{feature_name}']
            
            # Process numerical features if any
            numerical_features = []
            for key in sequence_data.keys():
                if key.startswith('num_'):
                    numerical_features.append(sequence_data[key])
            
            if numerical_features:
                # Combine all numerical features along the feature axis
                inputs['input_numerical'] = np.concatenate(
                    [feat.reshape(feat.shape[0], feat.shape[1], 1) for feat in numerical_features],
                    axis=2
                )
            
            # Process temporal features if any
            if 'temporal_features' in sequence_data:
                inputs['input_temporal'] = sequence_data['temporal_features']
            
            # Add mask input
            inputs['input_mask'] = sequence_data['mask']
            
            # Add static features if available and used
            if self.use_static_features and 'static_features' in sequence_data:
                inputs['input_static'] = sequence_data['static_features']
            
            return inputs
            
        except Exception as e:
            self.logger.error(f"Error preparing model inputs: {str(e)}")
            raise
    
    def fit(
        self,
        train_data: Dict[str, Any],
        y_train: np.ndarray,
        validation_data: Optional[Tuple[Dict[str, Any], np.ndarray]] = None,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        save_best_only: bool = True,
        class_weights: Optional[Dict[int, float]] = None
    ) -> Any:
        """
        Trains the model on provided sequence data.
        
        Args:
            train_data: Training sequence data
            y_train: Training labels
            validation_data: Tuple of validation sequence data and labels
            batch_size: Batch size for training
            epochs: Number of epochs to train
            patience: Patience for early stopping
            save_best_only: Whether to save only the best model
            class_weights: Optional class weights for imbalanced data
        
        Returns:
            Training history
        """
        try:
            # Ensure model is built
            if self.model is None:
                raise ValueError("Model not built. Call build_model first.")
            
            # Prepare training inputs
            X_train = self.prepare_inputs(train_data)
            
            # Prepare validation inputs if provided
            validation_inputs = None
            if validation_data is not None:
                val_data, val_labels = validation_data
                X_val = self.prepare_inputs(val_data)
                validation_inputs = (X_val, val_labels)
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=patience // 2,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Add model checkpoint if requested
            if save_best_only:
                checkpoint_dir = Path(DIRS['checkpoints']) / 'gru'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = str(checkpoint_dir / 'gru_model_best.h5')
                
                callbacks.append(
                    ModelCheckpoint(
                        filepath=checkpoint_path,
                        monitor='val_loss' if validation_data else 'loss',
                        save_best_only=True,
                        verbose=1
                    )
                )
            
            # Add TensorBoard callback for visualization
            log_dir = Path(DIRS['logs']) / 'tensorboard' / 'gru'
            log_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                TensorBoard(
                    log_dir=str(log_dir),
                    histogram_freq=1
                )
            )
            
            # Log training start
            self.logger.info(f"Starting GRU model training for {epochs} epochs with batch size {batch_size}")
            if class_weights:
                self.logger.info(f"Using class weights: {class_weights}")
            
            # Train the model
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=validation_inputs,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=2
            )
            
            # Store training history in metadata
            self.history = history.history
            self.metadata['training_history'] = {
                'epochs': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1]),
                'final_accuracy': float(history.history['accuracy'][-1])
            }
            
            if validation_data:
                self.metadata['training_history']['final_val_loss'] = float(history.history['val_loss'][-1])
                self.metadata['training_history']['final_val_accuracy'] = float(history.history['val_accuracy'][-1])
            
            # Plot training history if available
            if history.history:
                self._plot_training_history(history.history)
            
            # Log completion
            self.logger.info("GRU model training completed")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict_proba(self, sequence_data: Dict[str, Any]) -> np.ndarray:
        """
        Predicts risk probabilities from sequence data.
        
        Args:
            sequence_data: Dictionary with sequence data
        
        Returns:
            Array of positive class probabilities
        """
        try:
            # Ensure model exists
            if self.model is None:
                raise ValueError("Model not built or trained")
            
            # Prepare inputs
            X = self.prepare_inputs(sequence_data)
            
            # Get probability predictions
            proba = self.model.predict(X)
            
            # Return probabilities
            return proba.flatten()
            
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise
    
    def predict(self, sequence_data: Dict[str, Any], threshold: float = 0.5) -> np.ndarray:
        """
        Predicts binary risk class from sequence data.
        
        Args:
            sequence_data: Dictionary with sequence data
            threshold: Classification threshold
        
        Returns:
            Array of binary predictions
        """
        try:
            # Get probability predictions
            proba = self.predict_proba(sequence_data)
            
            # Apply threshold to convert to binary predictions
            return (proba >= threshold).astype(int)
            
        except Exception as e:
            self.logger.error(f"Error during binary prediction: {str(e)}")
            raise
    
    def evaluate(
        self,
        sequence_data: Dict[str, Any],
        y_true: np.ndarray,
        threshold: float = 0.5,
        fairness_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates model performance.
        
        Args:
            sequence_data: Dictionary with sequence data
            y_true: True labels
            threshold: Classification threshold
            fairness_params: Dictionary with fairness evaluation parameters
        
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Get predictions
            y_prob = self.predict_proba(sequence_data)
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate standard performance metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'auc': roc_auc_score(y_true, y_prob),
                'threshold': threshold
            }
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['confusion_matrix'] = {
                'tn': int(tn), 
                'fp': int(fp), 
                'fn': int(fn), 
                'tp': int(tp)
            }
            
            # Add fairness evaluation if parameters provided
            if fairness_params:
                if 'protected_attributes' in fairness_params:
                    protected_cols = fairness_params['protected_attributes']
                    if 'protected_data' in fairness_params:
                        protected_features = fairness_params['protected_data']
                        
                        # Calculate fairness metrics
                        fairness_metrics = calculate_fairness_metrics(
                            y_true, 
                            y_prob, 
                            protected_features,
                            threshold=threshold
                        )
                        
                        # Analyze bias patterns
                        bias_analysis = analyze_bias_patterns(
                            protected_features, 
                            y_pred, 
                            y_true,
                            metadata={
                                'model_type': 'GRU',
                                'evaluation_date': pd.Timestamp.now().isoformat()
                            }
                        )
                        
                        # Add to metrics
                        metrics['fairness_metrics'] = fairness_metrics
                        metrics['bias_analysis'] = bias_analysis
                    else:
                        self.logger.warning("Protected_data not provided in fairness_params")
                else:
                    self.logger.warning("Protected_attributes not specified in fairness_params")
            
            # Update model metadata
            self.metadata['evaluation_metrics'] = metrics
            
            self.logger.info(f"Model evaluation completed: accuracy={metrics['accuracy']:.4f}, "
                            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                            f"f1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            raise
    
    def save_model(
        self, 
        filepath: str, 
        include_metadata: bool = True, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Saves model to disk with optional metadata.
        
        Args:
            filepath: Path to save model
            include_metadata: Whether to include metadata
            metadata: Dictionary of metadata to save
        
        Returns:
            None
        """
        try:
            # Ensure model exists
            if self.model is None:
                raise ValueError("No model to save")
            
            # Resolve filepath
            model_path = Path(filepath)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the keras model
            self.model.save(str(model_path))
            self.logger.info(f"Model saved to {model_path}")
            
            # Save metadata if requested
            if include_metadata:
                # Combine instance metadata with provided metadata
                model_metadata = self.metadata.copy()
                if metadata:
                    model_metadata.update(metadata)
                
                # Add timestamp
                model_metadata['save_timestamp'] = pd.Timestamp.now().isoformat()
                
                # Save feature configuration
                if self.feature_config:
                    model_metadata['feature_config'] = self.feature_config
                
                # Save metadata separately in JSON format for easier access
                metadata_path = model_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata, f, indent=2, default=str)
                self.logger.info(f"Model metadata saved to {metadata_path}")
                
                # Save configuration separately for model reconstruction
                config_path = model_path.with_name(f"{model_path.stem}_config.pkl")
                with open(config_path, 'wb') as f:
                    pickle.dump({
                        'hyperparams': self.hyperparams,
                        'feature_config': self.feature_config
                    }, f)
                self.logger.info(f"Model configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, 
        filepath: str, 
        custom_objects: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ) -> 'GRUModel':
        """
        Loads model from disk.
        
        Args:
            filepath: Path to saved model
            custom_objects: Dictionary of custom objects for keras model loading
            logger: Logger for tracking model loading
        
        Returns:
            Loaded GRUModel instance
        """
        try:
            logger = logger or logging.getLogger('edupredict')
            logger.info(f"Loading GRU model from {filepath}")
            
            # Resolve filepath
            model_path = Path(filepath)
            
            # Load configuration
            config_path = model_path.with_name(f"{model_path.stem}_config.pkl")
            if config_path.exists():
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
                hyperparams = config['hyperparams']
                feature_config = config.get('feature_config')
            else:
                logger.warning(f"Configuration file not found at {config_path}. Using defaults.")
                hyperparams = {}
                feature_config = None
            
            # Create instance with loaded hyperparameters
            instance = cls(
                seq_length=hyperparams.get('seq_length', 100),
                gru_units=hyperparams.get('gru_units', 64),
                dropout_rate=hyperparams.get('dropout_rate', 0.3),
                learning_rate=hyperparams.get('learning_rate', 0.001),
                embedding_dims=hyperparams.get('embedding_dims', 32),
                use_static_features=hyperparams.get('use_static_features', True),
                bidirectional=hyperparams.get('bidirectional', True),
                logger=logger
            )
            
            # Store the feature configuration
            instance.feature_config = feature_config
            
            # Load the keras model
            instance.model = load_model(filepath, custom_objects=custom_objects)
            
            # Load metadata if available
            metadata_path = model_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    instance.metadata = json.load(f)
            
            logger.info(f"Successfully loaded GRU model")
            
            return instance
            
        except Exception as e:
            if logger:
                logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plots training history.
        
        Args:
            history: Training history dictionary
        
        Returns:
            None
        """
        try:
            # Create figure with multiple subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot training & validation accuracy
            axes[0].plot(history['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in history:
                axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
            axes[0].set_title('Model Accuracy')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].legend()
            
            # Plot training & validation loss
            axes[1].plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[1].plot(history['val_loss'], label='Validation Loss')
            axes[1].set_title('Model Loss')
            axes[1].set_ylabel('Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            viz_dir = Path(DIRS['viz_training'])
            viz_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(viz_dir / 'gru_training_history.png'))
            plt.close(fig)
            
            self.logger.info(f"Training history visualization saved to {viz_dir / 'gru_training_history.png'}")
            
        except Exception as e:
            self.logger.warning(f"Error plotting training history: {str(e)}")


def tune_gru_model(
    train_data: Dict[str, Any],
    y_train: np.ndarray,
    val_data: Dict[str, Any],
    y_val: np.ndarray,
    param_grid: Dict[str, List[Any]],
    n_trials: int = 10,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Tunes GRU model hyperparameters.
    
    Args:
        train_data: Training sequence data
        y_train: Training labels
        val_data: Validation sequence data
        y_val: Validation labels
        param_grid: Dictionary of hyperparameter options
        n_trials: Number of random hyperparameter combinations to try
        logger: Logger for tracking tuning process
    
    Returns:
        Tuple of best hyperparameters and results dictionary
    """
    try:
        import numpy as np
        from sklearn.model_selection import ParameterSampler
        
        logger = logger or logging.getLogger('edupredict')
        logger.info(f"Starting GRU hyperparameter tuning with {n_trials} trials")
        
        # Default parameters to use
        default_params = {
            'seq_length': train_data.get('mask', np.array([[100]])).shape[1],
            'gru_units': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'embedding_dims': 32,
            'use_static_features': 'static_features' in train_data,
            'bidirectional': True,
            'batch_size': 32,
            'epochs': 30,
            'patience': 5
        }
        
        # Generate random hyperparameter combinations
        param_list = list(ParameterSampler(
            param_grid, 
            n_iter=n_trials, 
            random_state=RANDOM_SEED
        ))
        
        # Track results for each combination
        results = []
        best_val_auc = 0
        best_params = None
        best_model = None
        
        # Set up directories for tuning results
        tuning_dir = Path(DIRS['checkpoints']) / 'gru_tuning'
        tuning_dir.mkdir(parents=True, exist_ok=True)
        
        # Try each hyperparameter combination
        for i, params in enumerate(param_list):
            try:
                # Fill in defaults for missing parameters
                for key, value in default_params.items():
                    if key not in params:
                        params[key] = value
                
                logger.info(f"Trial {i+1}/{n_trials} with parameters: {params}")
                
                # Create and build model
                model = GRUModel(
                    seq_length=params['seq_length'],
                    gru_units=params['gru_units'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate'],
                    embedding_dims=params['embedding_dims'],
                    use_static_features=params['use_static_features'],
                    bidirectional=params['bidirectional'],
                    logger=logger
                )
                
                # Get categorical dimensions from data
                categorical_dims = {}
                for key, value in train_data.items():
                    if key.startswith('cat_'):
                        feature_name = key[4:]  # Remove 'cat_' prefix
                        categorical_dims[feature_name] = value.shape[2]
                
                # Count numerical features
                numerical_features = sum(1 for key in train_data if key.startswith('num_'))
                
                # Get static features dimension if used
                static_features = train_data['static_features'].shape[1] if params['use_static_features'] and 'static_features' in train_data else 0
                
                # Build model
                model.build_model(
                    categorical_dims=categorical_dims,
                    numerical_features=numerical_features,
                    static_features=static_features
                )
                
                # Train model
                history = model.fit(
                    train_data=train_data,
                    y_train=y_train,
                    validation_data=(val_data, y_val),
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    patience=params['patience'],
                    save_best_only=False
                )
                
                # Evaluate model
                metrics = model.evaluate(
                    sequence_data=val_data,
                    y_true=y_val
                )
                
                # Record results
                trial_result = {
                    'trial': i+1,
                    'params': params,
                    'val_accuracy': metrics['accuracy'],
                    'val_precision': metrics['precision'],
                    'val_recall': metrics['recall'],
                    'val_f1': metrics['f1'],
                    'val_auc': metrics['auc']
                }
                results.append(trial_result)
                
                # Update best model if this one is better
                if metrics['auc'] > best_val_auc:
                    best_val_auc = metrics['auc']
                    best_params = params.copy()
                    best_model = model
                    
                    # Save best model so far
                    best_model.save_model(
                        str(tuning_dir / 'best_gru_model.h5'),
                        include_metadata=True,
                        metadata={'tuning_trial': i+1}
                    )
                
                logger.info(f"Trial {i+1} completed: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"Error in trial {i+1}: {str(e)}")
                results.append({
                    'trial': i+1,
                    'params': params,
                    'error': str(e)
                })
        
        # Save all results to file
        results_df = pd.DataFrame(results)
        results_path = Path(DIRS['reports']) / 'gru_tuning_results.csv'
        results_df.to_csv(results_path, index=False)
        
        # Log best parameters
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best validation AUC: {best_val_auc:.4f}")
        
        # Return best parameters and results
        return best_params, {'results_df': results_df, 'best_model': best_model}
        
    except Exception as e:
        if logger:
            logger.error(f"Error during GRU hyperparameter tuning: {str(e)}")
        raise


def find_optimal_threshold(
    model: GRUModel,
    val_data: Dict[str, Any],
    y_val: np.ndarray,
    metric: str = 'f1',
    fairness_aware: bool = False,
    protected_data: Optional[pd.DataFrame] = None,
    protected_attributes: Optional[List[str]] = None,
    fairness_thresholds: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Finds optimal classification threshold.
    
    Args:
        model: Trained GRU model
        val_data: Validation sequence data
        y_val: Validation labels
        metric: Metric to optimize ('accuracy', 'precision', 'recall', 'f1', etc.)
        fairness_aware: Whether to consider fairness in threshold selection
        protected_data: DataFrame with protected attributes
        protected_attributes: List of protected attribute columns
        fairness_thresholds: Dictionary of thresholds for fairness metrics
        logger: Logger for tracking threshold optimization
    
    Returns:
        Dictionary with optimal threshold and metrics
    """
    try:
        logger = logger or logging.getLogger('edupredict')
        logger.info(f"Finding optimal threshold optimizing {metric}")
        
        # Get probability predictions
        y_prob = model.predict_proba(val_data)
        
        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, 81)  # Test from 0.1 to 0.9 in 0.01 increments
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            if metric == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred)
            elif metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'balanced_accuracy':
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (tpr + tnr) / 2
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            results.append({
                'threshold': threshold,
                'score': score
            })
        
        # Find best threshold
        results_df = pd.DataFrame(results)
        best_idx = results_df['score'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_score = results_df.loc[best_idx, 'score']
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} with {metric} score: {best_score:.4f}")
        
        result = {
            'threshold': float(best_threshold),
            f'{metric}_score': float(best_score)
        }
        
        # If fairness-aware, find demographic-specific thresholds
        if fairness_aware and protected_data is not None and protected_attributes:
            logger.info("Finding demographic-specific thresholds")
            
            demographic_thresholds = {}
            
            # Process each protected attribute
            for attr in protected_attributes:
                if attr in protected_data.columns:
                    group_thresholds = {}
                    
                    # Get unique groups
                    groups = protected_data[attr].unique()
                    
                    # Find optimal threshold for each group
                    for group in groups:
                        group_mask = protected_data[attr] == group
                        
                        # Skip if not enough samples
                        min_samples = FAIRNESS.get('min_group_size', 50)
                        if group_mask.sum() < min_samples:
                            logger.warning(f"Group {group} in {attr} has fewer than {min_samples} samples. Skipping.")
                            continue
                        
                        # Extract group-specific data
                        group_y_val = y_val[group_mask]
                        group_y_prob = y_prob[group_mask]
                        
                        # Try different thresholds for this group
                        group_results = []
                        for threshold in thresholds:
                            group_y_pred = (group_y_prob >= threshold).astype(int)
                            
                            # Calculate metric
                            if metric == 'accuracy':
                                score = accuracy_score(group_y_val, group_y_pred)
                            elif metric == 'precision':
                                score = precision_score(group_y_val, group_y_pred)
                            elif metric == 'recall':
                                score = recall_score(group_y_val, group_y_pred)
                            elif metric == 'f1':
                                score = f1_score(group_y_val, group_y_pred)
                            elif metric == 'balanced_accuracy':
                                tn, fp, fn, tp = confusion_matrix(group_y_val, group_y_pred).ravel()
                                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                                score = (tpr + tnr) / 2
                            
                            group_results.append({
                                'threshold': threshold,
                                'score': score
                            })
                        
                        # Find best threshold for this group
                        group_results_df = pd.DataFrame(group_results)
                        if not group_results_df.empty:
                            group_best_idx = group_results_df['score'].idxmax()
                            group_best_threshold = group_results_df.loc[group_best_idx, 'threshold']
                            group_best_score = group_results_df.loc[group_best_idx, 'score']
                            
                            logger.info(f"Optimal threshold for {attr}={group}: {group_best_threshold:.4f} "
                                       f"with {metric} score: {group_best_score:.4f}")
                            
                            group_thresholds[str(group)] = {
                                'threshold': float(group_best_threshold),
                                f'{metric}_score': float(group_best_score)
                            }
                    
                    demographic_thresholds[attr] = group_thresholds
                else:
                    logger.warning(f"Protected attribute {attr} not found in data")
            
            # Add demographic thresholds to result
            result['demographic_thresholds'] = demographic_thresholds
        
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"Error finding optimal threshold: {str(e)}")
        raise