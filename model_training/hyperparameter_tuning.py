import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def tune_random_forest(X_train, y_train, 
                      param_grid: Optional[Dict] = None,
                      cv: int = 5,
                      scoring: str = 'f1',
                      n_jobs: int = -1,
                      random_search: bool = True,
                      n_iter: int = 20,
                      verbose: int = 1) -> Tuple[Dict, Any]:
    """Tunes random forest hyperparameters using grid or random search."""
    
    from sklearn.ensemble import RandomForestClassifier
    
    # default param grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
    
    # set up scoring metric
    if scoring == 'f1':
        scorer = make_scorer(f1_score)
    elif scoring == 'accuracy':
        scorer = make_scorer(accuracy_score)
    elif scoring == 'roc_auc':
        scorer = make_scorer(roc_auc_score)
    else:
        scorer = scoring
    
    # initialize base model
    base_model = RandomForestClassifier(random_state=0)
    
    # perform hyperparameter search
    if random_search:
        print("Performing RandomizedSearchCV...")
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=0,
            return_train_score=True
        )
    else:
        print("Performing GridSearchCV...")
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
    
    # fit search
    search.fit(X_train, y_train)
    
    # get best parameters and model
    best_params = search.best_params_
    best_model = search.best_estimator_
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV {scoring} score: {search.best_score_:.4f}")
    
    return best_params, best_model


def tune_gru_hyperparameters(X_train_dict, y_train, X_val_dict, y_val,
                            param_grid: Optional[Dict] = None,
                            epochs: int = 20,
                            batch_size: int = 32,
                            early_stopping_patience: int = 5,
                            verbose: int = 1) -> Tuple[Dict, Dict]:
    """Tunes GRU hyperparameters using manual validation loop."""
    
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from itertools import product
    import time
    
    # default param grid if none provided
    if param_grid is None:
        param_grid = {
            'gru_units': [32, 64, 128],
            'dense_units': [[32], [64], [32, 16]],
            'dropout_rate': [0.2, 0.3, 0.5],
            'learning_rate': [0.001, 0.0005]
        }
    
    # generate all combinations of parameters
    keys = list(param_grid.keys())
    all_combinations = list(product(*[param_grid[key] for key in keys]))
    all_params = [dict(zip(keys, combo)) for combo in all_combinations]
    
    print(f"Testing {len(all_params)} hyperparameter combinations...")
    
    # set up early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True
    )
    
    # track results
    results = []
    
    # evaluate each parameter combination
    for i, params in enumerate(all_params):
        print(f"\nTesting combination {i+1}/{len(all_params)}: {params}")
        start_time = time.time()
        
        # build model with current parameters
        model = build_gru_model(
            categorical_dim=X_train_dict['categorical'].shape[2] if 'categorical' in X_train_dict and X_train_dict['categorical'] is not None else None,
            numerical_dim=X_train_dict['numerical'].shape[2] if 'numerical' in X_train_dict and X_train_dict['numerical'] is not None else None,
            gru_units=params['gru_units'],
            dense_units=params['dense_units'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        )
        
        # prepare inputs
        train_inputs = {}
        val_inputs = {}
        
        if 'categorical' in X_train_dict and X_train_dict['categorical'] is not None:
            train_inputs['categorical_input'] = X_train_dict['categorical']
            val_inputs['categorical_input'] = X_val_dict['categorical']
        
        if 'numerical' in X_train_dict and X_train_dict['numerical'] is not None:
            train_inputs['numerical_input'] = X_train_dict['numerical']
            val_inputs['numerical_input'] = X_val_dict['numerical']
        
        # train model
        history = model.fit(
            train_inputs,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, y_val),
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        # evaluate model
        val_loss, val_accuracy, val_auc = model.evaluate(val_inputs, y_val, verbose=0)
        
        # calculate validation f1 score
        val_pred = (model.predict(val_inputs) > 0.5).astype(int).flatten()
        val_f1 = f1_score(y_val, val_pred)
        
        # track result
        elapsed_time = time.time() - start_time
        best_epoch = np.argmin(history.history['val_loss']) + 1
        result = {
            **params,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'best_epoch': best_epoch,
            'elapsed_time': elapsed_time
        }
        results.append(result)
        
        print(f"Validation metrics - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
        print(f"Best epoch: {best_epoch}, Elapsed time: {elapsed_time:.2f}s")
    
    # convert results to dataframe
    results_df = pd.DataFrame(results)
    
    # find best parameters
    best_idx = results_df['val_f1'].idxmax()
    best_params = {
        key: results_df.loc[best_idx, key] for key in keys
    }
    best_metrics = {
        'val_loss': results_df.loc[best_idx, 'val_loss'],
        'val_accuracy': results_df.loc[best_idx, 'val_accuracy'],
        'val_auc': results_df.loc[best_idx, 'val_auc'],
        'val_f1': results_df.loc[best_idx, 'val_f1'],
        'best_epoch': results_df.loc[best_idx, 'best_epoch'],
        'elapsed_time': results_df.loc[best_idx, 'elapsed_time']
    }
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation F1 score: {best_metrics['val_f1']:.4f}")
    
    return best_params, results_df


def build_gru_model(categorical_dim=None, numerical_dim=None, 
                   gru_units=64, dense_units=[32], 
                   dropout_rate=0.3, learning_rate=0.001):
    """Builds GRU model with specified architecture."""
    
    import tensorflow as tf
    from tensorflow.keras.layers import GRU, Dense, Input, Dropout, Concatenate
    from tensorflow.keras.models import Model
    
    # validate inputs
    if categorical_dim is None and numerical_dim is None:
        raise ValueError("At least one of categorical_dim or numerical_dim must be specified")
    
    # define inputs
    inputs = []
    
    # categorical features
    if categorical_dim is not None:
        cat_input = Input(shape=(None, categorical_dim), name='categorical_input')
        inputs.append(cat_input)
    
    # numerical features
    if numerical_dim is not None:
        num_input = Input(shape=(None, numerical_dim), name='numerical_input')
        inputs.append(num_input)
    
    # combine inputs if needed
    if len(inputs) > 1:
        x = Concatenate()(inputs)
    else:
        x = inputs[0]
    
    # gru layers
    x = GRU(gru_units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = GRU(gru_units)(x)
    x = Dropout(dropout_rate)(x)
    
    # dense layers
    for units in dense_units:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    
    # output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # create model
    model = Model(inputs=inputs, outputs=output)
    
    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model


def optimize_ensemble_weights(rf_probs, gru_probs, y_true, 
                             weight_steps: int = 21,
                             threshold_steps: int = 21,
                             metric: str = 'f1') -> Tuple[Dict, pd.DataFrame]:
    """Finds optimal ensemble weights and threshold."""
    
    # create grid of weights and thresholds
    rf_weights = np.linspace(0, 1, weight_steps)
    thresholds = np.linspace(0, 1, threshold_steps)
    
    # track results
    results = []
    
    for rf_weight in rf_weights:
        gru_weight = 1 - rf_weight
        
        # combine predictions
        ensemble_probs = rf_weight * rf_probs + gru_weight * gru_probs
        
        for threshold in thresholds:
            # make binary predictions
            ensemble_preds = (ensemble_probs >= threshold).astype(int)
            
            # calculate metrics
            if metric == 'f1':
                score = f1_score(y_true, ensemble_preds)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, ensemble_preds)
            elif metric == 'roc_auc':
                score = roc_auc_score(y_true, ensemble_probs)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            results.append({
                'rf_weight': rf_weight,
                'gru_weight': gru_weight,
                'threshold': threshold,
                metric: score
            })
    
    # convert to dataframe
    results_df = pd.DataFrame(results)
    
    # find best parameters
    best_idx = results_df[metric].idxmax()
    best_params = {
        'rf_weight': results_df.loc[best_idx, 'rf_weight'],
        'gru_weight': results_df.loc[best_idx, 'gru_weight'],
        'threshold': results_df.loc[best_idx, 'threshold'],
        'best_score': results_df.loc[best_idx, metric]
    }
    
    print(f"Best ensemble parameters: RF weight = {best_params['rf_weight']:.3f}, "
          f"GRU weight = {best_params['gru_weight']:.3f}, threshold = {best_params['threshold']:.3f}")
    print(f"Best {metric} score: {best_params['best_score']:.4f}")
    
    return best_params, results_df


def visualize_tuning_results(results_df: pd.DataFrame, 
                           x_col: str, 
                           y_col: str, 
                           hue_col: Optional[str] = None,
                           title: str = 'Hyperparameter Tuning Results',
                           save_path: Optional[str] = None):
    """Visualizes hyperparameter tuning results."""
    
    plt.figure(figsize=(10, 6))
    
    if hue_col:
        sns.lineplot(x=x_col, y=y_col, hue=hue_col, data=results_df, marker='o')
    else:
        sns.lineplot(x=x_col, y=y_col, data=results_df, marker='o')
    
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # add marker at best point
    best_idx = results_df[y_col].idxmax()
    best_x = results_df.loc[best_idx, x_col]
    best_y = results_df.loc[best_idx, y_col]
    plt.scatter(best_x, best_y, color='red', s=100, zorder=10, label=f'Best: {best_y:.4f}')
    
    plt.legend(title=hue_col if hue_col else None)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_ensemble_weights(results_df: pd.DataFrame, 
                              metric: str = 'f1',
                              save_path: Optional[str] = None):
    """Visualizes ensemble weight optimization results."""
    
    # prepare data for heatmap
    pivot_df = results_df.pivot_table(
        index='rf_weight', 
        columns='threshold', 
        values=metric
    )
    
    plt.figure(figsize=(12, 8))
    
    # plot heatmap
    sns.heatmap(
        pivot_df, 
        annot=False, 
        cmap='viridis', 
        cbar_kws={'label': metric.upper()}
    )
    
    # find best parameters
    best_idx = results_df[metric].idxmax()
    best_rf_weight = results_df.loc[best_idx, 'rf_weight']
    best_threshold = results_df.loc[best_idx, 'threshold']
    
    # highlight best parameters
    plt.scatter(
        pivot_df.columns.get_loc(best_threshold) + 0.5, 
        pivot_df.index.get_loc(best_rf_weight) + 0.5, 
        color='red', 
        marker='*', 
        s=200, 
        label=f'Best: RF={best_rf_weight:.2f}, Thresh={best_threshold:.2f}'
    )
    
    plt.title(f'Ensemble {metric.upper()} Score by RF Weight and Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Random Forest Weight')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    # also show contour plot
    plt.figure(figsize=(10, 6))
    
    contour = plt.contourf(
        pivot_df.columns, 
        pivot_df.index, 
        pivot_df.values, 
        levels=20, 
        cmap='viridis'
    )
    plt.colorbar(contour, label=metric.upper())
    
    plt.scatter(
        best_threshold,
        best_rf_weight,
        color='red', 
        marker='*', 
        s=200, 
        label=f'Best: RF={best_rf_weight:.2f}, Thresh={best_threshold:.2f}'
    )
    
    plt.title(f'Ensemble {metric.upper()} Score Contour')
    plt.xlabel('Threshold')
    plt.ylabel('Random Forest Weight')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        contour_path = save_path.replace('.png', '_contour.png')
        plt.savefig(contour_path, dpi=300, bbox_inches='tight')
        print(f"Saved contour visualization to {contour_path}")
    
    plt.show()


def save_tuning_results(results_df: pd.DataFrame, 
                       best_params: Dict, 
                       file_path: str,
                       model_name: str = 'model',
                       append_timestamp: bool = True):
    """Saves hyperparameter tuning results to file."""
    
    import time
    
    # create directory if not exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # add timestamp if needed
    if append_timestamp:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base, ext = os.path.splitext(file_path)
        file_path = f"{base}_{model_name}_{timestamp}{ext}"
    
    # save results
    results = {
        'best_params': best_params,
        'results_df': results_df
    }
    
    joblib.dump(results, file_path)
    print(f"Saved tuning results to {file_path}")
    
    return file_path