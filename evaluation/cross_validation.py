import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

from config import EVALUATION, FAIRNESS, RANDOM_SEED
from evaluation.performance_metrics import calculate_model_metrics
from evaluation.fairness_metrics import calculate_group_metrics, calculate_fairness_metrics

# Set up the logger
logger = logging.getLogger('edupredict2')

def perform_cross_validation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    protected_attributes: Optional[List[str]] = None,
    n_folds: Optional[int] = None,
    random_state: Optional[int] = None,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive cross-validation including fairness evaluation.
    
    Args:
        model: Model to evaluate
        X: Feature DataFrame
        y: Target Series
        protected_attributes: List of protected attribute columns
        n_folds: Number of CV folds (default from config)
        random_state: Random seed (default from config)
        output_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        Dictionary containing CV results
    """
    if logger is None:
        logger = logging.getLogger('edupredict2')
        
    logger.info("Starting cross-validation evaluation")
    
    # Use defaults from config if not specified
    if n_folds is None:
        n_folds = EVALUATION.get('cv_folds', 5)
    
    if random_state is None:
        random_state = RANDOM_SEED
        
    if protected_attributes is None:
        protected_attributes = list(FAIRNESS.get('protected_attributes', {}).keys())
    
    # Validate protected attributes exist in data
    if protected_attributes:
        missing_attrs = [attr for attr in protected_attributes if attr not in X.columns]
        if missing_attrs:
            logger.warning(f"Protected attributes {missing_attrs} not found in data. They will be ignored.")
            protected_attributes = [attr for attr in protected_attributes if attr not in missing_attrs]
        
        if not protected_attributes:
            logger.warning("No valid protected attributes found. Fairness evaluation will be skipped.")
    
    # Perform stratified CV
    cv_results = perform_stratified_cv(
        model=model,
        X=X,
        y=y,
        n_folds=n_folds,
        stratify_cols=protected_attributes,
        random_state=random_state
    )
    
    # Evaluate CV results
    evaluation = evaluate_cv_results(cv_results)
    
    # Save results if output directory provided
    if output_dir:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save overall metrics
            metrics_path = output_dir / 'cv_metrics.json'
            pd.DataFrame([evaluation['summary']]).to_json(metrics_path, orient='records')
            
            # Save predictions
            preds_path = output_dir / 'cv_predictions.csv'
            cv_results['predictions'].to_csv(preds_path, index=False)
            
            # Save feature importance if available
            if cv_results.get('feature_importance') is not None:
                importance_path = output_dir / 'feature_importance.csv'
                cv_results['feature_importance'].to_csv(importance_path, index=False)
            
            # Save fairness metrics if available
            if evaluation.get('fairness'):
                fairness_path = output_dir / 'fairness_metrics.json'
                pd.DataFrame([evaluation['fairness']]).to_json(fairness_path, orient='records')
                
            logger.info(f"Cross-validation results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving cross-validation results: {str(e)}")
    
    return {
        'cv_results': cv_results,
        'evaluation': evaluation
    }

def perform_stratified_cv(
    model: Any, 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_folds: int = 5, 
    stratify_cols: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs stratified cross-validation preserving demographic distributions.
    
    Args:
        model: Model object with fit/predict methods
        X: Feature DataFrame
        y: Target Series
        n_folds: Number of CV folds
        stratify_cols: Columns to stratify on
        random_state: Random seed
        
    Returns:
        Dictionary with CV results
    """
    logger.info(f"Performing {n_folds}-fold stratified cross-validation")
    
    # Use default values from config if not specified
    if n_folds is None:
        n_folds = EVALUATION.get('cv_folds', 5)
    
    if stratify_cols is None:
        stratify_cols = EVALUATION.get('stratify_cols', [])
    
    # Check if stratify columns are present in X
    missing_cols = [col for col in stratify_cols if col not in X.columns]
    if missing_cols:
        logger.warning(f"Stratify columns {missing_cols} not found in data. They will be ignored.")
        stratify_cols = [col for col in stratify_cols if col not in missing_cols]
    
    # Create stratification target
    if stratify_cols:
        logger.info(f"Stratifying on columns: {stratify_cols}")
        
        # Create composite stratification column
        stratify_values = y.astype(str)
        for col in stratify_cols:
            stratify_values = stratify_values + '_' + X[col].astype(str)
    else:
        # Just stratify on the target
        stratify_values = y
    
    # Initialize cross-validation splitter
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Initialize results
    fold_results = []
    predictions = []
    fold_indices = []
    feature_importances = []  # For models that provide feature importance
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, stratify_values)):
        logger.info(f"Training fold {fold_idx + 1}/{n_folds}")
        
        # Split data for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:,1]
            else:
                logger.warning(f"Model doesn't have predict_proba method. Using predict for probabilities.")
                y_prob = model.predict(X_test)
            
            y_pred = model.predict(X_test)
            
            # Store predictions and indices for later analysis
            fold_pred_df = pd.DataFrame({
                'fold': fold_idx,
                'index': test_idx,
                'true': y_test.values,
                'pred': y_pred,
                'prob': y_prob
            })
            predictions.append(fold_pred_df)
            fold_indices.append({
                'fold': fold_idx,
                'train_indices': train_idx,
                'test_indices': test_idx
            })
            
            # Calculate performance metrics for this fold
            fold_metrics = calculate_model_metrics(
                y_true=y_test.values,
                y_pred=y_pred,
                y_prob=y_prob,
                logger=logger
            )
            
            # Add fold number
            fold_metrics['fold'] = fold_idx
            fold_results.append(fold_metrics)
            
            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_imp = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_,
                    'fold': fold_idx
                })
                feature_importances.append(feature_imp)
            elif hasattr(model, 'coef_'):
                # For linear models
                coefs = model.coef_
                if len(coefs.shape) > 1:
                    coefs = coefs[0]  # For multi-class models
                feature_imp = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(coefs),
                    'fold': fold_idx
                })
                feature_importances.append(feature_imp)
            
        except Exception as e:
            logger.error(f"Error in fold {fold_idx}: {str(e)}")
    
    # Combine all predictions
    all_predictions = pd.concat(predictions, ignore_index=True)
    
    # Calculate overall performance
    overall_metrics = calculate_model_metrics(
        y_true=all_predictions['true'].values,
        y_pred=all_predictions['pred'].values,
        y_prob=all_predictions['prob'].values,
        logger=logger
    )
    
    # Combine feature importances if available
    combined_importances = None
    if feature_importances:
        combined_importances = pd.concat(feature_importances, ignore_index=True)
    
    # Check demographic fairness if stratify_cols contains protected attributes
    fairness_metrics = {}
    prot_attrs = set(stratify_cols).intersection(set(FAIRNESS['protected_attributes']))
    
    if prot_attrs:
        logger.info(f"Evaluating fairness on protected attributes: {prot_attrs}")
        
        for attr in prot_attrs:
            # Create mapping of indices to attribute values
            attr_values = X[attr].values
            
            # Calculate group metrics
            group_metrics = calculate_group_metrics(
                y_true=all_predictions['true'].values,
                y_pred=all_predictions['pred'].values,
                y_prob=all_predictions['prob'].values,
                group_values=attr_values[all_predictions['index']],  # Select attribute values for test indices
                logger=logger
            )
            
            # Calculate fairness metrics
            attr_fairness = calculate_fairness_metrics(
                group_metrics=group_metrics,
                logger=logger
            )
            
            fairness_metrics[attr] = attr_fairness
    
    # Compile results
    cv_results = {
        'fold_results': fold_results,
        'overall_metrics': overall_metrics,
        'predictions': all_predictions,
        'fold_indices': fold_indices,
        'feature_importance': combined_importances,
        'fairness_metrics': fairness_metrics
    }
    
    logger.info(f"Cross-validation completed with {len(fold_results)} successful folds")
    return cv_results


def evaluate_cv_results(
    cv_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyzes cross-validation performance across folds.
    
    Args:
        cv_results: CV results from perform_stratified_cv
        
    Returns:
        Dictionary with performance statistics
    """
    logger.info("Evaluating cross-validation results")
    
    # Extract fold results
    fold_results = cv_results.get('fold_results', [])
    
    if not fold_results:
        logger.warning("No fold results found to evaluate")
        return {}
    
    # Metrics to analyze
    metrics_to_analyze = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    
    # Filter to metrics that are present in all folds
    available_metrics = set(fold_results[0].keys())
    for fold_result in fold_results:
        available_metrics = available_metrics.intersection(set(fold_result.keys()))
    
    metrics_to_analyze = [m for m in metrics_to_analyze if m in available_metrics]
    
    # Initialize results
    evaluation = {
        'metrics': {},
        'summary': {}
    }
    
    # Calculate statistics for each metric
    for metric in metrics_to_analyze:
        # Extract values across folds
        values = [fold[metric] for fold in fold_results if metric in fold]
        
        if values:
            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            evaluation['metrics'][metric] = {
                'values': values,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'range': max_val - min_val,
                'cv': std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
            }
    
    # Calculate overall stability score (based on coefficient of variation)
    if evaluation['metrics']:
        cv_values = [stats['cv'] for stats in evaluation['metrics'].values()]
        stability_score = 1 - np.mean(cv_values)  # Higher is more stable
        
        evaluation['summary']['stability_score'] = stability_score
        evaluation['summary']['avg_cv'] = np.mean(cv_values)
    
    # Add overall metrics from CV
    if 'overall_metrics' in cv_results:
        evaluation['summary']['overall_metrics'] = cv_results['overall_metrics']
    
    # Add fairness evaluation if available
    if 'fairness_metrics' in cv_results and cv_results['fairness_metrics']:
        evaluation['fairness'] = cv_results['fairness_metrics']
        
        # Check if any fairness metrics violate thresholds
        fairness_violations = {}
        for attr, metrics in cv_results['fairness_metrics'].items():
            attr_violations = []
            
            for metric, value in metrics.items():
                # Skip special keys like group names
                if metric in ['max_positive_rate_group', 'min_positive_rate_group', 'max_tpr_group', 'min_tpr_group', 'min_impact_ratio_groups']:
                    continue
                    
                # Check against thresholds
                if metric in FAIRNESS.get('thresholds', {}):
                    threshold = FAIRNESS['thresholds'][metric]
                    
                    # For disparate impact ratio, higher is better
                    if metric == 'disparate_impact_ratio':
                        if value < threshold:
                            attr_violations.append({
                                'metric': metric,
                                'value': value,
                                'threshold': threshold,
                                'violation': True
                            })
                    # For other metrics, lower is better
                    elif value > threshold:
                        attr_violations.append({
                            'metric': metric,
                            'value': value,
                            'threshold': threshold,
                            'violation': True
                        })
            
            if attr_violations:
                fairness_violations[attr] = attr_violations
        
        evaluation['fairness_violations'] = fairness_violations
        evaluation['has_fairness_violations'] = len(fairness_violations) > 0
    
    # Add feature importance analysis if available
    if 'feature_importance' in cv_results and cv_results['feature_importance'] is not None:
        feature_imp_df = cv_results['feature_importance']
        
        # Calculate mean importance for each feature
        feature_importance = feature_imp_df.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
        feature_importance = feature_importance.sort_values('mean', ascending=False)
        
        # Calculate stability of feature rankings
        feature_ranks = []
        for fold in feature_imp_df['fold'].unique():
            fold_importance = feature_imp_df[feature_imp_df['fold'] == fold]
            fold_importance = fold_importance.sort_values('importance', ascending=False)
            fold_importance['rank'] = np.arange(1, len(fold_importance) + 1)
            feature_ranks.append(fold_importance[['feature', 'rank']])
        
        feature_ranks_df = pd.concat(feature_ranks)
        rank_stability = feature_ranks_df.groupby('feature')['rank'].std().reset_index()
        
        evaluation['feature_importance'] = {
            'importance': feature_importance.to_dict('records'),
            'rank_stability': rank_stability.to_dict('records')
        }
    
    logger.info("Cross-validation evaluation complete")
    return evaluation


def compare_models_with_cv(
    models: Dict[str, Any], 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_folds: int = 5, 
    stratify_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compares multiple models using cross-validation.
    
    Args:
        models: Dictionary mapping names to model objects
        X: Feature DataFrame
        y: Target Series
        n_folds: Number of CV folds
        stratify_cols: Columns to stratify on
        
    Returns:
        DataFrame comparing model performance
    """
    logger.info(f"Comparing {len(models)} models using cross-validation")
    
    # Use default values from config if not specified
    if n_folds is None:
        n_folds = EVALUATION.get('cv_folds', 5)
    
    if stratify_cols is None:
        stratify_cols = EVALUATION.get('stratify_cols', [])
    
    # Results for each model
    model_results = {}
    model_metrics = []
    
    # Run CV for each model
    for model_name, model in models.items():
        logger.info(f"Running CV for model '{model_name}'")
        
        try:
            # Perform cross-validation
            cv_results = perform_stratified_cv(
                model=model,
                X=X,
                y=y,
                n_folds=n_folds,
                stratify_cols=stratify_cols,
                random_state=RANDOM_SEED
            )
            
            # Evaluate results
            eval_results = evaluate_cv_results(cv_results)
            
            # Save complete results
            model_results[model_name] = {
                'cv_results': cv_results,
                'evaluation': eval_results
            }
            
            # Extract metrics for comparison
            metrics_row = {'model': model_name}
            
            # Add performance metrics
            if 'metrics' in eval_results:
                for metric, stats in eval_results['metrics'].items():
                    metrics_row[f'{metric}_mean'] = stats['mean']
                    metrics_row[f'{metric}_std'] = stats['std']
            
            # Add stability score
            if 'summary' in eval_results and 'stability_score' in eval_results['summary']:
                metrics_row['stability_score'] = eval_results['summary']['stability_score']
            
            # Add fairness information
            if 'has_fairness_violations' in eval_results:
                metrics_row['has_fairness_violations'] = eval_results['has_fairness_violations']
            
            model_metrics.append(metrics_row)
            
        except Exception as e:
            logger.error(f"Error comparing model '{model_name}': {str(e)}")
    
    # Convert to DataFrame
    if model_metrics:
        comparison_df = pd.DataFrame(model_metrics)
        
        # Sort by F1 score if available, otherwise by accuracy
        if 'f1_mean' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_mean', ascending=False)
        elif 'accuracy_mean' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('accuracy_mean', ascending=False)
        
        logger.info(f"Model comparison completed for {len(comparison_df)} models")
        return comparison_df
    else:
        logger.warning("No model comparison data available")
        return pd.DataFrame(columns=['model'])