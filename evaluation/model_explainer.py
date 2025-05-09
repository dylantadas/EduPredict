import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# Import SHAP for explainable AI features
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config import DIRS, PROTECTED_ATTRIBUTES

# Set up the logger
logger = logging.getLogger('edupredict2')

def generate_shap_explanations(
    model: Any, 
    X: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generates SHAP explanations for model predictions.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        
    Returns:
        Dictionary with SHAP values and explanations
    """
    logger.info("Generating SHAP explanations")
    
    # Check if SHAP is available
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Install with 'pip install shap'")
        return {
            'error': "SHAP library not available",
            'shap_values': None,
            'expected_value': None
        }
    
    # Initialize results
    explanations = {}
    
    try:
        # Determine model type to use appropriate SHAP explainer
        if hasattr(model, 'predict_proba'):
            # Try different explainers based on the model type
            
            # For tree-based models (Random Forest, Gradient Boosting, etc.)
            if hasattr(model, 'estimators_') or str(type(model)).endswith("RandomForestClassifier'>") or str(type(model)).endswith("GradientBoostingClassifier'>"):
                logger.info("Using TreeExplainer for SHAP values")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                explanations['explainer_type'] = 'TreeExplainer'
            
            # For neural network models (TF/Keras, PyTorch)
            elif 'keras' in str(type(model)).lower() or 'tensorflow' in str(type(model)).lower() or 'torch' in str(type(model)).lower():
                logger.info("Using DeepExplainer for SHAP values")
                # Create background dataset (sample of training data)
                background = shap.kmeans(X, 100)  # Use k-means to get representative samples
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(X)
                explanations['explainer_type'] = 'DeepExplainer'
            
            # For any other model, use Kernel explainer (model-agnostic)
            else:
                logger.info("Using KernelExplainer for SHAP values")
                # Function to predict probabilities (focusing on positive class)
                predict_fn = lambda x: model.predict_proba(x)[:,1]
                # Create background dataset (sample of training data)
                background = shap.kmeans(X, 100)  # Use k-means to get representative samples
                explainer = shap.KernelExplainer(predict_fn, background)
                # Calculate SHAP values (this can be computationally expensive)
                # Use a sample if dataset is large
                max_samples = 500
                if len(X) > max_samples:
                    logger.info(f"Using {max_samples} samples for SHAP calculations")
                    sample_indices = np.random.choice(len(X), max_samples, replace=False)
                    shap_values = explainer.shap_values(X.iloc[sample_indices])
                    explanations['sample_indices'] = sample_indices.tolist()
                else:
                    shap_values = explainer.shap_values(X)
                explanations['explainer_type'] = 'KernelExplainer'
        else:
            logger.warning("Model doesn't have predict_proba method, using basic explainer")
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            explanations['explainer_type'] = 'Explainer'
        
        # Store SHAP values
        # Handle different return types from different explainers
        if isinstance(shap_values, list):
            # For binary classification, typically returns a list with values for each class
            explanations['shap_values'] = shap_values[1]  # Values for positive class
            explanations['shap_values_negative'] = shap_values[0]  # Values for negative class
        else:
            explanations['shap_values'] = shap_values
        
        # Store expected value
        if hasattr(explainer, 'expected_value'):
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                explanations['expected_value'] = expected_value[1]  # For positive class
            else:
                explanations['expected_value'] = expected_value
        
        # Store feature names
        explanations['feature_names'] = X.columns.tolist()
        
        # Calculate summary statistics for SHAP values
        if isinstance(explanations['shap_values'], np.ndarray):
            # Calculate mean absolute SHAP value for each feature
            mean_abs_shap = np.mean(np.abs(explanations['shap_values']), axis=0)
            explanations['feature_importance'] = {
                'names': X.columns.tolist(),
                'values': mean_abs_shap.tolist()
            }
            
            # Calculate mean SHAP value for each feature (directional)
            mean_shap = np.mean(explanations['shap_values'], axis=0)
            explanations['feature_impact'] = {
                'names': X.columns.tolist(),
                'values': mean_shap.tolist()
            }
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {str(e)}")
        explanations['error'] = str(e)
    
    logger.info("SHAP explanation generation complete")
    return explanations


def create_feature_impact_visualizations(
    explanations: Dict[str, Any], 
    output_path: str
) -> List[str]:
    """
    Creates visualizations of feature impacts.
    
    Args:
        explanations: SHAP explanations
        output_path: Path to save visualizations
        
    Returns:
        List of paths to saved visualizations
    """
    logger.info("Creating feature impact visualizations")
    
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Install with 'pip install shap'")
        return []
    
    # Check for errors in explanations
    if 'error' in explanations:
        logger.error(f"Cannot create visualizations due to error: {explanations['error']}")
        return []
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    try:
        # Get shap values and feature names
        shap_values = explanations.get('shap_values')
        feature_names = explanations.get('feature_names')
        
        if shap_values is None or feature_names is None:
            logger.error("Missing required data in explanations")
            return []
        
        # Convert feature names to list if needed
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        
        # 1. Summary Plot - Shows feature importance and impact direction
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            features=feature_names,
            feature_names=feature_names,
            show=False
        )
        summary_path = f"{output_path}_summary.png"
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(summary_path)
        logger.info(f"Saved summary plot to {summary_path}")
        
        # 2. Bar Plot - Shows average impact magnitude
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            features=feature_names,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        bar_path = f"{output_path}_importance_bar.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(bar_path)
        logger.info(f"Saved bar plot to {bar_path}")
        
        # 3. Feature Dependence Plots - For top N most important features
        top_n = 5
        
        # Get indices of top N features by importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(-feature_importance)[:top_n]
        
        for i, idx in enumerate(top_indices):
            feature_name = feature_names[idx]
            plt.figure(figsize=(10, 6))
            
            # Create dependence plot
            shap.dependence_plot(
                idx,
                shap_values,
                features=feature_names,
                feature_names=feature_names,
                show=False
            )
            
            dependence_path = f"{output_path}_dependence_{i}_{feature_name.replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_paths.append(dependence_path)
            logger.info(f"Saved dependence plot for {feature_name} to {dependence_path}")
        
        # 4. Create SHAP force plot for sample predictions
        if 'expected_value' in explanations:
            sample_size = min(5, len(shap_values))  # Limit to 5 samples
            indices = np.random.choice(len(shap_values), sample_size, replace=False)
            
            # Create force plot
            force_plot = shap.force_plot(
                explanations['expected_value'],
                shap_values[indices],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            force_path = f"{output_path}_force_plot.png"
            plt.savefig(force_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_paths.append(force_path)
            logger.info(f"Saved force plot to {force_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
    
    logger.info(f"Created {len(saved_paths)} feature impact visualizations")
    return saved_paths


def explain_individual_prediction(
    model: Any, 
    student_data: pd.Series
) -> Dict[str, Any]:
    """
    Provides explanation for individual student prediction.
    
    Args:
        model: Trained model
        student_data: Series with student features
        
    Returns:
        Dictionary with prediction explanation
    """
    logger.info("Generating individual prediction explanation")
    
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Install with 'pip install shap'")
        return {
            'error': "SHAP library not available",
            'prediction': None
        }
    
    try:
        # Convert student data to DataFrame if it's a Series
        if isinstance(student_data, pd.Series):
            X = pd.DataFrame([student_data])
        else:
            X = pd.DataFrame(student_data)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            prediction_prob = model.predict_proba(X)[0, 1]  # Probability of positive class
            prediction_class = int(prediction_prob >= 0.5)  # Binary prediction
        else:
            prediction_class = model.predict(X)[0]
            prediction_prob = None
        
        # Generate SHAP explanation
        if hasattr(model, 'estimators_') or str(type(model)).endswith("RandomForestClassifier'>"):
            # For tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            expected_value = explainer.expected_value
            
            if isinstance(shap_values, list):
                # Binary classification
                shap_values_explain = shap_values[1][0]  # For positive class
                expected_value = expected_value[1] if isinstance(expected_value, list) else expected_value
            else:
                shap_values_explain = shap_values[0]
        else:
            # For other models
            predict_fn = lambda x: model.predict_proba(x)[:,1]
            explainer = shap.Explainer(model)
            explanation = explainer(X)
            shap_values_explain = explanation.values[0]
            expected_value = explanation.base_values[0] if hasattr(explanation, 'base_values') else 0
        
        # Create result dictionary
        result = {
            'prediction': {
                'class': int(prediction_class),
                'probability': float(prediction_prob) if prediction_prob is not None else None
            },
            'base_value': float(expected_value),
            'features': X.columns.tolist(),
            'feature_values': X.iloc[0].tolist(),
            'shap_values': shap_values_explain.tolist() if isinstance(shap_values_explain, np.ndarray) else shap_values_explain
        }
        
        # Sort features by impact
        feature_impacts = []
        for i, (feature, value, shap_value) in enumerate(zip(result['features'], result['feature_values'], result['shap_values'])):
            feature_impacts.append({
                'feature': feature,
                'value': value,
                'impact': shap_value,
                'abs_impact': abs(shap_value)
            })
        
        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)
        
        # Separate features into positive and negative impacts
        positive_impacts = []
        negative_impacts = []
        
        for impact in feature_impacts:
            if impact['impact'] > 0:
                positive_impacts.append(impact)
            else:
                negative_impacts.append(impact)
        
        result['positive_impacts'] = positive_impacts
        result['negative_impacts'] = negative_impacts
        
        # Add interpretation
        result['interpretation'] = {
            'summary': f"The model {'predicts' if prediction_class == 1 else 'does not predict'} the student will pass.",
            'confidence': f"Model confidence: {prediction_prob:.2%}" if prediction_prob is not None else "Unknown confidence",
            'top_factors': [impact['feature'] for impact in feature_impacts[:3]],
            'base_prediction': f"Base prediction value: {expected_value:.2f}"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error explaining individual prediction: {str(e)}")
        return {
            'error': str(e),
            'prediction': None
        }


def generate_global_explanations(
    model: Any, 
    X: pd.DataFrame, 
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Generates global model explanations.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        feature_names: List of feature names
        
    Returns:
        Dictionary with global explanations
    """
    logger.info("Generating global model explanations")
    
    # Initialize results
    results = {
        'feature_names': feature_names,
        'feature_importance': {},
        'insights': []
    }
    
    try:
        # Check if model provides feature importance directly
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            results['feature_importance'] = {
                'source': 'model.feature_importances_',
                'values': importances.tolist() if isinstance(importances, np.ndarray) else importances
            }
            
            # Get top features
            if len(feature_names) == len(importances):
                top_indices = np.argsort(-importances)[:10]  # Top 10
                results['top_features'] = [{
                    'name': feature_names[i],
                    'importance': float(importances[i])
                } for i in top_indices]
        
        elif hasattr(model, 'coef_'):
            # For linear models
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0]  # For multi-class
            
            results['feature_importance'] = {
                'source': 'model.coef_',
                'values': np.abs(coefs).tolist() if isinstance(coefs, np.ndarray) else abs(coefs)
            }
            
            # Get top features by absolute coefficient
            if len(feature_names) == len(coefs):
                top_indices = np.argsort(-np.abs(coefs))[:10]  # Top 10
                results['top_features'] = [{
                    'name': feature_names[i],
                    'coefficient': float(coefs[i]),
                    'importance': float(abs(coefs[i]))
                } for i in top_indices]
        
        # Generate explanations using SHAP if available
        if SHAP_AVAILABLE:
            # Sample data if it's too large
            max_samples = 1000  # Maximum number of samples for SHAP
            if len(X) > max_samples:
                logger.info(f"Sampling {max_samples} records for SHAP calculation")
                sample_indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X.iloc[sample_indices]
            else:
                X_sample = X
            
            # Generate SHAP explanations
            shap_explanations = generate_shap_explanations(model, X_sample)
            
            if 'error' not in shap_explanations:
                results['shap'] = {
                    'available': True,
                    'feature_importance': shap_explanations.get('feature_importance'),
                    'explainer_type': shap_explanations.get('explainer_type')
                }
                
                # Add feature importance from SHAP if not already present
                if 'feature_importance' not in results:
                    fi = shap_explanations.get('feature_importance')
                    if fi:
                        results['feature_importance'] = {
                            'source': 'shap',
                            'values': fi.get('values')
                        }
            else:
                results['shap'] = {
                    'available': False,
                    'error': shap_explanations.get('error')
                }
        else:
            results['shap'] = {
                'available': False,
                'error': 'SHAP library not available'
            }
        
        # Generate insights from feature importance
        if 'top_features' in results:
            results['insights'].append({
                'type': 'top_features',
                'message': f"The most important predictors of student success are: {', '.join([f['name'] for f in results['top_features'][:3]])}"
            })
        
        # Look for protected attributes among important features
        if 'top_features' in results:
            protected_attrs_in_top = []
            for attr_name in PROTECTED_ATTRIBUTES.keys():
                for feature in results['top_features']:
                    if attr_name in feature['name'].lower():
                        protected_attrs_in_top.append(attr_name)
                        break
            
            if protected_attrs_in_top:
                results['insights'].append({
                    'type': 'protected_attributes',
                    'message': f"The model relies on protected attributes: {', '.join(protected_attrs_in_top)}. Consider examining for potential bias."
                })
        
    except Exception as e:
        logger.error(f"Error generating global explanations: {str(e)}")
        results['error'] = str(e)
    
    logger.info("Global explanation generation complete")
    return results