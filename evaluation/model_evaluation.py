from typing import Dict, Any, Optional
import numpy as np
from .performance_metrics import calculate_model_metrics
from .fairness_analysis import evaluate_model_fairness, generate_fairness_report
from ..utils.model_utils import save_model_artifacts

def evaluate_model(
    model: Any,
    test_data: Dict,
    protected_attributes: Optional[Dict] = None,
    fairness_thresholds: Optional[Dict] = None,
    output_paths: Optional[Dict] = None,
    model_name: str = "model"
) -> Dict:
    """Consolidated model evaluation function."""
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(test_data['X_test'])
        if hasattr(model, 'threshold_'):
            y_pred = (y_prob >= model.threshold_).astype(int)
        else:
            y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(test_data['X_test'])
        y_prob = None
    
    # Calculate performance metrics
    metrics = calculate_model_metrics(
        test_data['y_test'],
        y_pred,
        y_prob if y_prob is not None else y_pred
    )
    
    # Evaluate fairness if protected attributes provided
    if protected_attributes is not None:
        fairness_results = evaluate_model_fairness(
            test_data['y_test'],
            y_pred,
            y_prob if y_prob is not None else y_pred,
            protected_attributes,
            fairness_thresholds
        )
        metrics['fairness'] = fairness_results
        
        if output_paths:
            fairness_report = generate_fairness_report(
                fairness_results,
                fairness_thresholds,
                save_path=os.path.join(output_paths['report_dir'], f'{model_name}_fairness_report.md')
            )
            metrics['fairness_report'] = fairness_report
    
    # Save artifacts if paths provided
    if output_paths:
        save_model_artifacts(
            model,
            metrics,
            model.get_params() if hasattr(model, 'get_params') else {},
            output_paths['model_dir'],
            model_name
        )
    
    return metrics
