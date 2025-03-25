# re-export functions from performance_metrics.py
from .performance_metrics import (
    analyze_feature_importance,
    analyze_feature_correlations,
    calculate_model_metrics,
    calculate_fairness_metrics,
    plot_roc_curves,
    plot_fairness_metrics
)