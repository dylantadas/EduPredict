import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
from sklearn.utils import resample
from config import FAIRNESS, PROTECTED_ATTRIBUTES, BIAS_MITIGATION, DIRS

logger = logging.getLogger('edupredict')

def apply_reweighting(
    features: pd.DataFrame,
    labels: np.ndarray,
    protected_attr: str,
    params: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Applies instance reweighting to mitigate bias.
    
    Args:
        features: Input features
        labels: Target labels
        protected_attr: Protected attribute to balance for
        params: Optional parameters for reweighting
        
    Returns:
        Tuple of instance weights and reweighting metadata
    """
    try:
        params = params or BIAS_MITIGATION['reweight_options']
        weights = np.ones(len(labels))
        metadata = {}
        
        # Get group information
        groups = features[protected_attr].unique()
        group_sizes = features[protected_attr].value_counts()
        
        # Calculate target proportions
        total_samples = len(labels)
        target_size = total_samples / len(groups)  # Equal representation
        
        # Calculate weights for each group
        for group in groups:
            group_mask = features[protected_attr] == group
            current_size = group_sizes[group]
            
            # Calculate weight multiplier
            weight_multiplier = target_size / current_size
            
            # Clip weights if specified
            if params.get('weight_clipping'):
                weight_multiplier = min(
                    weight_multiplier,
                    params['weight_clipping']
                )
            
            # Apply weights
            weights[group_mask] *= weight_multiplier
            
            metadata[f"{group}_weight"] = float(weight_multiplier)
        
        # Normalize weights
        weights = weights / (weights.sum() / total_samples)
        
        return weights, metadata
        
    except Exception as e:
        logger.error(f"Error applying reweighting: {str(e)}")
        raise

def apply_sampling(
    features: pd.DataFrame,
    labels: np.ndarray,
    protected_attr: str,
    method: str = 'oversample',
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Applies sampling-based bias mitigation.
    
    Args:
        features: Input features
        labels: Target labels
        protected_attr: Protected attribute to balance for
        method: 'oversample' or 'undersample'
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of resampled features, labels, and metadata
    """
    try:
        metadata = {'original_sizes': {}, 'final_sizes': {}}
        
        # Get group information
        groups = features[protected_attr].unique()
        group_sizes = features[protected_attr].value_counts()
        
        # Record original sizes
        for group in groups:
            metadata['original_sizes'][group] = int(group_sizes[group])
        
        # Determine target size
        if method == 'oversample':
            target_size = group_sizes.max()
        else:  # undersample
            target_size = max(
                group_sizes.min(),
                BIAS_MITIGATION['min_group_size']
            )
        
        # Apply sampling
        resampled_features = []
        resampled_labels = []
        
        for group in groups:
            group_mask = features[protected_attr] == group
            group_features = features[group_mask]
            group_labels = labels[group_mask]
            
            if len(group_features) == target_size:
                resampled_features.append(group_features)
                resampled_labels.append(group_labels)
                continue
            
            # Perform resampling
            resampled_group_features, resampled_group_labels = resample(
                group_features,
                group_labels,
                n_samples=int(target_size),
                random_state=random_state,
                replace=(method == 'oversample')
            )
            
            resampled_features.append(resampled_group_features)
            resampled_labels.append(resampled_group_labels)
            
            metadata['final_sizes'][group] = int(target_size)
        
        # Combine resampled data
        features_resampled = pd.concat(resampled_features, axis=0)
        labels_resampled = np.concatenate(resampled_labels)
        
        metadata['method'] = method
        metadata['total_samples'] = {
            'original': len(labels),
            'resampled': len(labels_resampled)
        }
        
        return features_resampled, labels_resampled, metadata
        
    except Exception as e:
        logger.error(f"Error applying sampling: {str(e)}")
        raise

def mitigate_bias(
    features: pd.DataFrame,
    labels: np.ndarray,
    fairness_metrics: Dict[str, Dict[str, Any]],
    method: Optional[str] = None
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Applies appropriate bias mitigation techniques based on detected issues.
    
    Args:
        features: Input features
        labels: Target labels
        fairness_metrics: Previously calculated fairness metrics
        method: Optional override for mitigation method
        
    Returns:
        Tuple of processed features, labels, and mitigation metadata
    """
    try:
        method = method or BIAS_MITIGATION['method']
        metadata = {'mitigations_applied': []}
        
        # Check which attributes need mitigation
        attributes_to_mitigate = []
        for attr, metrics in fairness_metrics.items():
            if metrics['threshold_violations']:
                attributes_to_mitigate.append(attr)
        
        if not attributes_to_mitigate:
            logger.info("No fairness violations detected, skipping bias mitigation")
            return features, labels, metadata
        
        # Apply mitigation for each problematic attribute
        processed_features = features.copy()
        processed_labels = labels.copy()
        
        for attr in attributes_to_mitigate:
            if method == 'reweight':
                weights, weight_metadata = apply_reweighting(
                    processed_features,
                    processed_labels,
                    attr
                )
                metadata[f"{attr}_weights"] = weight_metadata
                metadata['mitigations_applied'].append({
                    'attribute': attr,
                    'method': 'reweight',
                    'metadata': weight_metadata
                })
                
            elif method in ['oversample', 'undersample']:
                proc_features, proc_labels, sampling_metadata = apply_sampling(
                    processed_features,
                    processed_labels,
                    attr,
                    method=method
                )
                processed_features = proc_features
                processed_labels = proc_labels
                metadata[f"{attr}_sampling"] = sampling_metadata
                metadata['mitigations_applied'].append({
                    'attribute': attr,
                    'method': method,
                    'metadata': sampling_metadata
                })
        
        # Export mitigation metadata
        output_path = DIRS['reports_fairness'] / 'bias_mitigation.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return processed_features, processed_labels, metadata
        
    except Exception as e:
        logger.error(f"Error in bias mitigation: {str(e)}")
        raise