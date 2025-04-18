import os
import gc
import psutil
import tensorflow as tf
from typing import Dict, Any

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def optimize_memory():
    """Optimize memory usage by clearing caches and running garbage collection."""
    gc.collect()
    tf.keras.backend.clear_session()

def setup_gpu(use_gpu: bool = True):
    """Configure GPU usage for TensorFlow."""
    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return True, f"Using GPU: {len(gpus)} device(s) available"
            except RuntimeError as e:
                return False, f"Error setting up GPU: {e}"
        return False, "No GPU available. Using CPU for training."
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return False, "Using CPU for training (GPU disabled)"

def save_model_artifacts(model: Any, 
                        metrics: Dict,
                        params: Dict,
                        output_dir: str,
                        model_name: str):
    """Save model artifacts in a consistent way."""
    # Create model subdirectory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f"{model_name}_model")
    if hasattr(model, 'save_model'):
        model.save_model(model_path)
    elif hasattr(model, 'save'):
        model.save(model_path)

    # Save metrics and parameters
    import json
    with open(os.path.join(model_dir, f"{model_name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(model_dir, f"{model_name}_params.json"), 'w') as f:
        json.dump(params, f, indent=2)
