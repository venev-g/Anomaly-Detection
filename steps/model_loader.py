import logging
from typing import Annotated, Tuple, Dict, Any
import xgboost as xgb
from zenml import step
from zenml import get_step_context

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def model_loader(
    model_name: str = "anomaly_detector",
    model_version: str = None
) -> Annotated[Tuple[xgb.Booster, Dict[str, Any]], "loaded_model"]:
    """
    Load a trained model from the model registry.
    
    Args:
        model_name (str): Name of the model to load
        model_version (str): Version of the model to load (None for latest)
        
    Returns:
        Tuple[xgb.Booster, Dict[str, Any]]: Loaded model and metadata
    """
    try:
        logging.info(f"Loading model: {model_name}, version: {model_version}")
        
        # Get the step context to access the model registry
        step_context = get_step_context()
        
        # Load the model from the registry
        if model_version:
            model = step_context.model_registry.load_artifact(f"{model_name}:{model_version}")
        else:
            model = step_context.model_registry.load_artifact(f"{model_name}:latest")
        
        # Extract model and metadata
        if isinstance(model, tuple):
            loaded_model, metadata = model
        else:
            loaded_model = model
            metadata = {"model_name": model_name, "version": model_version}
        
        logging.info(f"Successfully loaded model: {model_name}")
        logging.info(f"Model metadata: {metadata}")
        
        return loaded_model, metadata
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        # Fallback: return None model with error info
        error_metadata = {
            "error": str(e),
            "model_name": model_name,
            "version": model_version,
            "status": "failed_to_load"
        }
        return None, error_metadata