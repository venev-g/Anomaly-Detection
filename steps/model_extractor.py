import logging
from typing import Annotated, Dict, Any, Tuple
import xgboost as xgb
from zenml import step

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def extract_binary_model(
    binary_model_and_info: Tuple[xgb.Booster, Dict[str, Any]]
) -> Annotated[xgb.Booster, "binary_model"]:
    """
    Extract the trained model from the model and info tuple.
    
    Args:
        binary_model_and_info: Tuple containing (model, training_info)
        
    Returns:
        xgb.Booster: The trained XGBoost model
    """
    try:
        logging.info("Extracting binary model from model_and_info tuple")
        
        # Extract the model (first element of the tuple)
        model, training_info = binary_model_and_info
        
        logging.info(f"Successfully extracted binary model")
        logging.info(f"Model type: {training_info.get('model_type', 'unknown')}")
        
        return model
        
    except Exception as e:
        logging.error(f"Error extracting binary model: {str(e)}")
        raise


@step
def extract_multiclass_model(
    multiclass_model_and_info: Tuple[xgb.Booster, Dict[str, Any]]
) -> Annotated[xgb.Booster, "multiclass_model"]:
    """
    Extract the trained model from the model and info tuple.
    
    Args:
        multiclass_model_and_info: Tuple containing (model, training_info)
        
    Returns:
        xgb.Booster: The trained XGBoost model
    """
    try:
        logging.info("Extracting multiclass model from model_and_info tuple")
        
        # Extract the model (first element of the tuple)
        model, training_info = multiclass_model_and_info
        
        logging.info(f"Successfully extracted multiclass model")
        logging.info(f"Model type: {training_info.get('model_type', 'unknown')}")
        logging.info(f"Number of classes: {training_info.get('num_classes', 'unknown')}")
        
        return model
        
    except Exception as e:
        logging.error(f"Error extracting multiclass model: {str(e)}")
        raise