import logging
from typing import Annotated, Tuple, Dict, Any
import xgboost as xgb
from zenml import step
from zenml.client import Client

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def latest_model_loader() -> Annotated[Tuple[xgb.Booster, Dict[str, Any]], "latest_model"]:
    """
    Load the latest trained binary model from ZenML artifacts.
    
    Returns:
        Tuple[xgb.Booster, Dict[str, Any]]: Latest trained model and metadata
    """
    try:
        logging.info("Loading latest trained binary model from artifacts")
        
        # Get ZenML client
        client = Client()
        
        # Find the latest binary_model_and_info artifact
        artifacts = client.list_artifacts(name="binary_model_and_info")
        
        if not artifacts:
            raise ValueError("No binary_model_and_info artifacts found")
        
        # Get the most recent artifact (they're sorted by date descending)
        latest_artifact = artifacts[0]
        logging.info(f"Using artifact: {latest_artifact.id}")
        
        # Load the artifact data
        artifact_version = client.get_artifact_version(latest_artifact.id)
        model_and_info = artifact_version.load()
        
        if isinstance(model_and_info, tuple) and len(model_and_info) == 2:
            model, info = model_and_info
            logging.info(f"Successfully loaded model: {info.get('model_type', 'unknown')}")
            return model, info
        else:
            raise ValueError(f"Unexpected artifact format: {type(model_and_info)}")
            
    except Exception as e:
        logging.error(f"Error loading latest model: {str(e)}")
        raise


@step  
def model_to_mlflow_format(
    model_and_info: Tuple[xgb.Booster, Dict[str, Any]]
) -> Annotated[xgb.Booster, "mlflow_model"]:
    """
    Extract just the model from the tuple for MLflow deployment.
    
    Args:
        model_and_info: Tuple containing (model, info) 
        
    Returns:
        xgb.Booster: The XGBoost model ready for MLflow deployment
    """
    try:
        logging.info("Extracting model for MLflow deployment")
        
        model, info = model_and_info
        
        logging.info(f"Preparing model for deployment: {info.get('model_type', 'unknown')}")
        logging.info(f"Model features: {info.get('num_features', 'unknown')}")
        
        return model
        
    except Exception as e:
        logging.error(f"Error preparing model for MLflow: {str(e)}")
        raise