import logging
from typing import Annotated, Dict, Any, Tuple
import xgboost as xgb
from zenml import step
from src.model_builder import ModelBuilderFactory

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def binary_model_building_step(
    preprocessed_data: Dict[str, Any],
    config_path: str = "config.yaml"
) -> Annotated[Tuple[xgb.Booster, Dict[str, Any]], "binary_model_and_info"]:
    """
    Model building step for binary anomaly detection.
    
    Args:
        preprocessed_data (Dict[str, Any]): Preprocessed data dictionary
        config_path (str): Path to configuration file
        
    Returns:
        Tuple[xgb.Booster, Dict[str, Any]]: Trained binary model and training info
    """
    try:
        logging.info("Starting binary model building")
        
        # Extract binary classification data
        binary_data = preprocessed_data['binary_classification']
        X_train = binary_data['X_train']
        y_train = binary_data['y_train']
        X_test = binary_data['X_test']
        y_test = binary_data['y_test']
        
        # Create model builder
        model_builder = ModelBuilderFactory.get_model_builder('binary', config_path)
        
        # Build and train the model
        model, training_info = model_builder.build_model(X_train, y_train, X_test, y_test)
        
        logging.info("Binary model building completed successfully")
        logging.info(f"Model type: {training_info['model_type']}")
        logging.info(f"Number of features: {training_info['num_features']}")
        
        return model, training_info
        
    except Exception as e:
        logging.error(f"Error in binary model building step: {str(e)}")
        raise


@step
def multiclass_model_building_step(
    preprocessed_data: Dict[str, Any],
    config_path: str = "config.yaml"
) -> Annotated[Tuple[xgb.Booster, Dict[str, Any]], "multiclass_model_and_info"]:
    """
    Model building step for multiclass attack type classification.
    
    Args:
        preprocessed_data (Dict[str, Any]): Preprocessed data dictionary
        config_path (str): Path to configuration file
        
    Returns:
        Tuple[xgb.Booster, Dict[str, Any]]: Trained multiclass model and training info
    """
    try:
        logging.info("Starting multiclass model building")
        
        # Extract multiclass classification data
        multiclass_data = preprocessed_data['multiclass_classification']
        X_train = multiclass_data['X_train']
        y_train = multiclass_data['y_train']
        X_test = multiclass_data['X_test']
        y_test = multiclass_data['y_test']
        
        # Create model builder
        model_builder = ModelBuilderFactory.get_model_builder('multiclass', config_path)
        
        # Build and train the model
        model, training_info = model_builder.build_model(X_train, y_train, X_test, y_test)
        
        logging.info("Multiclass model building completed successfully")
        logging.info(f"Model type: {training_info['model_type']}")
        logging.info(f"Number of classes: {training_info['num_classes']}")
        logging.info(f"Number of features: {training_info['num_features']}")
        
        return model, training_info
        
    except Exception as e:
        logging.error(f"Error in multiclass model building step: {str(e)}")
        raise