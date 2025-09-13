import logging
from typing import Annotated, Dict, Any, Tuple
import xgboost as xgb
from zenml import step
from src.model_evaluator import ModelEvaluatorFactory

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def binary_model_evaluation_step(
    binary_model_and_info: Tuple[xgb.Booster, Dict[str, Any]],
    preprocessed_data: Dict[str, Any],
    config_path: str = "config.yaml"
) -> Annotated[Dict[str, Any], "binary_evaluation_results"]:
    """
    Model evaluation step for binary anomaly detection.
    
    Args:
        binary_model_and_info (Tuple[xgb.Booster, Dict[str, Any]]): Trained binary model and info
        preprocessed_data (Dict[str, Any]): Preprocessed data dictionary
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Binary model evaluation results
    """
    try:
        logging.info("Starting binary model evaluation")
        
        # Unpack model and info
        model, training_info = binary_model_and_info
        
        # Extract binary classification test data
        binary_data = preprocessed_data['binary_classification']
        X_test = binary_data['X_test']
        y_test = binary_data['y_test']
        
        # Create model evaluator
        model_evaluator = ModelEvaluatorFactory.get_model_evaluator('binary', config_path)
        
        # Evaluate the model
        evaluation_results = model_evaluator.evaluate_model(model, X_test, y_test)
        
        # Add training info to results
        evaluation_results['training_info'] = training_info
        
        logging.info("Binary model evaluation completed successfully")
        logging.info(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        logging.info(f"AUC: {evaluation_results['auc']:.4f}")
        logging.info(f"Precision: {evaluation_results['precision']:.4f}")
        logging.info(f"Recall: {evaluation_results['recall']:.4f}")
        
        return evaluation_results
        
    except Exception as e:
        logging.error(f"Error in binary model evaluation step: {str(e)}")
        raise


@step
def multiclass_model_evaluation_step(
    multiclass_model_and_info: Tuple[xgb.Booster, Dict[str, Any]],
    preprocessed_data: Dict[str, Any],
    config_path: str = "config.yaml"
) -> Annotated[Dict[str, Any], "multiclass_evaluation_results"]:
    """
    Model evaluation step for multiclass attack type classification.
    
    Args:
        multiclass_model_and_info (Tuple[xgb.Booster, Dict[str, Any]]): Trained multiclass model and info
        preprocessed_data (Dict[str, Any]): Preprocessed data dictionary
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Multiclass model evaluation results
    """
    try:
        logging.info("Starting multiclass model evaluation")
        
        # Unpack model and info
        model, training_info = multiclass_model_and_info
        
        # Extract multiclass classification test data
        multiclass_data = preprocessed_data['multiclass_classification']
        X_test = multiclass_data['X_test']
        y_test = multiclass_data['y_test']
        
        # Get label encoder
        label_encoder = preprocessed_data['label_encoder']
        
        # Create model evaluator
        model_evaluator = ModelEvaluatorFactory.get_model_evaluator('multiclass', config_path)
        
        # Evaluate the model
        evaluation_results = model_evaluator.evaluate_model(
            model, X_test, y_test, label_encoder=label_encoder
        )
        
        # Add training info to results
        evaluation_results['training_info'] = training_info
        
        logging.info("Multiclass model evaluation completed successfully")
        logging.info(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        logging.info(f"Number of classes: {len(evaluation_results['class_names'])}")
        
        return evaluation_results
        
    except Exception as e:
        logging.error(f"Error in multiclass model evaluation step: {str(e)}")
        raise