import logging
from typing import Annotated
import pandas as pd
import numpy as np
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def predictor(
    service: MLFlowDeploymentService,
    input_data: pd.DataFrame,
) -> Annotated[pd.DataFrame, "predictions"]:
    """
    Make predictions using the deployed model service.
    
    Args:
        service (MLFlowDeploymentService): The prediction service
        input_data (pd.DataFrame): Input data for prediction
        
    Returns:
        pd.DataFrame: Predictions with input data
    """
    try:
        logging.info("Making predictions using deployed model service")
        logging.info(f"Input data shape: {input_data.shape}")
        
        # Make predictions using the service
        predictions = service.predict(input_data)
        
        if isinstance(predictions, np.ndarray):
            # Handle different prediction formats
            if predictions.ndim == 1:
                # Binary classification - single probability
                pred_df = pd.DataFrame({
                    'prediction_probability': predictions,
                    'prediction_binary': (predictions > 0.5).astype(int),
                    'prediction_label': ['Anomaly' if p > 0.5 else 'Normal' for p in predictions]
                })
            else:
                # Multiclass classification - multiple probabilities
                pred_df = pd.DataFrame(predictions, columns=[f'class_{i}_prob' for i in range(predictions.shape[1])])
                pred_df['predicted_class'] = np.argmax(predictions, axis=1)
        else:
            # Handle other prediction formats
            pred_df = pd.DataFrame({'predictions': predictions})
        
        # Combine input data with predictions
        result_df = pd.concat([input_data.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        
        logging.info("Predictions completed successfully")
        logging.info(f"Result shape: {result_df.shape}")
        
        # Log prediction summary
        if 'prediction_label' in result_df.columns:
            normal_count = np.sum(result_df['prediction_label'] == 'Normal')
            anomaly_count = np.sum(result_df['prediction_label'] == 'Anomaly')
            logging.info(f"Prediction summary: {normal_count} Normal, {anomaly_count} Anomaly")
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise