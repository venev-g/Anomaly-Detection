import logging
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pipelines.training_pipeline import anomaly_detection_training_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@pipeline
def continuous_deployment_pipeline():
    """
    Continuous deployment pipeline for anomaly detection.
    
    This pipeline:
    1. Runs the training pipeline to get trained models
    2. Deploys the binary classification model using MLflow
    """
    logging.info("Starting continuous deployment pipeline")
    
    # Run the training pipeline to get trained models
    (
        binary_model_and_info, 
        multiclass_model_and_info, 
        binary_evaluation_results, 
        multiclass_evaluation_results
    ) = anomaly_detection_training_pipeline()
    
    # Deploy the binary model (primary model for anomaly detection)
    binary_model, _ = binary_model_and_info
    
    # Deploy the model using MLflow
    mlflow_model_deployer_step(
        workers=3,
        deploy_decision=True,
        model=binary_model,
    )
    
    logging.info("Continuous deployment pipeline completed")


@pipeline(enable_cache=False)
def inference_pipeline():
    """
    Batch inference pipeline for anomaly detection.
    
    This pipeline:
    1. Loads sample data for inference
    2. Loads the deployed model service
    3. Makes predictions on the sample data
    """
    logging.info("Starting inference pipeline")
    
    # Load batch data for inference
    batch_data = dynamic_importer()
    
    # Load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )
    
    # Run predictions on the batch data
    predictions = predictor(
        service=model_deployment_service, 
        input_data=batch_data
    )
    
    logging.info("Inference pipeline completed")
    
    return predictions


if __name__ == "__main__":
    # Example of running the deployment pipeline
    logging.info("Running deployment pipeline example")
    
    # Run continuous deployment
    continuous_deployment_pipeline()
    
    # Run inference
    inference_results = inference_pipeline()
    
    logging.info("Deployment pipeline example completed")