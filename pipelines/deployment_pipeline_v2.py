import logging
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from steps.latest_model_loader import latest_model_loader, model_to_mlflow_format
from steps.dynamic_importer import dynamic_importer
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@pipeline
def deployment_pipeline():
    """
    Deployment pipeline that loads the latest trained model and deploys it.
    
    This pipeline:
    1. Loads the latest trained binary model from ZenML artifacts
    2. Deploys it using MLflow model deployer
    """
    logging.info("Starting deployment pipeline")
    
    # Load the latest trained model
    latest_model_and_info = latest_model_loader()
    
    # Extract just the model for MLflow deployment
    mlflow_model = model_to_mlflow_format(latest_model_and_info)
    
    # Deploy the model using MLflow
    mlflow_model_deployer_step(
        workers=3,
        deploy_decision=True,
        model=mlflow_model,
    )
    
    logging.info("Deployment pipeline completed")


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
    
    # Run deployment
    deployment_pipeline()
    
    # Run inference
    inference_results = inference_pipeline()
    
    logging.info("Deployment pipeline example completed")