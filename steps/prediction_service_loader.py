import logging
from typing import Annotated
from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def prediction_service_loader(
    pipeline_name: str,
    step_name: str,
    running: bool = True,
) -> Annotated[MLFlowDeploymentService, "prediction_service"]:
    """
    Load a prediction service for the deployed model.
    
    Args:
        pipeline_name (str): Name of the deployment pipeline
        step_name (str): Name of the model deployer step
        running (bool): Whether to load only running services
        
    Returns:
        MLFlowDeploymentService: The loaded prediction service
    """
    try:
        logging.info(f"Loading prediction service from pipeline: {pipeline_name}, step: {step_name}")
        
        # Get the MLflow model deployer
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()
        
        # Find the deployment service
        services = model_deployer.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=step_name,
            running=running,
        )
        
        if not services:
            raise RuntimeError(
                f"No MLflow prediction service found for pipeline '{pipeline_name}' "
                f"and step '{step_name}'. Please run the deployment pipeline first."
            )
        
        service = services[0]
        logging.info(f"Loaded prediction service: {service}")
        logging.info(f"Service status: {service.get_status()}")
        logging.info(f"Prediction URL: {service.prediction_url}")
        
        return service
        
    except Exception as e:
        logging.error(f"Error loading prediction service: {str(e)}")
        raise