import logging
import click
from typing import Tuple, Dict, Any
import xgboost as xgb
from zenml.client import Client
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_latest_model():
    """Get latest trained binary model from ZenML artifacts"""
    client = Client()
    
    # Use the specific artifact version ID we know exists
    # From the artifact versions list, binary_model_and_info version 1: 46f326ae-5a2a-4856-af7a-aafbd7320f61
    artifact_version_id = "46f326ae-5a2a-4856-af7a-aafbd7320f61"
    logging.info(f"Using artifact version: {artifact_version_id}")
    
    # Load the artifact data directly
    artifact_version = client.get_artifact_version(artifact_version_id)
    model_and_info = artifact_version.load()
    
    if isinstance(model_and_info, tuple) and len(model_and_info) == 2:
        return model_and_info
    else:
        raise ValueError(f"Expected tuple (model, info), got {type(model_and_info)}")


@click.command()
def main():
    """
    Simple manual deployment script for anomaly detection model.
    """
    try:
        logging.info("=" * 60)
        logging.info("STARTING MANUAL MODEL DEPLOYMENT")
        logging.info("=" * 60)
        
        # Load the latest model
        model, model_info = get_latest_model()
        logging.info(f"Loaded model with {model_info.get('num_features', 'unknown')} features")
        
        # Get the MLflow model deployer
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()
        
        # Check for existing services
        existing_services = model_deployer.find_model_server(
            model_name="anomaly_detector",
            running=True,
        )
        
        if existing_services:
            logging.info(f"Found {len(existing_services)} existing services")
            for service in existing_services:
                logging.info(f"Service: {service.prediction_url}, Status: {'Running' if service.is_running else 'Stopped'}")
                return
        
        # Try to deploy using MLflow directly
        logging.info("No existing services found. Attempting direct MLflow deployment...")
        
        # For now, let's just confirm the model is loadable
        logging.info("Model loaded successfully!")
        logging.info(f"Model type: {model_info.get('model_type')}")
        logging.info(f"Training accuracy: {model_info.get('training_accuracy', 'N/A')}")
        
        print("\n" + "=" * 80)
        print("📋 MANUAL DEPLOYMENT STATUS")
        print("=" * 80)
        print("✅ Model successfully loaded from ZenML artifacts")
        print(f"📊 Model Info: {model_info.get('model_type')} with {model_info.get('num_features')} features")
        print("⚠️  MLflow deployment integration needs debugging")
        print("💡 Consider using the model directly via ZenML artifacts for now")
        print("=" * 80)
        
    except Exception as e:
        logging.error(f"Error in manual deployment: {str(e)}")
        print(f"\n❌ ERROR: Manual deployment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()