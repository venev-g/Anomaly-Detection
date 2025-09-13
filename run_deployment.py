import click
import logging
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def main(stop_service: bool):
    """
    Run the anomaly detection deployment pipeline.
    
    This will deploy the trained anomaly detection models and start a prediction service
    for real-time network intrusion detection.
    """
    model_name = "anomaly_detector"

    if stop_service:
        try:
            logging.info("Stopping anomaly detection prediction service...")
            
            # Get the MLflow model deployer stack component
            model_deployer = MLFlowModelDeployer.get_active_model_deployer()

            # Fetch existing services with same pipeline name, step name, and model name
            existing_services = model_deployer.find_model_server(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                model_name=model_name,
                running=True,
            )

            if existing_services:
                existing_services[0].stop(timeout=10)
                print("✅ Anomaly detection prediction service stopped successfully.")
            else:
                print("ℹ️  No running anomaly detection prediction service found.")
            return
            
        except Exception as e:
            logging.error(f"Error stopping service: {str(e)}")
            print(f"❌ Error stopping service: {str(e)}")
            return

    try:
        logging.info("=" * 60)
        logging.info("STARTING ANOMALY DETECTION DEPLOYMENT PIPELINE")
        logging.info("=" * 60)

        # Run the continuous deployment pipeline
        logging.info("Running continuous deployment pipeline...")
        continuous_deployment_pipeline()

        # Get the active model deployer
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # Run the inference pipeline
        logging.info("Running inference pipeline...")
        inference_pipeline()

        logging.info("=" * 60)
        logging.info("DEPLOYMENT PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)

        # Get tracking URI for MLflow
        tracking_uri = get_tracking_uri()

        print("\n" + "=" * 80)
        print("🚀 ANOMALY DETECTION SERVICE DEPLOYED SUCCESSFULLY! 🚀")
        print("=" * 80)
        
        print("\n📊 DEPLOYMENT SUMMARY:")
        print("   ✅ Binary classification model deployed")
        print("   ✅ MLflow prediction service started")
        print("   ✅ Batch inference pipeline executed")
        print("   ✅ Sample predictions generated")

        print("\n📈 EXPERIMENT TRACKING:")
        print("   Run the following command to view detailed results in MLflow UI:")
        print(f"   mlflow ui --backend-store-uri '{tracking_uri}'")

        # Fetch existing services to get service URL
        service = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

        if service and len(service) > 0:
            print("\n🌐 PREDICTION SERVICE:")
            print("   The anomaly detection prediction server is running locally as a daemon")
            print("   process and accepts inference requests at:")
            print(f"   {service[0].prediction_url}")
            print(f"   \n   Service Status: {service[0].get_status()}")
            
            print("\n🔍 USAGE:")
            print("   • Send POST requests to the prediction URL with network traffic data")
            print("   • Use sample_predict.py for testing predictions")
            print("   • The model detects normal vs anomalous network traffic")
            
            print("\n🛑 STOP SERVICE:")
            print("   To stop the service, re-run this command with --stop-service:")
            print("   python run_deployment.py --stop-service")
        else:
            print("\n⚠️  Warning: Could not find the deployed service details.")

        print("=" * 80)

    except Exception as e:
        logging.error(f"Error in deployment pipeline: {str(e)}")
        print(f"\n❌ ERROR: Deployment pipeline failed with error: {str(e)}")
        print("Please check the logs for detailed error information.")
        print("Make sure you have run the training pipeline first: python run_pipeline.py")
        raise


if __name__ == "__main__":
    main()