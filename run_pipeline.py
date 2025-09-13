import click
import logging
from pipelines.training_pipeline import anomaly_detection_training_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@click.command()
@click.option(
    "--data-path",
    default="data/kddcup.data.corrected",
    help="Path to the KDD99 dataset file"
)
@click.option(
    "--config-path",
    default="config.yaml",
    help="Path to the configuration file"
)
def main(data_path: str, config_path: str):
    """
    Run the anomaly detection training pipeline.
    
    This will train both binary (anomaly vs normal) and multiclass (attack type) 
    XGBoost models for network intrusion detection using the KDD99 dataset.
    """
    try:
        logging.info("=" * 60)
        logging.info("STARTING ANOMALY DETECTION TRAINING PIPELINE")
        logging.info("=" * 60)
        
        logging.info(f"Data path: {data_path}")
        logging.info(f"Config path: {config_path}")
        
        # Run the training pipeline
        anomaly_detection_training_pipeline(
            data_file_path=data_path,
            config_path=config_path
        )
        
        logging.info("=" * 60)
        logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)
        
        # Get tracking URI for MLflow
        tracking_uri = get_tracking_uri()
        
        print("\n" + "=" * 80)
        print("🎉 ANOMALY DETECTION TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 80)
        print("\n📊 RESULTS SUMMARY:")
        print("   ✅ Binary classification model trained (Normal vs Anomaly)")
        print("   ✅ Multiclass classification model trained (Attack Type Classification)")
        print("   ✅ Both models evaluated with comprehensive metrics")
        print("   ✅ Feature importance analysis completed")
        print("   ✅ Confusion matrices and ROC curves generated")
        
        print("\n📈 EXPERIMENT TRACKING:")
        print("   Run the following command to view detailed results in MLflow UI:")
        print(f"   mlflow ui --backend-store-uri '{tracking_uri}'")
        
        print("\n🔍 MODELS TRAINED:")
        print("   • Binary XGBoost: Detects anomalous vs normal network traffic")
        print("   • Multiclass XGBoost: Classifies specific types of network attacks")
        print("   • Both models use GPU acceleration for fast training")
        
        print("\n📁 NEXT STEPS:")
        print("   • Review model performance in MLflow UI")
        print("   • Run deployment pipeline for model serving")
        print("   • Use sample_predict.py for testing predictions")
        
        print("=" * 80)
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        print(f"\n❌ ERROR: Training pipeline failed with error: {str(e)}")
        print("Please check the logs for detailed error information.")
        raise


if __name__ == "__main__":
    main()