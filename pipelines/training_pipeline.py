import logging
from typing import Tuple, Dict, Any
import xgboost as xgb
from zenml import pipeline, Model
from steps.data_ingestion_step import data_ingestion_step
from steps.data_preprocessing_step import data_preprocessing_step
from steps.model_building_step import binary_model_building_step, multiclass_model_building_step
from steps.model_evaluation_step import binary_model_evaluation_step, multiclass_model_evaluation_step

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@pipeline(
    model=Model(
        name="anomaly_detector"
    ),
)
def anomaly_detection_training_pipeline(
    data_file_path: str = "data/kddcup.data.corrected",
    config_path: str = "config.yaml"
) -> Tuple[
    Tuple[xgb.Booster, Dict[str, Any]], 
    Tuple[xgb.Booster, Dict[str, Any]], 
    Dict[str, Any], 
    Dict[str, Any]
]:
    """
    End-to-end training pipeline for anomaly detection.
    
    This pipeline performs:
    1. Data ingestion from KDD99 dataset
    2. Data preprocessing (feature engineering, encoding, splitting)
    3. Binary model training (normal vs anomaly)
    4. Multiclass model training (attack type classification)
    5. Model evaluation for both models
    
    Args:
        data_file_path (str): Path to KDD99 data file
        config_path (str): Path to configuration file
        
    Returns:
        Tuple containing trained models and evaluation results
    """
    
    # Step 1: Data Ingestion
    raw_data = data_ingestion_step(
        file_path=data_file_path,
        config_path=config_path
    )
    
    # Step 2: Data Preprocessing
    preprocessed_data = data_preprocessing_step(
        raw_data=raw_data,
        config_path=config_path
    )
    
    # Step 3: Binary Model Building (Normal vs Anomaly)
    binary_model_and_info = binary_model_building_step(
        preprocessed_data=preprocessed_data,
        config_path=config_path
    )
    
    # Step 4: Multiclass Model Building (Attack Type Classification)
    multiclass_model_and_info = multiclass_model_building_step(
        preprocessed_data=preprocessed_data,
        config_path=config_path
    )
    
    # Step 5: Binary Model Evaluation
    binary_evaluation_results = binary_model_evaluation_step(
        binary_model_and_info=binary_model_and_info,
        preprocessed_data=preprocessed_data,
        config_path=config_path
    )
    
    # Step 6: Multiclass Model Evaluation
    multiclass_evaluation_results = multiclass_model_evaluation_step(
        multiclass_model_and_info=multiclass_model_and_info,
        preprocessed_data=preprocessed_data,
        config_path=config_path
    )
    
    return (
        binary_model_and_info, 
        multiclass_model_and_info, 
        binary_evaluation_results, 
        multiclass_evaluation_results
    )


if __name__ == "__main__":
    # Run the training pipeline
    logging.info("Starting anomaly detection training pipeline")
    
    pipeline_run = anomaly_detection_training_pipeline()
    
    logging.info("Anomaly detection training pipeline completed successfully")
    print("Pipeline execution completed. Check MLflow UI for detailed results.")