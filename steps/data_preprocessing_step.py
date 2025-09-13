import logging
from typing import Annotated, Dict, Any
import pandas as pd
from zenml import step
from src.data_preprocessor import DataPreprocessor, KDD99PreprocessingStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def data_preprocessing_step(
    raw_data: pd.DataFrame,
    config_path: str = "config.yaml"
) -> Annotated[Dict[str, Any], "preprocessed_data"]:
    """
    Data preprocessing step for KDD99 network intrusion data.
    
    Args:
        raw_data (pd.DataFrame): Raw data from ingestion step
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Dictionary containing preprocessed data for both binary and multiclass classification
    """
    try:
        logging.info("Starting data preprocessing")
        
        # Create preprocessing strategy and processor
        strategy = KDD99PreprocessingStrategy(config_path)
        preprocessor = DataPreprocessor(strategy)
        
        # Preprocess the data
        preprocessed_data = preprocessor.preprocess_data(raw_data)
        
        logging.info("Data preprocessing completed successfully")
        logging.info(f"Binary classification data - Train: {preprocessed_data['binary_classification']['X_train'].shape}, Test: {preprocessed_data['binary_classification']['X_test'].shape}")
        logging.info(f"Multiclass classification data - Train: {preprocessed_data['multiclass_classification']['X_train'].shape}, Test: {preprocessed_data['multiclass_classification']['X_test'].shape}")
        
        return preprocessed_data
        
    except Exception as e:
        logging.error(f"Error in data preprocessing step: {str(e)}")
        raise