import logging
from typing import Annotated
import pandas as pd
from zenml import step
from src.data_ingester import DataIngestorFactory

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def data_ingestion_step(
    file_path: str,
    config_path: str = "config.yaml"
) -> Annotated[pd.DataFrame, "raw_data"]:
    """
    Data ingestion step for loading KDD99 network intrusion data.
    
    Args:
        file_path (str): Path to the KDD99 data file
        config_path (str): Path to configuration file
        
    Returns:
        pd.DataFrame: Raw data loaded from the file
    """
    try:
        logging.info(f"Starting data ingestion from: {file_path}")
        
        # Get KDD99 data ingestor
        ingestor = DataIngestorFactory.get_data_ingestor('kdd99', config_path=config_path)
        
        # Ingest the data
        raw_data = ingestor.ingest(file_path)
        
        logging.info(f"Data ingestion completed successfully. Shape: {raw_data.shape}")
        logging.info(f"Columns: {list(raw_data.columns)}")
        
        return raw_data
        
    except Exception as e:
        logging.error(f"Error in data ingestion step: {str(e)}")
        raise