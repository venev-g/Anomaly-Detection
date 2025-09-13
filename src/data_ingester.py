import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd
import yaml

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataIngestor(ABC):
    """Abstract base class for data ingestion strategies."""
    
    @abstractmethod
    def ingest(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


class CSVDataIngestor(DataIngestor):
    """Concrete implementation for CSV data ingestion."""
    
    def ingest(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Ingest data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            pd.DataFrame: The ingested data
        """
        try:
            logging.info(f"Ingesting data from CSV file: {file_path}")
            
            # Set default parameters for CSV reading
            csv_params = {
                'header': kwargs.get('header', None),
                'names': kwargs.get('names', None),
                'index_col': kwargs.get('index_col', False)
            }
            
            # Remove None values
            csv_params = {k: v for k, v in csv_params.items() if v is not None}
            
            df = pd.read_csv(file_path, **csv_params)
            logging.info(f"Successfully ingested {len(df)} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logging.error(f"Error ingesting data from {file_path}: {str(e)}")
            raise


class KDD99DataIngestor(DataIngestor):
    """Specialized data ingestor for KDD99 dataset."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {str(e)}")
            raise
    
    def ingest(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Ingest KDD99 dataset with proper column names.
        
        Args:
            file_path (str): Path to the KDD99 data file
            **kwargs: Additional arguments
            
        Returns:
            pd.DataFrame: The ingested KDD99 data with proper column names
        """
        try:
            logging.info(f"Ingesting KDD99 data from: {file_path}")
            
            # Get column names from config
            col_names = self.config.get('features', {}).get('column_names', [])
            
            if not col_names:
                raise ValueError("Column names not found in configuration")
            
            # Read the data without header and assign column names
            df = pd.read_csv(file_path, header=None, names=col_names, index_col=False)
            
            logging.info(f"Successfully ingested KDD99 data: {len(df)} rows, {len(df.columns)} columns")
            logging.info(f"Label distribution:\n{df['label'].value_counts()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error ingesting KDD99 data from {file_path}: {str(e)}")
            raise


class DataIngestorFactory:
    """Factory class to create appropriate data ingestors."""
    
    @staticmethod
    def get_data_ingestor(data_type: str, **kwargs) -> DataIngestor:
        """
        Get the appropriate data ingestor based on data type.
        
        Args:
            data_type (str): Type of data ('csv', 'kdd99')
            **kwargs: Additional arguments for specific ingestors
            
        Returns:
            DataIngestor: Appropriate data ingestor instance
        """
        if data_type.lower() == 'csv':
            return CSVDataIngestor()
        elif data_type.lower() == 'kdd99':
            config_path = kwargs.get('config_path', 'config.yaml')
            return KDD99DataIngestor(config_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")


# Example usage
if __name__ == "__main__":
    # Example usage for KDD99 data
    config_path = "config.yaml"  # Adjust path as needed
    data_path = "data/kddcup.data.corrected"  # Adjust path as needed
    
    try:
        # Get KDD99 data ingestor
        ingestor = DataIngestorFactory.get_data_ingestor('kdd99', config_path=config_path)
        
        # Ingest the data
        df = ingestor.ingest(data_path)
        
        # Display basic information
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")