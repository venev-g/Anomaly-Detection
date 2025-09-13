import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataPreprocessingStrategy(ABC):
    """Abstract base class for data preprocessing strategies."""
    
    @abstractmethod
    def preprocess(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Abstract method to preprocess data."""
        pass


class KDD99PreprocessingStrategy(DataPreprocessingStrategy):
    """Concrete strategy for KDD99 data preprocessing."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.label_encoder = LabelEncoder()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {str(e)}")
            raise
    
    def _reduce_anomalies(self, df: pd.DataFrame, pct_anomalies: float = 0.01) -> pd.DataFrame:
        """
        Reduce the number of anomalies to make dataset more realistic.
        
        Args:
            df (pd.DataFrame): Input dataframe
            pct_anomalies (float): Percentage of anomalies to keep
            
        Returns:
            pd.DataFrame: Dataframe with reduced anomalies
        """
        logging.info(f"Reducing anomalies to {pct_anomalies*100}% of normal data")
        
        labels = df['label'].copy()
        is_anomaly = labels != 'normal.'
        num_normal = np.sum(~is_anomaly)
        num_anomalies = int(pct_anomalies * num_normal)
        
        all_anomalies = labels[labels != 'normal.']
        anomalies_to_keep = np.random.choice(all_anomalies.index, size=num_anomalies, replace=False)
        
        anomalous_data = df.iloc[anomalies_to_keep].copy()
        normal_data = df[~is_anomaly].copy()
        
        new_df = pd.concat([normal_data, anomalous_data], axis=0)
        
        logging.info(f"Reduced dataset shape: {new_df.shape}")
        logging.info(f"New label distribution:\n{new_df['label'].value_counts()}")
        
        return new_df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        One-hot encode categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Categorical data and numeric data
        """
        cat_vars = self.config.get('features', {}).get('categorical_vars', [])
        
        logging.info(f"One-hot encoding categorical variables: {cat_vars}")
        
        # One-hot encode categorical variables
        cat_data = pd.get_dummies(df[cat_vars])
        
        # Get numeric variables
        numeric_vars = list(set(df.columns.values.tolist()) - set(cat_vars))
        numeric_vars.remove('label')
        numeric_data = df[numeric_vars].copy()
        
        logging.info(f"Categorical features shape: {cat_data.shape}")
        logging.info(f"Numeric features shape: {numeric_data.shape}")
        
        return cat_data, numeric_data
    
    def _prepare_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare labels for both binary and multiclass classification.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Binary labels and multiclass labels
        """
        # Fit label encoder
        labels = df['label'].copy()
        self.label_encoder.fit(labels)
        
        logging.info(f"Label classes: {self.label_encoder.classes_}")
        
        # Create multiclass labels (integers)
        multiclass_labels = self.label_encoder.transform(labels)
        
        # Create binary labels (0 = normal, 1 = anomaly)
        normal_idx = np.where(self.label_encoder.classes_ == 'normal.')[0][0]
        binary_labels = multiclass_labels.copy()
        binary_labels[binary_labels != normal_idx] = 1
        binary_labels[binary_labels == normal_idx] = 0
        
        logging.info(f"Binary label distribution: Normal={np.sum(binary_labels == 0)}, Anomaly={np.sum(binary_labels == 1)}")
        logging.info(f"Multiclass labels: {len(np.unique(multiclass_labels))} classes")
        
        return binary_labels, multiclass_labels
    
    def preprocess(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Preprocess KDD99 data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Preprocessed data dictionary
        """
        try:
            logging.info("Starting KDD99 data preprocessing")
            
            # Reduce anomalies if specified in config
            if self.config.get('data', {}).get('reduce_anomalies', False):
                pct_anomalies = self.config.get('data', {}).get('anomaly_percentage', 0.01)
                df = self._reduce_anomalies(df, pct_anomalies)
            
            # Encode categorical features
            cat_data, numeric_data = self._encode_categorical_features(df)
            
            # Combine numerical and categorical data
            features = pd.concat([numeric_data, cat_data], axis=1)
            
            # Prepare labels
            binary_labels, multiclass_labels = self._prepare_labels(df)
            
            # Split data
            test_size = self.config.get('data', {}).get('test_size', 0.25)
            random_state = self.config.get('data', {}).get('random_state', 42)
            
            # Split for binary classification
            X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
                features, binary_labels, test_size=test_size, random_state=random_state
            )
            
            # Split for multiclass classification
            X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
                features, multiclass_labels, test_size=test_size, random_state=random_state
            )
            
            logging.info(f"Training set shape: {X_train_bin.shape}")
            logging.info(f"Test set shape: {X_test_bin.shape}")
            
            # Prepare result dictionary
            result = {
                'binary_classification': {
                    'X_train': X_train_bin,
                    'X_test': X_test_bin,
                    'y_train': y_train_bin,
                    'y_test': y_test_bin
                },
                'multiclass_classification': {
                    'X_train': X_train_multi,
                    'X_test': X_test_multi,
                    'y_train': y_train_multi,
                    'y_test': y_test_multi
                },
                'label_encoder': self.label_encoder,
                'feature_names': list(features.columns),
                'num_classes': len(self.label_encoder.classes_)
            }
            
            logging.info("Data preprocessing completed successfully")
            return result
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise


class DataPreprocessor:
    """Context class for data preprocessing."""
    
    def __init__(self, strategy: DataPreprocessingStrategy):
        """
        Initialize with a preprocessing strategy.
        
        Args:
            strategy (DataPreprocessingStrategy): The preprocessing strategy to use
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: DataPreprocessingStrategy):
        """
        Set a new preprocessing strategy.
        
        Args:
            strategy (DataPreprocessingStrategy): New strategy to use
        """
        logging.info("Switching data preprocessing strategy")
        self._strategy = strategy
    
    def preprocess_data(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute data preprocessing using the current strategy.
        
        Args:
            df (pd.DataFrame): Input dataframe
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Preprocessed data dictionary
        """
        logging.info("Executing data preprocessing")
        return self._strategy.preprocess(df, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example usage
    try:
        # This would typically be called after data ingestion
        # df = ... # Load your data here
        
        config_path = "config.yaml"
        strategy = KDD99PreprocessingStrategy(config_path)
        preprocessor = DataPreprocessor(strategy)
        
        # preprocessed_data = preprocessor.preprocess_data(df)
        # print(f"Preprocessing completed. Keys: {list(preprocessed_data.keys())}")
        
        pass
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")