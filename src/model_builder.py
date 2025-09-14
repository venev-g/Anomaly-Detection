import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelBuildingStrategy(ABC):
    """Abstract base class for model building strategies."""
    
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                             X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[xgb.Booster, Dict[str, Any]]:
        """
        Abstract method to build and train a model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels
            X_test (pd.DataFrame): Test features  
            y_test (np.ndarray): Test labels
            
        Returns:
            Tuple[xgb.Booster, Dict[str, Any]]: Trained model and training info
        """
        pass


class BinaryXGBoostStrategy(ModelBuildingStrategy):
    """Strategy for binary classification using XGBoost."""
    
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
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                             X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[xgb.Booster, Dict[str, Any]]:
        """
        Build and train binary XGBoost model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels (binary: 0=normal, 1=anomaly)
            X_test (pd.DataFrame): Test features  
            y_test (np.ndarray): Test labels
            
        Returns:
            Tuple[xgb.Booster, Dict[str, Any]]: Trained model and training info
        """
        try:
            logging.info("Building binary XGBoost model for anomaly detection")
            
            # Get XGBoost parameters from config
            xgb_config = self.config.get('xgboost', {}).get('binary_classification', {})
            
            # Prepare parameters
            params = {
                'max_depth': xgb_config.get('max_depth', 8),
                'max_leaves': xgb_config.get('max_leaves', 256),
                'alpha': xgb_config.get('alpha', 0.9),
                'eta': xgb_config.get('eta', 0.1),
                'gamma': xgb_config.get('gamma', 0.1),
                'learning_rate': xgb_config.get('learning_rate', 0.1),
                'subsample': xgb_config.get('subsample', 1),
                'reg_lambda': xgb_config.get('reg_lambda', 1),
                'scale_pos_weight': xgb_config.get('scale_pos_weight', 2),
                'tree_method': xgb_config.get('tree_method', 'gpu_hist'),
                'n_gpus': xgb_config.get('n_gpus', 1),
                'objective': xgb_config.get('objective', 'binary:logistic'),
                'verbose': xgb_config.get('verbose', True)
            }
            
            num_rounds = xgb_config.get('num_rounds', 10)
            
            logging.info(f"XGBoost parameters: {params}")
            logging.info(f"Number of training rounds: {num_rounds}")
            
            # Create DMatrix objects
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # Set evaluation sets
            evals = [(dtest, 'test'), (dtrain, 'train')]
            
            # Train the model
            logging.info("Starting XGBoost training...")
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_rounds,
                evals=evals,
                verbose_eval=xgb_config.get('verbose', True)
            )
            
            # Log model and metrics to MLflow (using the already active run)
            logging.info("Logging binary model to MLflow...")
            
            try:
                # Get predictions for signature inference
                train_predictions = model.predict(dtrain)
                signature = infer_signature(X_train, train_predictions)
                
                # Log parameters with binary prefix to avoid conflicts
                binary_params = {f"binary_{k}": v for k, v in params.items()}
                mlflow.log_params(binary_params)
                mlflow.log_param("binary_num_rounds", num_rounds)
                mlflow.log_param("binary_num_features", X_train.shape[1])
                mlflow.log_param("binary_train_samples", X_train.shape[0])
                mlflow.log_param("binary_test_samples", X_test.shape[0])
                
                # Log the XGBoost model
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="binary_model",
                    signature=signature,
                    input_example=X_train.head(5),
                    registered_model_name="binary_anomaly_detection_model"
                )
                
                logging.info("Binary model successfully logged to MLflow")
            except Exception as e:
                logging.warning(f"Failed to log binary model to MLflow: {str(e)}")
                # Continue execution even if MLflow logging fails
            
            # Prepare training info
            training_info = {
                'model_type': 'binary_xgboost',
                'parameters': params,
                'num_rounds': num_rounds,
                'feature_names': list(X_train.columns),
                'num_features': X_train.shape[1],
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0]
            }
            
            logging.info("Binary XGBoost model training completed successfully")
            return model, training_info
            
        except Exception as e:
            logging.error(f"Error in binary XGBoost model training: {str(e)}")
            raise


class MulticlassXGBoostStrategy(ModelBuildingStrategy):
    """Strategy for multiclass classification using XGBoost."""
    
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
    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                             X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[xgb.Booster, Dict[str, Any]]:
        """
        Build and train multiclass XGBoost model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels (multiclass: integers for each attack type)
            X_test (pd.DataFrame): Test features  
            y_test (np.ndarray): Test labels
            
        Returns:
            Tuple[xgb.Booster, Dict[str, Any]]: Trained model and training info
        """
        try:
            logging.info("Building multiclass XGBoost model for attack type classification")
            
            # Get XGBoost parameters from config
            xgb_config = self.config.get('xgboost', {}).get('multiclass_classification', {})
            
            # Calculate number of classes
            num_classes = len(np.unique(y_train))
            logging.info(f"Number of classes: {num_classes}")
            
            # Prepare parameters
            params = {
                'max_depth': xgb_config.get('max_depth', 8),
                'max_leaves': xgb_config.get('max_leaves', 256),
                'alpha': xgb_config.get('alpha', 0.9),
                'eta': xgb_config.get('eta', 0.1),
                'gamma': xgb_config.get('gamma', 0.1),
                'learning_rate': xgb_config.get('learning_rate', 0.1),
                'subsample': xgb_config.get('subsample', 1),
                'reg_lambda': xgb_config.get('reg_lambda', 1),
                'tree_method': xgb_config.get('tree_method', 'gpu_hist'),
                'n_gpus': xgb_config.get('n_gpus', 1),
                'objective': xgb_config.get('objective', 'multi:softprob'),
                'num_class': num_classes,
                'verbose': xgb_config.get('verbose', True)
            }
            
            num_rounds = xgb_config.get('num_rounds', 10)
            
            logging.info(f"XGBoost parameters: {params}")
            logging.info(f"Number of training rounds: {num_rounds}")
            
            # Create DMatrix objects
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # Set evaluation sets
            evals = [(dtest, 'test'), (dtrain, 'train')]
            
            # Train the model
            logging.info("Starting XGBoost training...")
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_rounds,
                evals=evals,
                verbose_eval=xgb_config.get('verbose', True)
            )
            
            # Log model and metrics to MLflow (using the already active run)
            logging.info("Logging multiclass model to MLflow...")
            
            try:
                # Get predictions for signature inference
                train_predictions = model.predict(dtrain)
                signature = infer_signature(X_train, train_predictions)
                
                # Log parameters with multiclass prefix to avoid conflicts
                multiclass_params = {f"multiclass_{k}": v for k, v in params.items()}
                mlflow.log_params(multiclass_params)
                mlflow.log_param("multiclass_num_rounds", num_rounds)
                mlflow.log_param("multiclass_num_classes", num_classes)
                mlflow.log_param("multiclass_num_features", X_train.shape[1])
                mlflow.log_param("multiclass_train_samples", X_train.shape[0])
                mlflow.log_param("multiclass_test_samples", X_test.shape[0])
                
                # Log the XGBoost model
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="multiclass_model",
                    signature=signature,
                    input_example=X_train.head(5),
                    registered_model_name="multiclass_attack_detection_model"
                )
                
                logging.info("Multiclass model successfully logged to MLflow")
            except Exception as e:
                logging.warning(f"Failed to log multiclass model to MLflow: {str(e)}")
                # Continue execution even if MLflow logging fails
            
            # Prepare training info
            training_info = {
                'model_type': 'multiclass_xgboost',
                'parameters': params,
                'num_rounds': num_rounds,
                'num_classes': num_classes,
                'feature_names': list(X_train.columns),
                'num_features': X_train.shape[1],
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0]
            }
            
            logging.info("Multiclass XGBoost model training completed successfully")
            return model, training_info
            
        except Exception as e:
            logging.error(f"Error in multiclass XGBoost model training: {str(e)}")
            raise


class ModelBuilder:
    """Context class for model building."""
    
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initialize with a model building strategy.
        
        Args:
            strategy (ModelBuildingStrategy): The model building strategy to use
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Set a new model building strategy.
        
        Args:
            strategy (ModelBuildingStrategy): New strategy to use
        """
        logging.info("Switching model building strategy")
        self._strategy = strategy
    
    def build_model(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                   X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[xgb.Booster, Dict[str, Any]]:
        """
        Execute model building using the current strategy.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels
            X_test (pd.DataFrame): Test features  
            y_test (np.ndarray): Test labels
            
        Returns:
            Tuple[xgb.Booster, Dict[str, Any]]: Trained model and training info
        """
        logging.info("Executing model building with selected strategy")
        return self._strategy.build_and_train_model(X_train, y_train, X_test, y_test)


# Factory for creating model builders
class ModelBuilderFactory:
    """Factory class to create appropriate model builders."""
    
    @staticmethod
    def get_model_builder(model_type: str, config_path: str = "config.yaml") -> ModelBuilder:
        """
        Get the appropriate model builder based on model type.
        
        Args:
            model_type (str): Type of model ('binary', 'multiclass')
            config_path (str): Path to configuration file
            
        Returns:
            ModelBuilder: Appropriate model builder instance
        """
        if model_type.lower() == 'binary':
            strategy = BinaryXGBoostStrategy(config_path)
        elif model_type.lower() == 'multiclass':
            strategy = MulticlassXGBoostStrategy(config_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return ModelBuilder(strategy)


# Example usage
if __name__ == "__main__":
    # Example usage
    try:
        config_path = "config.yaml"
        
        # Create binary model builder
        binary_builder = ModelBuilderFactory.get_model_builder('binary', config_path)
        
        # Create multiclass model builder
        multiclass_builder = ModelBuilderFactory.get_model_builder('multiclass', config_path)
        
        # Training would be done with actual data:
        # model, info = binary_builder.build_model(X_train, y_train, X_test, y_test)
        
        logging.info("Model builders created successfully")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")