from abc import ABC, abstractmethod


# Factory Pattern Example for Anomaly Detection Models
class AnomalyDetectionModel(ABC):
    """Abstract base class for anomaly detection models."""
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """Make predictions."""
        pass


class XGBoostAnomalyModel(AnomalyDetectionModel):
    """Concrete XGBoost model for anomaly detection."""
    
    def train(self, X_train, y_train):
        return f"Training XGBoost model with {len(X_train)} samples"
    
    def predict(self, X_test):
        return f"XGBoost predictions for {len(X_test)} samples"


class IsolationForestAnomalyModel(AnomalyDetectionModel):
    """Concrete Isolation Forest model for anomaly detection."""
    
    def train(self, X_train, y_train):
        return f"Training Isolation Forest model with {len(X_train)} samples"
    
    def predict(self, X_test):
        return f"Isolation Forest predictions for {len(X_test)} samples"


class AutoEncoderAnomalyModel(AnomalyDetectionModel):
    """Concrete AutoEncoder model for anomaly detection."""
    
    def train(self, X_train, y_train):
        return f"Training AutoEncoder model with {len(X_train)} samples"
    
    def predict(self, X_test):
        return f"AutoEncoder predictions for {len(X_test)} samples"


# Factory class
class AnomalyDetectionModelFactory:
    """Factory for creating anomaly detection models."""
    
    @staticmethod
    def create_model(model_type: str) -> AnomalyDetectionModel:
        """Create an anomaly detection model based on the specified type."""
        
        if model_type.lower() == "xgboost":
            return XGBoostAnomalyModel()
        elif model_type.lower() == "isolation_forest":
            return IsolationForestAnomalyModel()
        elif model_type.lower() == "autoencoder":
            return AutoEncoderAnomalyModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Sample data
    X_train = [1, 2, 3, 4, 5]
    y_train = [0, 0, 1, 0, 1]
    X_test = [6, 7, 8]
    
    # Create different models using the factory
    models = ["xgboost", "isolation_forest", "autoencoder"]
    
    for model_type in models:
        print(f"\nCreating {model_type} model:")
        model = AnomalyDetectionModelFactory.create_model(model_type)
        print(model.train(X_train, y_train))
        print(model.predict(X_test))