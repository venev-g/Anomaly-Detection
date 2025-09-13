from abc import ABC, abstractmethod


# Template Pattern Example for Anomaly Detection Pipeline
class AnomalyDetectionPipeline(ABC):
    """Template method pattern for anomaly detection pipeline."""
    
    def execute_pipeline(self):
        """Template method that defines the skeleton of the algorithm."""
        print("Starting Anomaly Detection Pipeline...")
        
        # Step 1: Load data
        data = self.load_data()
        
        # Step 2: Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Step 3: Train model
        model = self.train_model(processed_data)
        
        # Step 4: Evaluate model
        results = self.evaluate_model(model, processed_data)
        
        # Step 5: Deploy model
        self.deploy_model(model)
        
        print("Anomaly Detection Pipeline completed!")
        return results
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def load_data(self):
        """Load data - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def preprocess_data(self, data):
        """Preprocess data - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train_model(self, data):
        """Train model - must be implemented by subclasses."""
        pass
    
    # Hook methods - optional implementation by subclasses
    def evaluate_model(self, model, data):
        """Default evaluation - can be overridden."""
        return f"Default evaluation of {model} on {len(data)} samples"
    
    def deploy_model(self, model):
        """Default deployment - can be overridden."""
        print(f"Default deployment of {model}")


class NetworkAnomalyDetectionPipeline(AnomalyDetectionPipeline):
    """Concrete implementation for network anomaly detection."""
    
    def load_data(self):
        """Load network traffic data."""
        return "Network traffic data loaded from KDD99 dataset"
    
    def preprocess_data(self, data):
        """Preprocess network data with feature engineering."""
        return f"Preprocessed {data} with one-hot encoding and normalization"
    
    def train_model(self, data):
        """Train XGBoost model for network anomaly detection."""
        return f"XGBoost model trained on {data}"
    
    def evaluate_model(self, model, data):
        """Evaluate with network-specific metrics."""
        return f"Network model evaluation: AUC, Precision, Recall for {model}"
    
    def deploy_model(self, model):
        """Deploy model as network monitoring service."""
        print(f"Deployed {model} as network monitoring service")


class ApplicationAnomalyDetectionPipeline(AnomalyDetectionPipeline):
    """Concrete implementation for application anomaly detection."""
    
    def load_data(self):
        """Load application logs."""
        return "Application logs loaded from monitoring system"
    
    def preprocess_data(self, data):
        """Preprocess application data."""
        return f"Preprocessed {data} with log parsing and feature extraction"
    
    def train_model(self, data):
        """Train Isolation Forest for application anomaly detection."""
        return f"Isolation Forest model trained on {data}"
    
    def evaluate_model(self, model, data):
        """Evaluate with application-specific metrics."""
        return f"Application model evaluation: F1-score, Accuracy for {model}"


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("NETWORK ANOMALY DETECTION PIPELINE")
    print("=" * 60)
    
    network_pipeline = NetworkAnomalyDetectionPipeline()
    network_results = network_pipeline.execute_pipeline()
    print(f"Results: {network_results}")
    
    print("\n" + "=" * 60)
    print("APPLICATION ANOMALY DETECTION PIPELINE")
    print("=" * 60)
    
    app_pipeline = ApplicationAnomalyDetectionPipeline()
    app_results = app_pipeline.execute_pipeline()
    print(f"Results: {app_results}")