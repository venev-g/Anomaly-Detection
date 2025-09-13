from abc import ABC, abstractmethod


# Strategy Pattern Example for Anomaly Detection Models
class AnomalyDetectionStrategy(ABC):
    """Strategy interface for different anomaly detection approaches."""
    
    @abstractmethod
    def detect_anomaly(self, data):
        """Detect anomalies in the given data."""
        pass


class XGBoostAnomalyDetectionStrategy(AnomalyDetectionStrategy):
    """Concrete strategy using XGBoost for anomaly detection."""
    
    def detect_anomaly(self, data):
        return f"Detecting anomalies using XGBoost on data with shape {data.shape if hasattr(data, 'shape') else len(data)}"


class IsolationForestAnomalyDetectionStrategy(AnomalyDetectionStrategy):
    """Concrete strategy using Isolation Forest for anomaly detection."""
    
    def detect_anomaly(self, data):
        return f"Detecting anomalies using Isolation Forest on data with shape {data.shape if hasattr(data, 'shape') else len(data)}"


class AutoEncoderAnomalyDetectionStrategy(AnomalyDetectionStrategy):
    """Concrete strategy using AutoEncoder for anomaly detection."""
    
    def detect_anomaly(self, data):
        return f"Detecting anomalies using AutoEncoder on data with shape {data.shape if hasattr(data, 'shape') else len(data)}"


# Context class
class AnomalyDetector:
    """Context class that uses different anomaly detection strategies."""
    
    def __init__(self, strategy: AnomalyDetectionStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: AnomalyDetectionStrategy):
        """Change the anomaly detection strategy at runtime."""
        self.strategy = strategy
    
    def detect(self, data):
        """Execute anomaly detection using the current strategy."""
        return self.strategy.detect_anomaly(data)


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = [1, 2, 3, 4, 5]
    
    # Create detector with XGBoost strategy
    detector = AnomalyDetector(XGBoostAnomalyDetectionStrategy())
    print(detector.detect(sample_data))
    
    # Switch to Isolation Forest strategy
    detector.set_strategy(IsolationForestAnomalyDetectionStrategy())
    print(detector.detect(sample_data))
    
    # Switch to AutoEncoder strategy
    detector.set_strategy(AutoEncoderAnomalyDetectionStrategy())
    print(detector.detect(sample_data))