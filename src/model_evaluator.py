import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelEvaluationStrategy(ABC):
    """Abstract base class for model evaluation strategies."""
    
    @abstractmethod
    def evaluate(self, model: xgb.Booster, X_test: pd.DataFrame, y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """Abstract method to evaluate a model."""
        pass


class BinaryModelEvaluationStrategy(ModelEvaluationStrategy):
    """Strategy for evaluating binary classification models."""
    
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
    
    def _plot_confusion_matrix(self, cm: np.ndarray, target_names: list, 
                              title: str = 'Confusion Matrix') -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            target_names (list): Target class names
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names, ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return fig
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        return fig
    
    def evaluate(self, model: xgb.Booster, X_test: pd.DataFrame, y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        Evaluate binary classification model.
        
        Args:
            model (xgb.Booster): Trained XGBoost model
            X_test (pd.DataFrame): Test features
            y_test (np.ndarray): Test labels (binary)
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            logging.info("Evaluating binary classification model")
            
            # Create DMatrix for prediction
            dtest = xgb.DMatrix(X_test)
            
            # Get predictions (probabilities)
            y_pred_proba = model.predict(dtest)
            
            # Get threshold from config
            threshold = self.config.get('evaluation', {}).get('binary_threshold', 0.5)
            
            # Convert probabilities to binary predictions
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])
            
            # Get feature importance
            importance_dict = model.get_score(importance_type='gain')
            feature_importance = pd.DataFrame(
                list(importance_dict.items()), 
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)
            
            # Create plots
            cm_plot = self._plot_confusion_matrix(cm, ['Normal', 'Anomaly'])
            roc_plot = self._plot_roc_curve(y_test, y_pred_proba)
            
            # Calculate additional metrics
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Prepare results
            results = {
                'model_type': 'binary',
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'specificity': specificity,
                'confusion_matrix': cm,
                'classification_report': report,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'threshold': threshold,
                'plots': {
                    'confusion_matrix': cm_plot,
                    'roc_curve': roc_plot
                },
                'summary': {
                    'total_samples': len(y_test),
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'anomaly_detection_rate': recall,
                    'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
                }
            }
            
            logging.info(f"Binary model evaluation completed. Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            return results
            
        except Exception as e:
            logging.error(f"Error in binary model evaluation: {str(e)}")
            raise


class MulticlassModelEvaluationStrategy(ModelEvaluationStrategy):
    """Strategy for evaluating multiclass classification models."""
    
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
    
    def _plot_confusion_matrix(self, cm: np.ndarray, target_names: list, 
                              title: str = 'Confusion Matrix') -> plt.Figure:
        """
        Plot confusion matrix for multiclass.
        
        Args:
            cm (np.ndarray): Confusion matrix
            target_names (list): Target class names
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Rotate labels if there are many classes
        if len(target_names) > 10:
            ax.set_xticklabels(target_names, rotation=45, ha='right')
            ax.set_yticklabels(target_names, rotation=0)
        
        return fig
    
    def evaluate(self, model: xgb.Booster, X_test: pd.DataFrame, y_test: np.ndarray, 
                **kwargs) -> Dict[str, Any]:
        """
        Evaluate multiclass classification model.
        
        Args:
            model (xgb.Booster): Trained XGBoost model
            X_test (pd.DataFrame): Test features
            y_test (np.ndarray): Test labels (multiclass integers)
            **kwargs: Additional arguments (should include label_encoder)
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            logging.info("Evaluating multiclass classification model")
            
            # Get label encoder from kwargs
            label_encoder = kwargs.get('label_encoder')
            if label_encoder is None:
                raise ValueError("Label encoder is required for multiclass evaluation")
            
            # Create DMatrix for prediction
            dtest = xgb.DMatrix(X_test)
            
            # Get predictions (probabilities for each class)
            y_pred_proba = model.predict(dtest)
            
            # Convert probabilities to class predictions
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Get class names
            class_names = label_encoder.classes_
            report = classification_report(y_test, y_pred, target_names=class_names)
            
            # Get feature importance
            importance_dict = model.get_score(importance_type='gain')
            feature_importance = pd.DataFrame(
                list(importance_dict.items()), 
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)
            
            # Create confusion matrix plot
            cm_plot = self._plot_confusion_matrix(cm, class_names)
            
            # Calculate per-class metrics
            per_class_metrics = {}
            for i, class_name in enumerate(class_names):
                # Calculate metrics for each class vs all others
                y_true_binary = (y_test == i).astype(int)
                y_pred_binary = (y_pred == i).astype(int)
                
                if np.sum(y_true_binary) > 0:  # Only if class exists in test set
                    precision = np.sum((y_pred_binary == 1) & (y_true_binary == 1)) / np.sum(y_pred_binary == 1) if np.sum(y_pred_binary == 1) > 0 else 0
                    recall = np.sum((y_pred_binary == 1) & (y_true_binary == 1)) / np.sum(y_true_binary == 1)
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    per_class_metrics[class_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'support': int(np.sum(y_true_binary))
                    }
            
            # Prepare results
            results = {
                'model_type': 'multiclass',
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'per_class_metrics': per_class_metrics,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'class_names': list(class_names),
                'plots': {
                    'confusion_matrix': cm_plot
                },
                'summary': {
                    'total_samples': len(y_test),
                    'num_classes': len(class_names),
                    'correctly_classified': int(np.sum(y_pred == y_test)),
                    'misclassified': int(np.sum(y_pred != y_test))
                }
            }
            
            logging.info(f"Multiclass model evaluation completed. Accuracy: {accuracy:.4f}")
            return results
            
        except Exception as e:
            logging.error(f"Error in multiclass model evaluation: {str(e)}")
            raise


class ModelEvaluator:
    """Context class for model evaluation."""
    
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initialize with a model evaluation strategy.
        
        Args:
            strategy (ModelEvaluationStrategy): The evaluation strategy to use
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Set a new evaluation strategy.
        
        Args:
            strategy (ModelEvaluationStrategy): New strategy to use
        """
        logging.info("Switching model evaluation strategy")
        self._strategy = strategy
    
    def evaluate_model(self, model: xgb.Booster, X_test: pd.DataFrame, y_test: np.ndarray, 
                      **kwargs) -> Dict[str, Any]:
        """
        Execute model evaluation using the current strategy.
        
        Args:
            model (xgb.Booster): Trained model
            X_test (pd.DataFrame): Test features
            y_test (np.ndarray): Test labels
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logging.info("Executing model evaluation with selected strategy")
        return self._strategy.evaluate(model, X_test, y_test, **kwargs)


# Factory for creating model evaluators
class ModelEvaluatorFactory:
    """Factory class to create appropriate model evaluators."""
    
    @staticmethod
    def get_model_evaluator(model_type: str, config_path: str = "config.yaml") -> ModelEvaluator:
        """
        Get the appropriate model evaluator based on model type.
        
        Args:
            model_type (str): Type of model ('binary', 'multiclass')
            config_path (str): Path to configuration file
            
        Returns:
            ModelEvaluator: Appropriate model evaluator instance
        """
        if model_type.lower() == 'binary':
            strategy = BinaryModelEvaluationStrategy(config_path)
        elif model_type.lower() == 'multiclass':
            strategy = MulticlassModelEvaluationStrategy(config_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return ModelEvaluator(strategy)


# Example usage
if __name__ == "__main__":
    # Example usage
    try:
        config_path = "config.yaml"
        
        # Create binary model evaluator
        binary_evaluator = ModelEvaluatorFactory.get_model_evaluator('binary', config_path)
        
        # Create multiclass model evaluator
        multiclass_evaluator = ModelEvaluatorFactory.get_model_evaluator('multiclass', config_path)
        
        # Evaluation would be done with actual model and data:
        # results = binary_evaluator.evaluate_model(model, X_test, y_test)
        
        logging.info("Model evaluators created successfully")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")