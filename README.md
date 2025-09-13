# Anomaly Detection System

A production-grade end-to-end anomaly detection system built with XGBoost and GPU acceleration, designed for network intrusion detection using the KDD99 dataset. This system implements modern MLOps practices with ZenML pipelines, MLflow experiment tracking, and follows established design patterns for maintainability and scalability.

## 🏗️ Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
anomaly-detection-system/
├── config.yaml                    # Central configuration
├── requirements.txt               # Dependencies
├── run_pipeline.py               # Training pipeline runner
├── run_deployment.py             # Deployment pipeline runner
├── sample_predict.py             # Prediction testing script
├── src/                          # Core modules
│   ├── data_ingester.py         # Data ingestion strategies
│   ├── data_preprocessor.py     # Data preprocessing
│   ├── model_builder.py         # Model building strategies
│   └── model_evaluator.py       # Model evaluation
├── steps/                        # ZenML pipeline steps
├── pipelines/                    # Training & deployment pipelines
└── explanations/                 # Design pattern examples
```

## 🚀 Key Features

- **GPU-Accelerated XGBoost**: Leverages CUDA for faster training and inference
- **Dual Classification Modes**: Binary anomaly detection and multiclass attack classification
- **MLOps Integration**: ZenML pipelines with MLflow experiment tracking
- **Design Patterns**: Strategy, Factory, and Template patterns for maintainability
- **Production Ready**: Comprehensive logging, error handling, and configuration management
- **Real-time Inference**: Deployed model service for live predictions

## 📊 Supported Detection Types

### Binary Classification
- **Normal** vs **Anomaly** detection
- Optimized for high recall to catch all potential threats
- Uses AUC and ROC curve analysis

### Multiclass Classification
- Detailed attack type classification:
  - **Normal**: Legitimate network traffic
  - **DoS**: Denial of Service attacks
  - **Probe**: Surveillance and probing attacks
  - **R2L**: Remote to Local attacks
  - **U2R**: User to Root attacks

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd anomaly-detection-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the KDD99 dataset**:
   The system expects the KDD99 dataset at `data/kddcup.data.corrected`. You can download it from:
   ```bash
   mkdir -p data
   wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
   gunzip kddcup.data_10_percent.gz
   mv kddcup.data_10_percent data/kddcup.data.corrected
   ```

## 🏃‍♂️ Usage

### Configuration

The system is fully configurable through `config.yaml`. Key sections include:

```yaml
# Model Configuration
model:
  binary_xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    tree_method: "gpu_hist"  # GPU acceleration
    
  multiclass_xgboost:
    n_estimators: 100
    max_depth: 8
    learning_rate: 0.1
    tree_method: "gpu_hist"

# Data Configuration
data:
  test_size: 0.2
  random_state: 42
  sampling:
    anomaly_fraction: 0.1  # Reduce anomaly samples for balance
```

### Training Pipeline

Run the complete training pipeline:

```bash
python run_pipeline.py --config config.yaml
```

This will:
1. Ingest and preprocess the KDD99 dataset
2. Train both binary and multiclass XGBoost models
3. Evaluate models with comprehensive metrics
4. Register models with MLflow
5. Generate evaluation plots and reports

### Deployment Pipeline

Deploy trained models for inference:

```bash
python run_deployment.py --config config.yaml
```

This will:
1. Load the best trained models
2. Deploy them as MLflow services
3. Set up batch inference capabilities
4. Enable real-time prediction endpoints

### Testing Predictions

Test the deployed model:

```bash
python sample_predict.py --config config.yaml
```

This script demonstrates how to:
- Load sample data
- Make predictions using deployed models
- Interpret results for both binary and multiclass outputs

## 📈 Model Performance

The system provides comprehensive evaluation metrics:

### Binary Classification Metrics
- **AUC Score**: Area Under the ROC Curve
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: For anomaly detection optimization
- **ROC Curve**: Visual performance analysis
- **Confusion Matrix**: Detailed classification breakdown

### Multiclass Classification Metrics
- **Per-class Accuracy**: Individual attack type performance
- **Macro/Micro F1-Score**: Overall multiclass performance
- **Confusion Matrix**: Attack type classification details
- **Feature Importance**: Top contributing features

### Sample Output
```
Binary Classification Results:
- AUC Score: 0.9856
- Accuracy: 0.9423
- Precision: 0.9234
- Recall: 0.9156

Multiclass Classification Results:
- Overall Accuracy: 0.9234
- Macro F1-Score: 0.8967
- Per-class Performance:
  * Normal: F1=0.96
  * DoS: F1=0.94
  * Probe: F1=0.89
  * R2L: F1=0.67
  * U2R: F1=0.45
```

## 🔧 Design Patterns

The system implements several design patterns for maintainability:

### Strategy Pattern
- **Data Ingestion**: Multiple data source strategies
- **Preprocessing**: Different preprocessing approaches
- **Model Building**: Various algorithm strategies
- **Evaluation**: Different evaluation metrics

### Factory Pattern
- **Model Factory**: Dynamic model creation based on configuration
- **Data Ingester Factory**: Automatic ingester selection
- **Evaluator Factory**: Metric-specific evaluator creation

### Template Pattern
- **Pipeline Template**: Common pipeline structure
- **Evaluation Template**: Standard evaluation workflow

See the `explanations/` directory for detailed examples.

## 🔍 Monitoring & Logging

The system includes comprehensive monitoring:

- **MLflow Tracking**: Experiment tracking and model versioning
- **Structured Logging**: Detailed operation logs
- **Performance Metrics**: Training and inference performance
- **Model Drift Detection**: Monitor prediction distributions

## 🚨 Security Considerations

When deploying in production:

1. **Data Privacy**: Ensure network data is properly anonymized
2. **Model Security**: Protect model endpoints from unauthorized access
3. **Resource Monitoring**: Monitor GPU memory and compute usage
4. **Alert Systems**: Set up alerts for anomaly detection failures
5. **Model Updates**: Regular retraining on fresh attack patterns

## 📊 Extending the System

### Adding New Data Sources

1. Create a new ingester in `src/data_ingester.py`:
   ```python
   class NewDataIngestor(DataIngestor):
       def ingest_data(self, data_path: str) -> pd.DataFrame:
           # Implementation for new data source
           pass
   ```

2. Update the factory in `DataIngestorFactory`
3. Add configuration in `config.yaml`

### Adding New Models

1. Create new strategy in `src/model_builder.py`:
   ```python
   class NewModelStrategy(ModelBuildingStrategy):
       def build_model(self, X_train, y_train):
           # Implementation for new model
           pass
   ```

2. Update `ModelBuilderFactory`
3. Add model configuration

### Custom Evaluation Metrics

1. Create evaluation strategy in `src/model_evaluator.py`:
   ```python
   class CustomEvaluationStrategy(ModelEvaluationStrategy):
       def evaluate_model(self, model, X_test, y_test):
           # Custom evaluation implementation
           pass
   ```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```
   Set tree_method: "hist" in config.yaml for CPU training
   ```

2. **Memory Issues**:
   ```
   Reduce batch_size or n_estimators in configuration
   ```

3. **Data Loading Errors**:
   ```
   Verify data path and format in config.yaml
   ```

4. **MLflow Connection Issues**:
   ```
   Check MLflow server status and configuration
   ```

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [ZenML Documentation](https://docs.zenml.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [KDD99 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code patterns
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for production-grade anomaly detection**