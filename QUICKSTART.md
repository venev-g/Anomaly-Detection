# 🛡️ Anomaly Detection System

A production-grade end-to-end network intrusion detection system built with XGBoost and GPU acceleration, designed for the KDD99 dataset.

## 🚀 Quick Start

```bash
# 1. Navigate to the project
cd anomaly-detection-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup ZenML with MLflow
zenml init
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml stack register anomaly_detection_stack -e mlflow_tracker -d mlflow_deployer -a default -o default --set

# 4. Run training pipeline
python run_pipeline.py --config-path config.yaml

# 5. Deploy models
python run_deployment.py 

# 6. Test predictions
python sample_predict.py --config config.yaml
```

## 📖 Complete Guide

For detailed step-by-step instructions, see **[GETTING_STARTED.md](GETTING_STARTED.md)**

## 📊 Key Features

- **GPU-Accelerated XGBoost** - Fast training with CUDA support
- **Dual Classification** - Binary anomaly detection + multiclass attack classification
- **MLOps Pipeline** - ZenML orchestration with MLflow tracking
- **Production Ready** - Comprehensive logging, configuration, and deployment
- **Comprehensive EDA** - Advanced exploratory data analysis

## 🎯 Attack Detection Types

- **Normal Traffic** - Legitimate network connections
- **DoS Attacks** - Denial of Service attacks (neptune, smurf, back, etc.)
- **Probe Attacks** - Surveillance and probing (nmap, portsweep, etc.)
- **R2L Attacks** - Remote to Local attacks (warezclient, guess_passwd, etc.)
- **U2R Attacks** - User to Root attacks (buffer_overflow, rootkit, etc.)

## 📁 Project Structure

```
anomaly-detection-system/
├── 📋 config.yaml              # Configuration
├── 📦 requirements.txt         # Dependencies
├── 🚀 run_pipeline.py         # Training pipeline
├── 🔄 run_deployment.py       # Deployment pipeline
├── 🧪 sample_predict.py       # Prediction testing
├── 📖 GETTING_STARTED.md      # Complete guide
├── 📁 src/                    # Core modules
├── 📁 steps/                  # ZenML pipeline steps
├── 📁 pipelines/              # Training & deployment workflows
├── 📁 analysis/               # EDA notebook and modules
└── 📁 explanations/           # Design pattern examples
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- **Model parameters** (XGBoost settings, GPU usage)
- **Data settings** (file paths, preprocessing options)
- **Training options** (test split, random seed)
- **Evaluation metrics** (threshold, metrics to compute)

## 📈 Model Performance

Expected performance on KDD99 dataset:
- **Binary Classification**: ~98% AUC, ~94% Accuracy
- **Multiclass Classification**: ~92% Overall Accuracy
- **Training Time**: 3-5x faster with GPU acceleration
- **Inference**: <10ms per prediction

## 🛠️ Requirements

- Python 3.8+
- CUDA-compatible GPU (optional)
- 8GB+ RAM
- 5GB storage

## 📚 Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete setup and usage guide
- **[analysis/EDA.ipynb](analysis/EDA.ipynb)** - Comprehensive data analysis
- **[explanations/](explanations/)** - Design pattern examples
- **[README.md](README.md)** - Detailed system documentation

## 🤝 Contributing

1. Follow the existing code patterns
2. Add tests for new functionality  
3. Update documentation
4. Submit pull requests

## 📄 License

MIT License - see LICENSE file for details.

---

**Built for production-grade network anomaly detection** 🔍🚀
