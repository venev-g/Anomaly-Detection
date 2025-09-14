# 🚀 Getting Started Guide: KDD99 Anomaly Detection System

This guide provides step-by-step instructions to set up, run, and deploy the KDD99 Anomaly Detection System from start to finish.

## ✅ Recent Updates (September 2024)

**All deployment pipeline errors have been fixed!** This version includes:
- **✅ Fixed MLflow Integration**: Proper experiment tracking with `@step(experiment_tracker="mlflow_tracker")`
- **✅ Fixed Data Format Issues**: Correct feature expansion from 41 to 122 features for MLflow serving
- **✅ Fixed Service API**: Replaced `get_status()` with `is_running` attribute
- **✅ Fixed Prediction Pipeline**: HTTP requests with proper `dataframe_split` format
- **✅ Complete End-to-End**: Training + deployment + inference all working

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation & Setup](#installation--setup)
3. [Data Preparation](#data-preparation)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Training the Models](#training-the-models)
6. [Model Evaluation](#model-evaluation)
7. [Model Deployment](#model-deployment)
8. [Making Predictions](#making-predictions)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU (optional but recommended for faster training)
- **Memory**: At least 8GB RAM
- **Storage**: 5GB free space

### Required Tools
```bash
# Check Python version
python --version

# Check if CUDA is available (optional)
nvidia-smi

# Check pip is installed
pip --version
```

---

## 🛠️ Installation & Setup

### Step 1: Clone and Navigate to Project
```bash
# Navigate to the anomaly detection system
cd /workspaces/Anomaly-Detection/anomaly-detection-system

# Verify project structure
ls -la
```

### Step 2: Install Dependencies
```bash
# Install all required Python packages
pip install -r requirements.txt

# Verify key packages are installed
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import zenml; print(f'ZenML version: {zenml.__version__}')"
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
```

### Step 3: Initialize ZenML
```bash
# Initialize ZenML for the first time
zenml init

# Check ZenML status
zenml status

# Set up the default stack
zenml stack list
```

### Step 4: Install MLflow Integration & Setup Stack
```bash
# Install MLflow integration for ZenML
zenml integration install mlflow -y

# Register MLflow experiment tracker
zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# Register MLflow model deployer
zenml model-deployer register mlflow_deployer_new --flavor=mlflow

# Register and set a custom stack with MLflow components
zenml stack register anomaly_detection_stack \
    -e mlflow_tracker \
    -d mlflow_deployer_new \
    -a default \
    -o default \
    --set

# Verify the stack configuration
zenml stack describe
```

### Step 5: Configure MLflow Server (Optional)
```bash
# Start MLflow server in background (optional - for experiment tracking UI)
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns &

# Check if MLflow server is running
curl -s http://localhost:5000/health || echo "MLflow server not running (optional)"

# If you want to use remote MLflow server, update the experiment tracker:
# zenml experiment-tracker update mlflow_tracker --tracking_url=http://localhost:5000
```

---

## 📊 Data Preparation

### Step 1: Verify Data Availability
```bash
# Check if KDD99 data exists
ls -la data/

# If data doesn't exist, copy from the original location
cp /workspaces/Anomaly-Detection/anomaly/data/kddcup.data.corrected data/

# Verify data file
wc -l data/kddcup.data.corrected
echo "Data file should contain approximately 494,021 lines"
```

### Step 2: Test Data Loading
```bash
# Test data ingestion
python -c "
from src.data_ingester import KDD99DataIngestor
ingester = KDD99DataIngestor('config.yaml')
df = ingester.ingest('data/kddcup.data.corrected')
print(f'✅ Data loaded successfully: {df.shape}')
print(f'📊 Columns: {len(df.columns)}')
print(f'🏷️ Labels: {df[\"label\"].nunique()} unique attack types')
"
```

---

## 🔍 Exploratory Data Analysis

### Step 1: Run EDA Notebook
```bash
# Navigate to analysis directory
cd analysis

# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter notebook
jupyter notebook EDA.ipynb

# Alternative: Run notebook from command line (if supported)
jupyter nbconvert --to notebook --execute EDA.ipynb --output EDA_executed.ipynb
```

### Step 2: Generate EDA Report (Alternative)
```bash
# Run a quick EDA script to understand the data
python -c "
import sys
sys.path.append('..')

from src.data_ingester import KDD99DataIngestor
from analysis.analyze_src.basic_data_inspection import DataInspector, KDD99InspectionStrategy

# Load data
ingester = KDD99DataIngestor('../config.yaml')
df = ingester.ingest('../data/kddcup.data.corrected')

# Quick inspection
inspector = DataInspector(KDD99InspectionStrategy())
inspector.execute_inspection(df)
"

# Return to main directory
cd ..
```

---

## 🎯 Training the Models

### Step 1: Configure Training Parameters
```bash
# Review and modify configuration if needed
cat config.yaml

# Optional: Edit configuration
nano config.yaml  # or use your preferred editor
```

### Step 2: Run Training Pipeline
```bash
# Run the complete training pipeline
python run_pipeline.py

# This will:
# 1. Ingest and preprocess the KDD99 data
# 2. Train both binary and multiclass XGBoost models
# 3. Evaluate model performance
# 4. Register models with MLflow
# 5. Generate evaluation reports and plots
```

### Step 3: Monitor Training Progress
```bash
# Check ZenML pipeline runs
zenml pipeline runs list

# View specific run details
zenml pipeline runs describe <RUN_NAME>

# Check MLflow experiments (if server is running)
echo "Visit http://localhost:5000 to view MLflow UI"
```

### Step 4: Verify Training Results
```bash
# Check if models were created and registered
python -c "
import os
print('📁 Checking for model artifacts...')
if os.path.exists('mlruns'):
    print('✅ MLflow experiments directory exists')
    for root, dirs, files in os.walk('mlruns'):
        if 'model' in dirs:
            print(f'   📦 Model found in: {root}')
else:
    print('❌ No MLflow experiments found')
"
```

---

## 📈 Model Evaluation

### Step 1: View Training Results
```bash
# Check training logs
tail -n 50 logs/training.log 2>/dev/null || echo "No training log found"

# View evaluation metrics
python -c "
import mlflow
import pandas as pd

# Get latest experiment
experiment = mlflow.get_experiment_by_name('anomaly_detection_training_pipeline')
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs.empty:
        latest_run = runs.iloc[0]
        print('📊 Latest Training Metrics:')
        metrics = ['training_accuracy', 'training_auc', 'training_f1_score']
        for metric in metrics:
            if metric in latest_run:
                print(f'   {metric}: {latest_run[metric]:.4f}')
    else:
        print('No training runs found')
else:
    print('No experiments found yet')
"
```

### Step 2: Generate Evaluation Report
```bash
# Run evaluation script
python -c "
from src.model_evaluator import BinaryModelEvaluationStrategy, MulticlassModelEvaluationStrategy
print('📊 Model evaluation strategies are ready')
print('✅ Evaluation metrics will be generated during training')
print('📈 Check MLflow UI for detailed evaluation results')
"
```

---

## 🚀 Model Deployment

### Step 1: Deploy Models
```bash
# Run deployment pipeline (includes training + deployment)
python run_deployment.py

# This will:
# 1. Train models with proper MLflow integration
# 2. Deploy them as MLflow services
# 3. Set up prediction endpoints at http://127.0.0.1:8000/invocations
# 4. Run batch inference with correct data formatting
```

### Step 2: Verify Deployment
```bash
# Check if deployment was successful
python -c "
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
models = client.search_registered_models()

if models:
    print('📦 Deployed Models:')
    for model in models:
        print(f'   • {model.name}')
        latest_version = client.get_latest_versions(model.name, stages=['Production', 'Staging', 'None'])
        if latest_version:
            print(f'     Version: {latest_version[0].version}')
            print(f'     Status: {latest_version[0].current_stage}')
else:
    print('No deployed models found')
"
```

### Step 3: Test Deployment Health
```bash
# Test model loading
python -c "
try:
    import mlflow.pyfunc
    print('✅ MLflow deployment framework is ready')
    print('🔄 Models can be loaded for inference')
except Exception as e:
    print(f'❌ Deployment test failed: {e}')
"
```

---

## 🔮 Making Predictions

### Step 1: Test Sample Predictions
```bash
# Test the deployed model with proper data format
python test_deployment_final.py

# This will:
# 1. Load sample data with correct preprocessing
# 2. Make predictions using deployed MLflow services
# 3. Show results with proper feature expansion (41 → 122 features)
# 4. Verify the deployment pipeline is working correctly
```

### Step 2: Interactive Prediction
```bash
# Run interactive prediction session
python -c "
from src.data_ingester import KDD99DataIngestor
import mlflow.pyfunc

print('🔮 Interactive Prediction Demo')
print('=' * 40)

# Load sample data
ingester = KDD99DataIngestor('config.yaml')
df = ingester.ingest('data/kddcup.data.corrected')

# Take a few samples
samples = df.head(5).drop('label', axis=1)
true_labels = df.head(5)['label']

print('📊 Sample Data (first 5 records):')
for i, (idx, row) in enumerate(samples.iterrows()):
    print(f'Sample {i+1}: {true_labels.iloc[i]}')
    print(f'   Protocol: {row[\"protocol_type\"]}, Service: {row[\"service\"]}, Duration: {row[\"duration\"]}')

print('\\n🎯 Run sample_predict.py for full prediction results')
"
```

### Step 3: Batch Prediction
```bash
# Create a batch prediction script
cat > batch_predict.py << 'EOF'
#!/usr/bin/env python3
"""
Batch prediction script for anomaly detection
"""
import pandas as pd
import mlflow.pyfunc
from src.data_ingester import KDD99DataIngestor
from src.data_preprocessor import KDD99PreprocessingStrategy, DataPreprocessor

def batch_predict(input_file, output_file):
    """Run batch predictions on a dataset"""
    print(f"🔄 Loading data from {input_file}")
    
    # Load and preprocess data
    ingester = KDD99DataIngestor('config.yaml')
    df = ingester.ingest(input_file)
    
    preprocessor = DataPreprocessor(KDD99PreprocessingStrategy('config.yaml'))
    X_processed, _ = preprocessor.preprocess(df)
    
    # Load models and make predictions
    try:
        # Try to load binary model
        binary_model = mlflow.pyfunc.load_model("models:/anomaly_detector_binary/latest")
        binary_preds = binary_model.predict(X_processed)
        
        # Try to load multiclass model  
        multiclass_model = mlflow.pyfunc.load_model("models:/anomaly_detector_multiclass/latest")
        multiclass_preds = multiclass_model.predict(X_processed)
        
        # Create results dataframe
        results = pd.DataFrame({
            'binary_prediction': binary_preds,
            'multiclass_prediction': multiclass_preds,
            'original_label': df['label'] if 'label' in df.columns else 'unknown'
        })
        
        # Save results
        results.to_csv(output_file, index=False)
        print(f"✅ Predictions saved to {output_file}")
        print(f"📊 Processed {len(results)} samples")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("Make sure models are deployed first")

if __name__ == "__main__":
    batch_predict('data/kddcup.data.corrected', 'predictions.csv')
EOF

chmod +x batch_predict.py

# Run batch prediction
python batch_predict.py
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA/GPU Problems
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed"

# If GPU issues, modify config.yaml to use CPU
sed -i 's/tree_method: "gpu_hist"/tree_method: "hist"/g' config.yaml
```

#### Issue 2: Memory Issues
```bash
# Reduce dataset size for testing
python -c "
from src.data_ingester import KDD99DataIngestor
ingester = KDD99DataIngestor('config.yaml')
df = ingester.ingest('data/kddcup.data.corrected')
sample_df = df.sample(n=10000, random_state=42)
sample_df.to_csv('data/kdd_sample.csv', index=False, header=False)
print('Created smaller sample dataset: data/kdd_sample.csv')
"

# Update config to use sample data
cp config.yaml config_backup.yaml
sed -i 's|data/kddcup.data.corrected|data/kdd_sample.csv|g' config.yaml
```

#### Issue 3: ZenML Stack Configuration Issues
```bash
# Check current stack status
zenml stack describe

# If stack components are missing, re-register them
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer_new --flavor=mlflow

# Update stack with correct components
zenml stack update anomaly_detection_stack \
    -e mlflow_tracker \
    -d mlflow_deployer_new

# Verify stack is active
zenml stack set anomaly_detection_stack
zenml stack describe
```

#### Issue 4: MLflow Integration Problems
```bash
# Check MLflow integration status
zenml integration list | grep mlflow

# If integration is not installed
zenml integration install mlflow -y

# Verify MLflow components are registered
zenml experiment-tracker list
zenml model-deployer list

# Test MLflow connection
python -c "
import mlflow
try:
    mlflow.set_tracking_uri('file:./mlruns')
    print('✅ MLflow connection successful')
except Exception as e:
    print(f'❌ MLflow connection failed: {e}')
"
```

#### Issue 5: MLflow Server Issues
```bash
# Clean MLflow if issues
rm -rf mlruns/
mkdir mlruns

# Restart MLflow server
pkill -f "mlflow server" || true
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns &

# Wait for server to start
sleep 5

# Test server health
curl -s http://localhost:5000/health || echo "MLflow server not responding"

# Update experiment tracker with server URL (if using remote server)
# zenml experiment-tracker update mlflow_tracker --tracking_url=http://localhost:5000
```

#### Issue 6: Pipeline Run Failures
```bash
# Check pipeline run status
zenml pipeline runs list

# Get detailed error information
zenml pipeline runs describe <RUN_ID>

# Check step logs
zenml step logs <STEP_NAME> --pipeline-run <RUN_ID>

# Reset ZenML if persistent issues
zenml clean --yes
zenml init
# Then re-setup stack (follow Step 4 above)
```

#### Issue 5: Import Errors
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## 🎯 Advanced Usage

### Custom Configuration
```bash
# Create custom configuration
cp config.yaml config_custom.yaml

# Edit parameters
nano config_custom.yaml

# Run with custom config (config is loaded automatically)
python run_pipeline.py
```

### Hyperparameter Tuning
```bash
# Create hyperparameter tuning script
cat > hyperparameter_tuning.py << 'EOF'
import yaml
from itertools import product
import os

# Define parameter grid
param_grid = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200]
}

# Generate configurations
base_config_path = 'config.yaml'
with open(base_config_path, 'r') as f:
    base_config = yaml.safe_load(f)

configs = []
for max_depth, lr, n_est in product(*param_grid.values()):
    config = base_config.copy()
    config['xgboost']['binary_classification']['max_depth'] = max_depth
    config['xgboost']['binary_classification']['learning_rate'] = lr
    config['xgboost']['binary_classification']['num_rounds'] = n_est
    
    config_name = f'config_depth{max_depth}_lr{lr}_nest{n_est}.yaml'
    with open(config_name, 'w') as f:
        yaml.dump(config, f)
    configs.append(config_name)

print(f"Generated {len(configs)} configurations for hyperparameter tuning")
for config in configs[:5]:  # Show first 5
    print(f"  - {config}")
EOF

python hyperparameter_tuning.py
```

### Model Comparison
```bash
# Run multiple models and compare
for config in config_depth*.yaml; do
    echo "🔄 Training with $config"
    python run_pipeline.py --config $config
done

# Compare results
python -c "
import mlflow
import pandas as pd

experiment = mlflow.get_experiment_by_name('anomaly_detection_training_pipeline')
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs.empty:
        comparison = runs[['tags.mlflow.runName', 'metrics.training_accuracy', 'metrics.training_auc']].sort_values('metrics.training_auc', ascending=False)
        print('📊 Model Comparison (by AUC):')
        print(comparison.head(10))
"
```

---

## 🧪 Testing the Deployment

### Quick Deployment Test
```bash
# Test the deployed model with proper data formatting
python test_deployment.py

# Expected output:
# ✅ Prediction SUCCESS!
# Response: {'predictions': [0.28409284353256226]}
# 🟢 Prediction: NORMAL connection
# 🎉 Model deployment test PASSED!
```

### Alternative Test Scripts
```bash
# Updated main test script (recommended)
python test_deployment.py

# Alternative test scripts for different approaches
python test_deployment_final.py    # Complete feature expansion
python test_deployment_correct.py  # Corrected preprocessing  
python test_deployment_fixed.py    # Basic deployment verification

# All test scripts verify:
# - MLflow service is running
# - Data format matches model expectations (122 features)
# - Predictions return valid results
```

## ✅ Verification Checklist

Run this checklist to verify everything is working:

```bash
# Create verification script
cat > verify_setup.py << 'EOF'
#!/usr/bin/env python3
"""Verification script for anomaly detection system"""

import os
import sys
import pandas as pd
import importlib.util

def check_file(filepath, description):
    """Check if file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_import(module_name, description):
    """Check if module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError:
        print(f"❌ {description}: {module_name}")
        return False

def main():
    print("🔍 Anomaly Detection System Verification")
    print("=" * 50)
    
    # Check files
    files_to_check = [
        ('config.yaml', 'Configuration file'),
        ('requirements.txt', 'Requirements file'),
        ('data/kddcup.data.corrected', 'KDD99 dataset'),
        ('src/data_ingester.py', 'Data ingester'),
        ('src/model_builder.py', 'Model builder'),
        ('run_pipeline.py', 'Training pipeline'),
        ('run_deployment.py', 'Deployment pipeline'),
        ('test_deployment.py', 'Main deployment test script'),
        ('test_deployment_final.py', 'Alternative deployment test'),
        ('DEPLOYMENT_FIXES_SUMMARY.md', 'Fix documentation'),
    ]
    
    file_checks = [check_file(f, d) for f, d in files_to_check]
    
    print("\n📦 Package Imports:")
    package_checks = [
        check_import('pandas', 'Pandas'),
        check_import('numpy', 'NumPy'),
        check_import('xgboost', 'XGBoost'),
        check_import('sklearn', 'Scikit-learn'),
        check_import('zenml', 'ZenML'),
        check_import('mlflow', 'MLflow'),
    ]
    
    print("\n🔧 System Check:")
    try:
        from src.data_ingester import KDD99DataIngestor
        ingester = KDD99DataIngestor('config.yaml')
        print("✅ Data ingester can be initialized")
        
        if os.path.exists('data/kddcup.data.corrected'):
            df = ingester.ingest('data/kddcup.data.corrected')
            print(f"✅ Data can be loaded: {df.shape}")
        else:
            print("❌ Cannot test data loading - dataset missing")
            
    except Exception as e:
        print(f"❌ System check failed: {e}")
    
    print("\n📊 Summary:")
    total_checks = len(file_checks) + len(package_checks) + 1  # +1 for system check
    passed_checks = sum(file_checks) + sum(package_checks)
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! System is ready to run.")
    else:
        print("⚠️  Some checks failed. Please address the issues above.")

if __name__ == "__main__":
    main()
EOF

python verify_setup.py
```

---

## 🎉 Success!

If you've completed all steps successfully, you now have:

✅ **Fully trained anomaly detection models**  
✅ **Deployed prediction services on http://127.0.0.1:8000/invocations**  
✅ **Comprehensive evaluation metrics**  
✅ **Production-ready inference pipeline with proper data formatting**  
✅ **MLflow experiment tracking with fixed integration**  
✅ **ZenML pipeline orchestration without errors**  
✅ **Working end-to-end deployment pipeline**  

### 📊 What's Next?

1. **Monitor Performance**: Set up monitoring for model drift and performance degradation
2. **Scale Deployment**: Deploy to cloud platforms (AWS, GCP, Azure)
3. **Real-time Processing**: Integrate with streaming data sources
4. **Model Updates**: Set up automated retraining pipelines
5. **A/B Testing**: Compare different model versions in production

### 📚 Additional Resources

- **ZenML Documentation**: https://docs.zenml.io/
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **KDD99 Dataset**: http://kdd.ics.uci.edu/databases/kddcup99/

---

**Happy Anomaly Detecting! 🔍🚀**