# Remote Access Guide for ZenML and MLflow

This guide explains how to access your ZenML anomaly detection pipeline remotely via SSH port forwarding.

## Current Status ✅

- **ZenML CLI**: Fully functional locally on the remote server
- **MLflow UI**: Running on port 5000, accessible remotely
- **Pipeline Status**: Multiple successful training and deployment runs
- **Model Deployment**: XGBoost models successfully deployed via MLflow

## Remote Access Setup

### Prerequisites
- SSH access to your remote server
- Local machine with web browser
- ZenML and MLflow running on remote server

### Step 1: SSH Port Forwarding

From your **local machine**, establish SSH connection with port forwarding:

```bash
# Forward MLflow UI (port 5000) to your local machine
ssh -L 5000:localhost:5000 username@your-remote-server

# Optional: Forward additional ports if needed
# ssh -L 5000:localhost:5000 -L 8080:localhost:8080 username@your-remote-server
```

### Step 2: Access MLflow UI

Once connected via SSH with port forwarding:

1. Open your local web browser
2. Navigate to: `http://localhost:5000`
3. You should see the MLflow UI with your experiments and models

### Step 3: Use ZenML via SSH

ZenML works perfectly via CLI when you're SSH'd into the remote server:

```bash
# Check pipeline status
zenml pipeline runs list

# View pipeline details
zenml pipeline runs describe <run-id>

# Check model deployments
zenml model deployer list

# View stack configuration
zenml stack describe

# List available models
zenml model list
```

## Working Pipeline Commands

### View Recent Successful Runs
```bash
# See all pipeline runs (successful ones marked with ✅)
zenml pipeline runs list

# Get details of a specific successful run
zenml pipeline runs describe <run-id>
```

### Test Model Deployment
```bash
# Test the deployment using the working test script
python test_deployment.py

# Alternative: Test deployment with corrected version
python test_deployment_correct.py
```

### Run New Pipelines
```bash
# Training pipeline
python run_pipeline.py

# Deployment pipeline  
python run_deployment.py
```

## Troubleshooting

### MLflow UI Not Loading
- Verify SSH port forwarding is active
- Check that MLflow server is running: `curl http://localhost:5000/health`
- Ensure no firewall blocking port 5000

### ZenML Dashboard Alternative
While the ZenML web dashboard isn't currently accessible remotely, you have full functionality via:
- **CLI commands** (as shown above)
- **MLflow UI** for experiment tracking and model management
- **Jupyter notebooks** for analysis (port forward 8888 if needed)

### SSH Connection Tips
```bash
# Keep SSH connection alive
ssh -o ServerAliveInterval=60 -L 5000:localhost:5000 username@server

# Run in background
ssh -f -N -L 5000:localhost:5000 username@server
```

## Verification Steps

1. **Test SSH Port Forwarding**:
   ```bash
   # On local machine after SSH connection
   curl http://localhost:5000/health
   # Should return: OK
   ```

2. **Verify ZenML Status**:
   ```bash
   # On remote server via SSH
   zenml status
   # Should show: Connected to the local ZenML database
   ```

3. **Check Recent Successful Runs**:
   ```bash
   zenml pipeline runs list --size 5
   # Should show recent ✅ successful runs
   ```

## Success Indicators

✅ **Working Components**:
- ZenML pipeline orchestration
- MLflow experiment tracking  
- XGBoost model training and deployment
- Feature engineering (41→122 features)
- Model serving endpoints
- Anomaly detection predictions

✅ **Recent Achievements**:
- Fixed MLflow model logging issues
- Resolved matplotlib figure materializer warnings
- Fixed prediction service loader problems
- Updated documentation to reflect fixes
- Verified deployment pipeline end-to-end

## Next Steps

1. **Access MLflow UI**: Use SSH port forwarding to view experiments and models
2. **Monitor Pipelines**: Use ZenML CLI to track pipeline runs and status
3. **Deploy New Models**: Run training and deployment pipelines as needed
4. **Analyze Results**: Use analysis notebooks with port forwarding if needed

For any issues, check the logs and use the CLI commands shown above to debug and monitor your ML pipelines.