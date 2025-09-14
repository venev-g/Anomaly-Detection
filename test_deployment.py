#!/usr/bin/env python3
"""
Test script to verify the MLflow model deployment is working correctly.
"""

import logging
import requests
import json
import pandas as pd
import numpy as np
from zenml.client import Client
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_model_deployment():
    """Test the deployed model with sample data."""
    try:
        logging.info("Testing MLflow model deployment...")
        
        # Get the MLflow model deployer
        client = Client()
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()
        
        # Find the deployed service
        services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name="model"
        )
        
        if not services:
            logging.error("No deployed services found!")
            return False
            
        service = services[0]
        logging.info(f"Found deployed service: {service}")
        
        # Get the prediction URL
        prediction_url = service.get_prediction_url()
        logging.info(f"Prediction URL: {prediction_url}")
        
        # Create sample data (should match the model's expected input format)
        # Let's create a simple test with the right number of features
        sample_data = pd.DataFrame({
            'duration': [0],
            'protocol_type': [1],
            'service': [1],
            'flag': [1],
            'src_bytes': [100],
            'dst_bytes': [200],
            'land': [0],
            'wrong_fragment': [0],
            'urgent': [0],
            'hot': [0],
            'num_failed_logins': [0],
            'logged_in': [1],
            'num_compromised': [0],
            'root_shell': [0],
            'su_attempted': [0],
            'num_root': [0],
            'num_file_creations': [0],
            'num_shells': [0],
            'num_access_files': [0],
            'num_outbound_cmds': [0],
            'is_host_login': [0],
            'is_guest_login': [0],
            'count': [1],
            'srv_count': [1],
            'serror_rate': [0.0],
            'srv_serror_rate': [0.0],
            'rerror_rate': [0.0],
            'srv_rerror_rate': [0.0],
            'same_srv_rate': [1.0],
            'diff_srv_rate': [0.0],
            'srv_diff_host_rate': [0.0],
            'dst_host_count': [255],
            'dst_host_srv_count': [1],
            'dst_host_same_srv_rate': [1.0],
            'dst_host_diff_srv_rate': [0.0],
            'dst_host_same_src_port_rate': [0.0],
            'dst_host_srv_diff_host_rate': [0.0],
            'dst_host_serror_rate': [0.0],
            'dst_host_srv_serror_rate': [0.0],
            'dst_host_rerror_rate': [0.0],
            'dst_host_srv_rerror_rate': [0.0]
        })
        
        logging.info(f"Sample data shape: {sample_data.shape}")
        logging.info(f"Sample data columns: {list(sample_data.columns)}")
        
        # Test different data formats
        formats_to_test = [
            ("DataFrame", sample_data),
            ("Numpy array", sample_data.values),
            ("JSON - instances", {"instances": sample_data.values.tolist()}),
            ("JSON - dataframe_split", {
                "dataframe_split": {
                    "columns": sample_data.columns.tolist(),
                    "data": sample_data.values.tolist()
                }
            })
        ]
        
        for format_name, data in formats_to_test:
            try:
                logging.info(f"\nTesting format: {format_name}")
                
                if format_name.startswith("JSON"):
                    # Make HTTP request directly
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(prediction_url, data=json.dumps(data), headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        logging.info(f"✅ {format_name} SUCCESS: {result}")
                        return True
                    else:
                        logging.warning(f"❌ {format_name} FAILED: {response.status_code} - {response.text}")
                else:
                    # Use service predict method
                    result = service.predict(data)
                    logging.info(f"✅ {format_name} SUCCESS: {result}")
                    return True
                    
            except Exception as e:
                logging.warning(f"❌ {format_name} FAILED: {str(e)}")
                continue
        
        logging.error("All formats failed!")
        return False
        
    except Exception as e:
        logging.error(f"Error testing deployment: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_deployment()
    if success:
        print("\n🎉 Model deployment test PASSED!")
    else:
        print("\n❌ Model deployment test FAILED!")