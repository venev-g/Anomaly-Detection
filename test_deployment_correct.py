#!/usr/bin/env python3
"""
Corrected test script with the exact feature order that the model expects.
"""
import pandas as pd
import numpy as np
import requests
import json
import yaml
from zenml.client import Client

def get_expected_feature_order():
    """Get the exact feature order that the model expects."""
    # This is the exact order from the preprocessing pipeline
    return [
        'num_access_files', 'dst_host_srv_serror_rate', 'dst_host_srv_rerror_rate',
        'dst_host_srv_count', 'same_srv_rate', 'num_shells', 'num_outbound_cmds',
        'srv_serror_rate', 'dst_host_same_src_port_rate', 'diff_srv_rate',
        'serror_rate', 'duration', 'hot', 'dst_host_diff_srv_rate', 'urgent',
        'srv_rerror_rate', 'dst_host_count', 'dst_host_rerror_rate',
        'dst_host_same_srv_rate', 'dst_host_serror_rate', 'root_shell',
        'num_failed_logins', 'dst_host_srv_diff_host_rate', 'src_bytes',
        'count', 'su_attempted', 'num_file_creations', 'srv_diff_host_rate',
        'rerror_rate', 'dst_bytes', 'num_root', 'srv_count', 'wrong_fragment',
        'num_compromised', 'land', 'logged_in', 'is_host_login', 'is_guest_login',
        'protocol_type_tcp', 'service_http', 'flag_SF'
    ]

def create_sample_data():
    """Create sample data in the exact format expected by the model."""
    print("Creating sample data in expected format...")
    
    feature_names = get_expected_feature_order()
    
    # Create a sample with realistic values
    sample_values = {
        # Numeric features
        'num_access_files': 0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0,
        'dst_host_srv_count': 1,
        'same_srv_rate': 1.0,
        'num_shells': 0,
        'num_outbound_cmds': 0,
        'srv_serror_rate': 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'diff_srv_rate': 0.0,
        'serror_rate': 0.0,
        'duration': 0,
        'hot': 0,
        'dst_host_diff_srv_rate': 0.0,
        'urgent': 0,
        'srv_rerror_rate': 0.0,
        'dst_host_count': 255,
        'dst_host_rerror_rate': 0.0,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_serror_rate': 0.0,
        'root_shell': 0,
        'num_failed_logins': 0,
        'dst_host_srv_diff_host_rate': 0.0,
        'src_bytes': 215,
        'count': 1,
        'su_attempted': 0,
        'num_file_creations': 0,
        'srv_diff_host_rate': 0.0,
        'rerror_rate': 0.0,
        'dst_bytes': 45076,
        'num_root': 0,
        'srv_count': 1,
        'wrong_fragment': 0,
        'num_compromised': 0,
        'land': 0,
        'logged_in': 1,
        'is_host_login': 0,
        'is_guest_login': 0,
        # One-hot encoded categorical features
        'protocol_type_tcp': True,
        'service_http': True,
        'flag_SF': True
    }
    
    # Create DataFrame with exact feature order
    data_row = []
    for feature in feature_names:
        data_row.append(sample_values[feature])
    
    # Convert boolean to int for consistency
    data_row = [int(x) if isinstance(x, bool) else x for x in data_row]
    
    return pd.DataFrame([data_row], columns=feature_names)

def test_model_service():
    """Test the deployed MLflow model service with correct data format."""
    print("Testing MLflow model deployment with correct feature order...")
    
    # Get ZenML client
    client = Client()
    
    # Find the deployed service
    services = client.active_stack.model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model"
    )
    
    if not services:
        print("❌ No deployed service found!")
        return False
    
    service = services[0]
    print(f"Found deployed service: {service}")
    print(f"Prediction URL: {service.prediction_url}")
    
    # Create sample data with correct feature order
    sample_data = create_sample_data()
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Feature columns (first 10): {list(sample_data.columns[:10])}")
    
    # Test with the MLflow dataframe_split format
    payload = {
        "dataframe_split": {
            "columns": sample_data.columns.tolist(),
            "data": sample_data.values.tolist()
        }
    }
    
    print(f"\nTesting with proper dataframe_split format...")
    try:
        response = requests.post(
            service.prediction_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction SUCCESS!")
            print(f"Response: {result}")
            
            # Try to interpret the result
            if isinstance(result, list) and len(result) > 0:
                prediction = result[0]
                if isinstance(prediction, (int, float)):
                    print(f"Predicted class: {prediction}")
                    if prediction == 0:
                        print("🟢 Prediction: NORMAL connection")
                    else:
                        print("🔴 Prediction: ANOMALY detected")
                elif isinstance(prediction, list):
                    print(f"Prediction probabilities: {prediction}")
            
            return True
        else:
            print(f"❌ Request FAILED: {response.status_code}")
            print(f"Error response: {response.text[:1000]}")
            return False
            
    except Exception as e:
        print(f"❌ Request ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_service()
    if success:
        print("\n🎉 Model deployment test PASSED!")
        print("The MLflow model service is working correctly!")
    else:
        print("\n💥 Model deployment test FAILED!")