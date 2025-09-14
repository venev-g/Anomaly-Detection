#!/usr/bin/env python3
"""
Test script to verify the deployed MLflow model with properly preprocessed data.
"""
import pandas as pd
import numpy as np
import requests
import json
from zenml.client import Client
from zenml.integrations.mlflow.services import MLFlowDeploymentService

def load_sample_data():
    """Load and preprocess sample data exactly like the training pipeline."""
    print("Loading sample data...")
    
    # Define column names from config
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]
    
    # Load a few rows from the data file
    try:
        df = pd.read_csv('data/kddcup.data.corrected', header=None, names=column_names, nrows=5)
        print(f"Loaded {len(df)} sample rows")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create synthetic sample data if file not found
        return create_synthetic_sample()

def create_synthetic_sample():
    """Create a synthetic sample that matches the expected format."""
    print("Creating synthetic sample data...")
    
    # Create a sample with typical network connection values
    sample_data = {
        'duration': [0],
        'protocol_type': ['tcp'],
        'service': ['http'],
        'flag': ['SF'],
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
        'dst_host_srv_rerror_rate': [0.0],
        'label': ['normal.']
    }
    
    return pd.DataFrame(sample_data)

def preprocess_data(df):
    """Preprocess data exactly like the training pipeline."""
    print("Preprocessing data...")
    
    # Define categorical variables from config
    categorical_vars = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    
    # Remove label column for prediction
    features_df = df.drop('label', axis=1)
    
    # One-hot encode categorical variables
    cat_data = pd.get_dummies(features_df[categorical_vars])
    
    # Get numeric variables
    numeric_vars = list(set(features_df.columns.values.tolist()) - set(categorical_vars))
    numeric_data = features_df[numeric_vars].copy()
    
    # Combine numerical and categorical data
    processed_features = pd.concat([numeric_data, cat_data], axis=1)
    
    print(f"Processed features shape: {processed_features.shape}")
    print(f"Feature columns: {list(processed_features.columns)}")
    
    return processed_features

def test_model_service():
    """Test the deployed MLflow model service."""
    print("Testing MLflow model deployment...")
    
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
    
    # Load and preprocess sample data
    sample_df = load_sample_data()
    processed_data = preprocess_data(sample_df.iloc[:1])  # Take first row
    
    # Test different data formats
    formats_to_test = [
        ("DataFrame as JSON", processed_data.to_json(orient='records')),
        ("DataFrame split format", {
            "columns": processed_data.columns.tolist(),
            "data": processed_data.values.tolist()
        }),
        ("MLflow dataframe_split format", {
            "dataframe_split": {
                "columns": processed_data.columns.tolist(),
                "data": processed_data.values.tolist()
            }
        }),
        ("Simple instances format", {
            "instances": processed_data.values.tolist()
        })
    ]
    
    for format_name, data_payload in formats_to_test:
        print(f"\nTesting format: {format_name}")
        try:
            response = requests.post(
                service.prediction_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data_payload) if not isinstance(data_payload, str) else data_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {format_name} SUCCESS!")
                print(f"Prediction result: {result}")
                return True
            else:
                print(f"❌ {format_name} FAILED: {response.status_code} - {response.text[:500]}")
                
        except Exception as e:
            print(f"❌ {format_name} ERROR: {str(e)}")
    
    print("\n❌ All formats failed!")
    return False

if __name__ == "__main__":
    success = test_model_service()
    if success:
        print("\n🎉 Model deployment test PASSED!")
    else:
        print("\n💥 Model deployment test FAILED!")