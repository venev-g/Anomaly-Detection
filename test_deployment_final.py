#!/usr/bin/env python3
"""
Final corrected test script with the alphabetical feature order that MLflow expects.
"""
import pandas as pd
import numpy as np
import requests
import json
from zenml.client import Client

def get_mlflow_expected_features():
    """Get the exact feature order that MLflow model expects (alphabetical)."""
    # From the error message, MLflow expects features in alphabetical order
    return [
        'count', 'diff_srv_rate', 'dst_bytes', 'dst_host_count', 'dst_host_diff_srv_rate',
        'dst_host_rerror_rate', 'dst_host_same_src_port_rate', 'dst_host_same_srv_rate',
        'dst_host_serror_rate', 'dst_host_srv_count', 'dst_host_srv_diff_host_rate',
        'dst_host_srv_rerror_rate', 'dst_host_srv_serror_rate', 'duration',
        'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR',
        'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH',
        'hot', 'is_guest_login', 'is_host_login', 'land', 'logged_in',
        'num_access_files', 'num_compromised', 'num_failed_logins', 'num_file_creations',
        'num_outbound_cmds', 'num_root', 'num_shells', 'protocol_type_icmp',
        'protocol_type_tcp', 'protocol_type_udp', 'rerror_rate', 'root_shell',
        'same_srv_rate', 'serror_rate', 'service_IRC', 'service_X11', 'service_Z39_50',
        'service_aol', 'service_auth', 'service_bgp', 'service_courier',
        'service_csnet_ns', 'service_ctf', 'service_daytime', 'service_discard',
        'service_domain', 'service_domain_u', 'service_echo', 'service_eco_i',
        'service_ecr_i', 'service_efs', 'service_exec', 'service_finger',
        'service_ftp', 'service_ftp_data', 'service_gopher', 'service_harvest',
        'service_hostnames', 'service_http', 'service_http_2784', 'service_http_443',
        'service_http_8001', 'service_imap4', 'service_iso_tsap', 'service_klogin',
        'service_kshell', 'service_ldap', 'service_link', 'service_login',
        'service_mtp', 'service_name', 'service_netbios_dgm', 'service_netbios_ns',
        'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp',
        'service_ntp_u', 'service_other', 'service_pm_dump', 'service_pop_2',
        'service_pop_3', 'service_printer', 'service_private', 'service_red_i',
        'service_remote_job', 'service_rje', 'service_shell', 'service_smtp',
        'service_sql_net', 'service_ssh', 'service_sunrpc', 'service_supdup',
        'service_systat', 'service_telnet', 'service_tftp_u', 'service_tim_i',
        'service_time', 'service_urh_i', 'service_urp_i', 'service_uucp',
        'service_uucp_path', 'service_vmnet', 'service_whois', 'src_bytes',
        'srv_count', 'srv_diff_host_rate', 'srv_rerror_rate', 'srv_serror_rate',
        'su_attempted', 'urgent', 'wrong_fragment'
    ]

def create_sample_with_all_features():
    """Create sample data with ALL possible one-hot encoded features."""
    print("Creating sample data with complete feature set...")
    
    feature_names = get_mlflow_expected_features()
    
    # Initialize all features to 0
    sample_values = {}
    for feature in feature_names:
        if 'rate' in feature or feature in ['same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate']:
            sample_values[feature] = 0.0  # Float for rate features
        else:
            sample_values[feature] = 0    # Integer for count features and binary flags
    
    # Set some realistic values for a normal HTTP connection
    sample_values.update({
        'duration': 0,
        'src_bytes': 215,
        'dst_bytes': 45076,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 1,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 1,
        'srv_count': 1,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 255,
        'dst_host_srv_count': 1,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0,
        # One-hot encoded features - set active ones to 1
        'protocol_type_tcp': 1,
        'service_http': 1,
        'flag_SF': 1
    })
    
    # Create DataFrame in the exact order MLflow expects
    data_row = [sample_values[feature] for feature in feature_names]
    
    return pd.DataFrame([data_row], columns=feature_names)

def test_model_service():
    """Test the deployed MLflow model service."""
    print("Testing MLflow model deployment with complete feature set...")
    
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
    
    # Create sample data with complete feature set
    sample_data = create_sample_with_all_features()
    print(f"Sample data shape: {sample_data.shape}")
    print(f"First 10 features: {list(sample_data.columns[:10])}")
    print(f"Last 10 features: {list(sample_data.columns[-10:])}")
    
    # Test with the MLflow dataframe_split format
    payload = {
        "dataframe_split": {
            "columns": sample_data.columns.tolist(),
            "data": sample_data.values.tolist()
        }
    }
    
    print(f"\nTesting with complete dataframe_split format...")
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
            
            # Interpret the result
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
                    if len(prediction) == 2:
                        print(f"P(Normal) = {prediction[0]:.4f}, P(Anomaly) = {prediction[1]:.4f}")
            
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
        print("✅ The MLflow model service is working correctly!")
        print("✅ The deployment pipeline prediction step should now work!")
    else:
        print("\n💥 Model deployment test still failed.")