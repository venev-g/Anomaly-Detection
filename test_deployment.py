#!/usr/bin/env python3
"""
Test script to verify the MLflow model deployment is working correctly.
Updated to handle the complete feature set (122 features) that MLflow expects.
"""

import logging
import requests
import json
import pandas as pd
import numpy as np
from zenml.client import Client

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_mlflow_expected_features():
    """Get the complete set of features that MLflow model expects (alphabetical order)."""
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
    logging.info("Creating sample data with complete feature set...")
    
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

def test_model_deployment():
    """Test the deployed model with sample data using the correct feature format."""
    try:
        logging.info("Testing MLflow model deployment with complete feature set...")
        
        # Get ZenML client
        client = Client()
        
        # Find the deployed service
        services = client.active_stack.model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name="model"
        )
        
        if not services:
            logging.error("No deployed services found!")
            return False
            
        service = services[0]
        logging.info(f"Found deployed service: {service}")
        logging.info(f"Prediction URL: {service.prediction_url}")
        
        # Create sample data with complete feature set
        sample_data = create_sample_with_all_features()
        logging.info(f"Sample data shape: {sample_data.shape}")
        logging.info(f"First 10 features: {list(sample_data.columns[:10])}")
        logging.info(f"Last 10 features: {list(sample_data.columns[-10:])}")
        
        # Test with the MLflow dataframe_split format (the format that works)
        payload = {
            "dataframe_split": {
                "columns": sample_data.columns.tolist(),
                "data": sample_data.values.tolist()
            }
        }
        
        logging.info("Testing with complete dataframe_split format...")
        try:
            response = requests.post(
                service.prediction_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"✅ Prediction SUCCESS!")
                logging.info(f"Response: {result}")
                
                # Interpret the result
                if isinstance(result, dict) and 'predictions' in result:
                    predictions = result['predictions']
                    if isinstance(predictions, list) and len(predictions) > 0:
                        prediction = predictions[0]
                        if isinstance(prediction, (int, float)):
                            logging.info(f"Predicted value: {prediction}")
                            if prediction < 0.5:
                                logging.info("🟢 Prediction: NORMAL connection")
                            else:
                                logging.info("🔴 Prediction: ANOMALY detected")
                        elif isinstance(prediction, list):
                            logging.info(f"Prediction probabilities: {prediction}")
                
                return True
            else:
                logging.error(f"❌ Request FAILED: {response.status_code}")
                logging.error(f"Error response: {response.text[:1000]}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Request ERROR: {str(e)}")
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