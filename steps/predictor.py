import logging
from typing import Annotated
import pandas as pd
import numpy as np
import requests
import json
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

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


def expand_features_to_mlflow_format(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Expand input data to match the complete MLflow model feature set.
    
    Args:
        input_data (pd.DataFrame): Preprocessed input data
        
    Returns:
        pd.DataFrame: Data with all expected features
    """
    expected_features = get_mlflow_expected_features()
    expanded_data = {}
    
    # Initialize all features to 0
    for feature in expected_features:
        expanded_data[feature] = 0
    
    # Map existing features from input data
    for col in input_data.columns:
        if col in expected_features:
            expanded_data[col] = input_data[col].iloc[0] if len(input_data) > 0 else 0
    
    # Create DataFrame with single row
    result_df = pd.DataFrame([expanded_data], columns=expected_features)
    
    logging.info(f"Expanded features from {len(input_data.columns)} to {len(expected_features)}")
    return result_df


@step
def predictor(
    service: MLFlowDeploymentService,
    input_data: pd.DataFrame,
) -> Annotated[pd.DataFrame, "predictions"]:
    """
    Make predictions using the deployed model service.
    
    Args:
        service (MLFlowDeploymentService): The prediction service
        input_data (pd.DataFrame): Input data for prediction
        
    Returns:
        pd.DataFrame: Predictions with input data
    """
    try:
        logging.info("Making predictions using deployed model service")
        logging.info(f"Input data shape: {input_data.shape}")
        logging.info(f"Input columns: {list(input_data.columns)}")
        
        # Expand input data to match MLflow model expectations
        expanded_data = expand_features_to_mlflow_format(input_data)
        logging.info(f"Expanded data shape: {expanded_data.shape}")
        
        # Prepare payload in MLflow dataframe_split format
        payload = {
            "dataframe_split": {
                "columns": expanded_data.columns.tolist(),
                "data": expanded_data.values.tolist()
            }
        }
        
        # Make HTTP request directly to the MLflow endpoint
        try:
            response = requests.post(
                service.prediction_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Prediction successful: {result}")
                
                # Extract predictions from MLflow response
                if 'predictions' in result:
                    predictions = result['predictions']
                else:
                    predictions = result
                
                # Process predictions
                if isinstance(predictions, list) and len(predictions) > 0:
                    pred_value = predictions[0]
                    
                    if isinstance(pred_value, (int, float)):
                        # Binary classification probability
                        pred_df = pd.DataFrame({
                            'prediction_probability': [pred_value],
                            'prediction_binary': [1 if pred_value > 0.5 else 0],
                            'prediction_label': ['Anomaly' if pred_value > 0.5 else 'Normal']
                        })
                    elif isinstance(pred_value, list):
                        # Multiclass probabilities
                        pred_df = pd.DataFrame(
                            [pred_value], 
                            columns=[f'class_{i}_prob' for i in range(len(pred_value))]
                        )
                        pred_df['predicted_class'] = [np.argmax(pred_value)]
                    else:
                        pred_df = pd.DataFrame({'predictions': [pred_value]})
                else:
                    pred_df = pd.DataFrame({'predictions': predictions})
                
            else:
                logging.error(f"Prediction request failed: {response.status_code} - {response.text}")
                raise Exception(f"MLflow prediction failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP request failed: {str(e)}")
            raise
        
        # Combine original input data with predictions
        result_df = pd.concat([
            input_data.reset_index(drop=True), 
            pred_df.reset_index(drop=True)
        ], axis=1)
        
        logging.info("Predictions completed successfully")
        logging.info(f"Result shape: {result_df.shape}")
        
        # Log prediction summary
        if 'prediction_label' in result_df.columns:
            normal_count = sum(result_df['prediction_label'] == 'Normal')
            anomaly_count = sum(result_df['prediction_label'] == 'Anomaly')
            logging.info(f"Prediction summary: {normal_count} Normal, {anomaly_count} Anomaly")
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise