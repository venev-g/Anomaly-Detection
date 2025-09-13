import logging
import requests
import json
import pandas as pd

from typing import Dict, Any
import click

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_sample_network_data() -> pd.DataFrame:
    """
    Create sample network traffic data for testing predictions.
    
    Returns:
        pd.DataFrame: Sample network traffic data
    """
    # Create sample data representing different types of network traffic
    sample_data = pd.DataFrame({
        # Basic connection features
        'duration': [0, 5, 10, 0, 2, 30, 1],
        'src_bytes': [181, 239, 0, 232, 8893, 1024, 500],
        'dst_bytes': [5450, 486, 0, 8153, 0, 2048, 300],
        'wrong_fragment': [0, 0, 0, 0, 0, 0, 0],
        'urgent': [0, 0, 0, 0, 0, 0, 0],
        'hot': [0, 0, 0, 0, 0, 0, 0],
        'num_failed_logins': [0, 0, 0, 0, 0, 3, 0],  # Row 5 has failed logins (suspicious)
        
        # Content features
        'num_compromised': [0, 0, 0, 0, 0, 0, 0],
        'root_shell': [0, 0, 0, 0, 0, 0, 0],
        'su_attempted': [0, 0, 0, 0, 0, 0, 0],
        'num_root': [0, 0, 0, 0, 0, 0, 0],
        'num_file_creations': [0, 0, 0, 0, 0, 0, 0],
        'num_shells': [0, 0, 0, 0, 0, 0, 0],
        'num_access_files': [0, 0, 0, 0, 0, 0, 0],
        'num_outbound_cmds': [0, 0, 0, 0, 0, 0, 0],
        
        # Traffic features
        'count': [8, 8, 1, 9, 2, 100, 5],  # Row 5 has high count (suspicious)
        'srv_count': [8, 8, 1, 9, 2, 100, 5],
        'serror_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],  # Row 5 has errors
        'srv_serror_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        'rerror_rate': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Row 2 has reject errors
        'srv_rerror_rate': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        'same_srv_rate': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'diff_srv_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'srv_diff_host_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        
        # Host-based features
        'dst_host_count': [9, 19, 1, 6, 1, 255, 10],  # Row 5 has max host count (suspicious)
        'dst_host_srv_count': [9, 19, 1, 6, 1, 255, 10],
        'dst_host_same_srv_rate': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'dst_host_diff_srv_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'dst_host_same_src_port_rate': [0.11, 0.05, 1.0, 0.17, 1.0, 0.01, 0.2],
        'dst_host_srv_diff_host_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'dst_host_serror_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        'dst_host_srv_serror_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        'dst_host_rerror_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'dst_host_srv_rerror_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        
        # One-hot encoded categorical features
        'protocol_type_icmp': [0, 0, 1, 0, 0, 0, 0],
        'protocol_type_tcp': [1, 1, 0, 1, 1, 1, 1],
        'protocol_type_udp': [0, 0, 0, 0, 0, 0, 0],
        'service_http': [1, 1, 0, 1, 0, 1, 1],
        'service_smtp': [0, 0, 0, 0, 1, 0, 0],
        'service_ecr_i': [0, 0, 1, 0, 0, 0, 0],
        'service_ftp': [0, 0, 0, 0, 0, 0, 0],
        'flag_SF': [1, 1, 0, 1, 1, 0, 1],  # Row 5 has different flag
        'flag_REJ': [0, 0, 1, 0, 0, 1, 0],  # Row 2 and 5 have REJ flag
        'flag_S0': [0, 0, 0, 0, 0, 0, 0],
        'land_0': [1, 1, 1, 1, 1, 1, 1],
        'logged_in_1': [1, 1, 0, 1, 0, 0, 1],  # Row 5 not logged in (suspicious)
        'logged_in_0': [0, 0, 1, 0, 1, 1, 0],
        'is_host_login_0': [1, 1, 1, 1, 1, 1, 1],
        'is_guest_login_0': [1, 1, 1, 1, 1, 1, 1]
    })
    
    # Add descriptions for each sample
    sample_descriptions = [
        "Normal HTTP connection",
        "Normal HTTP connection with moderate activity", 
        "ICMP connection with rejection (potential probe)",
        "Normal HTTP connection",
        "Normal SMTP connection", 
        "Suspicious connection with multiple failed logins and high activity",
        "Normal HTTP connection with low activity"
    ]
    
    sample_data['description'] = sample_descriptions
    
    return sample_data


def make_prediction_request(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a prediction request to the deployed model service.
    
    Args:
        url (str): Prediction service URL
        data (Dict[str, Any]): Input data for prediction
        
    Returns:
        Dict[str, Any]: Prediction response
    """
    try:
        headers = {
            'Content-Type': 'application/json',
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making prediction request: {str(e)}")
        raise


@click.command()
@click.option(
    "--service-url",
    default=None,
    help="URL of the prediction service (if not provided, will attempt to find running service)"
)
@click.option(
    "--sample-size",
    default=5,
    help="Number of sample predictions to make"
)
def main(service_url: str, sample_size: int):
    """
    Test the deployed anomaly detection model with sample network traffic data.
    """
    try:
        print("=" * 80)
        print("🔍 ANOMALY DETECTION PREDICTION TEST")
        print("=" * 80)
        
        # Create sample data
        logging.info("Creating sample network traffic data...")
        sample_data = create_sample_network_data()
        
        # Limit to requested sample size
        if sample_size < len(sample_data):
            sample_data = sample_data.head(sample_size)
        
        print(f"\n📊 Testing with {len(sample_data)} network traffic samples:")
        for i, desc in enumerate(sample_data['description']):
            print(f"   {i+1}. {desc}")
        
        # Get prediction service URL
        if not service_url:
            try:
                from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
                
                model_deployer = MLFlowModelDeployer.get_active_model_deployer()
                services = model_deployer.find_model_server(
                    pipeline_name="continuous_deployment_pipeline",
                    pipeline_step_name="mlflow_model_deployer_step",
                    running=True,
                )
                
                if not services:
                    raise RuntimeError("No running prediction service found. Please run the deployment pipeline first.")
                
                service_url = services[0].prediction_url
                print(f"\n🌐 Found running service at: {service_url}")
                
            except Exception as e:
                print(f"\n❌ Error finding prediction service: {str(e)}")
                print("Please provide the service URL manually using --service-url option")
                print("Or run the deployment pipeline first: python run_deployment.py")
                return
        
        # Remove description column for prediction
        prediction_data = sample_data.drop('description', axis=1)
        
        # Make predictions
        print("\n🤖 Making predictions...")
        logging.info(f"Sending data to: {service_url}")
        
        # Convert to format expected by MLflow
        input_data = {
            "instances": prediction_data.to_dict(orient="records")
        }
        
        predictions = make_prediction_request(service_url, input_data)
        
        # Process and display results
        print("\n✅ Predictions completed successfully!")
        print("\n" + "=" * 80)
        print("📋 PREDICTION RESULTS")
        print("=" * 80)
        
        if "predictions" in predictions:
            pred_values = predictions["predictions"]
            
            for i, (_, row) in enumerate(sample_data.iterrows()):
                pred_prob = pred_values[i] if isinstance(pred_values[i], (int, float)) else pred_values[i][0]
                pred_label = "🚨 ANOMALY" if pred_prob > 0.5 else "✅ NORMAL"
                confidence = pred_prob if pred_prob > 0.5 else (1 - pred_prob)
                
                print(f"\n🔸 Sample {i+1}: {row['description']}")
                print(f"   Prediction: {pred_label} (confidence: {confidence:.3f})")
                print(f"   Probability: {pred_prob:.4f}")
        else:
            print("Unexpected prediction format:")
            print(json.dumps(predictions, indent=2))
        
        print("\n" + "=" * 80)
        print("🎉 PREDICTION TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\n💡 INTERPRETATION:")
        print("   • Probability > 0.5: Network traffic classified as ANOMALOUS")
        print("   • Probability ≤ 0.5: Network traffic classified as NORMAL")
        print("   • Higher confidence indicates stronger prediction certainty")
        
        print("\n🔧 USAGE:")
        print("   • Use this script to test your deployed model with custom data")
        print("   • Modify create_sample_network_data() to test different scenarios")
        print("   • Integrate the prediction service into your network monitoring system")
        
    except Exception as e:
        logging.error(f"Error in prediction test: {str(e)}")
        print(f"\n❌ ERROR: Prediction test failed with error: {str(e)}")
        print("Please ensure the deployment pipeline is running and the service is accessible.")
        raise


if __name__ == "__main__":
    main()