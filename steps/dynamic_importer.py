import logging
from typing import Annotated
import pandas as pd
from zenml import step

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step
def dynamic_importer() -> Annotated[pd.DataFrame, "sample_data"]:
    """
    Import sample data for batch inference.
    
    This step creates sample network traffic data that can be used for testing
    the deployed anomaly detection models.
    
    Returns:
        pd.DataFrame: Sample network traffic data for prediction
    """
    try:
        logging.info("Creating sample network traffic data for inference")
        
        # Create sample data that matches KDD99 format
        # This is a simplified example - in production, this would load real network data
        sample_data = pd.DataFrame({
            'duration': [0, 5, 10, 0, 2],
            'src_bytes': [181, 239, 0, 232, 8893],
            'dst_bytes': [5450, 486, 0, 8153, 0],
            'wrong_fragment': [0, 0, 0, 0, 0],
            'urgent': [0, 0, 0, 0, 0],
            'hot': [0, 0, 0, 0, 0],
            'num_failed_logins': [0, 0, 0, 0, 0],
            'num_compromised': [0, 0, 0, 0, 0],
            'root_shell': [0, 0, 0, 0, 0],
            'su_attempted': [0, 0, 0, 0, 0],
            'num_root': [0, 0, 0, 0, 0],
            'num_file_creations': [0, 0, 0, 0, 0],
            'num_shells': [0, 0, 0, 0, 0],
            'num_access_files': [0, 0, 0, 0, 0],
            'num_outbound_cmds': [0, 0, 0, 0, 0],
            'count': [8, 8, 1, 9, 2],
            'srv_count': [8, 8, 1, 9, 2],
            'serror_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            'srv_serror_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            'rerror_rate': [0.0, 0.0, 1.0, 0.0, 0.0],
            'srv_rerror_rate': [0.0, 0.0, 1.0, 0.0, 0.0],
            'same_srv_rate': [1.0, 1.0, 1.0, 1.0, 1.0],
            'diff_srv_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            'srv_diff_host_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dst_host_count': [9, 19, 1, 6, 1],
            'dst_host_srv_count': [9, 19, 1, 6, 1],
            'dst_host_same_srv_rate': [1.0, 1.0, 1.0, 1.0, 1.0],
            'dst_host_diff_srv_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dst_host_same_src_port_rate': [0.11, 0.05, 1.0, 0.17, 1.0],
            'dst_host_srv_diff_host_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dst_host_serror_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dst_host_srv_serror_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dst_host_rerror_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            'dst_host_srv_rerror_rate': [0.0, 0.0, 0.0, 0.0, 0.0],
            # One-hot encoded categorical features (example values)
            'protocol_type_icmp': [0, 0, 1, 0, 0],
            'protocol_type_tcp': [1, 1, 0, 1, 1],
            'protocol_type_udp': [0, 0, 0, 0, 0],
            'service_http': [1, 1, 0, 1, 0],
            'service_smtp': [0, 0, 0, 0, 1],
            'service_ecr_i': [0, 0, 1, 0, 0],
            'flag_SF': [1, 1, 0, 1, 1],
            'flag_REJ': [0, 0, 1, 0, 0],
            'land_0': [1, 1, 1, 1, 1],
            'logged_in_1': [1, 1, 0, 1, 0],
            'logged_in_0': [0, 0, 1, 0, 1],
            'is_host_login_0': [1, 1, 1, 1, 1],
            'is_guest_login_0': [1, 1, 1, 1, 1]
        })
        
        logging.info(f"Created sample data with shape: {sample_data.shape}")
        logging.info("Sample data represents:")
        logging.info("  - Row 0: Normal HTTP connection")
        logging.info("  - Row 1: Normal HTTP connection")  
        logging.info("  - Row 2: Potential ICMP probe")
        logging.info("  - Row 3: Normal HTTP connection")
        logging.info("  - Row 4: Normal SMTP connection")
        
        return sample_data
        
    except Exception as e:
        logging.error(f"Error in dynamic importer step: {str(e)}")
        raise