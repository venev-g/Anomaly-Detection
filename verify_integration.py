#!/usr/bin/env python3
"""
ZenML/MLflow Integration Verification Script

This script verifies that the ZenML and MLflow integration is properly set up
and working correctly for the anomaly detection project.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, capture_output=True, check=False):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        return e


def check_imports():
    """Check if required packages can be imported."""
    print("🔍 Checking Python imports...")
    
    imports_to_check = [
        ("zenml", "ZenML"),
        ("mlflow", "MLflow"),
        ("xgboost", "XGBoost"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn")
    ]
    
    all_good = True
    for module, name in imports_to_check:
        try:
            __import__(module)
            print(f"  ✅ {name} imported successfully")
        except ImportError as e:
            print(f"  ❌ {name} import failed: {e}")
            all_good = False
    
    return all_good


def check_zenml_status():
    """Check ZenML initialization and status."""
    print("\n🔍 Checking ZenML status...")
    
    # Check if ZenML is initialized
    result = run_command("zenml status")
    if result.returncode == 0:
        print("  ✅ ZenML is initialized")
        print(f"  📋 Status: {result.stdout.strip()}")
    else:
        print("  ❌ ZenML not initialized")
        print("  💡 Run: zenml init")
        return False
    
    return True


def check_mlflow_integration():
    """Check if MLflow integration is installed."""
    print("\n🔍 Checking MLflow integration...")
    
    result = run_command("zenml integration list")
    if result.returncode == 0 and "mlflow" in result.stdout:
        if "✅" in result.stdout.split("mlflow")[1].split("\n")[0]:
            print("  ✅ MLflow integration is installed")
            return True
        else:
            print("  ⚠️ MLflow integration is available but not installed")
            print("  💡 Run: zenml integration install mlflow -y")
            return False
    else:
        print("  ❌ MLflow integration not found")
        return False


def check_stack_components():
    """Check if experiment tracker and model deployer are registered."""
    print("\n🔍 Checking stack components...")
    
    # Check experiment trackers
    result = run_command("zenml experiment-tracker list")
    if result.returncode == 0:
        if "mlflow_tracker" in result.stdout or "mlflow" in result.stdout:
            print("  ✅ MLflow experiment tracker found")
        else:
            print("  ❌ MLflow experiment tracker not registered")
            print("  💡 Run: zenml experiment-tracker register mlflow_tracker --flavor=mlflow")
            return False
    
    # Check model deployers
    result = run_command("zenml model-deployer list")
    if result.returncode == 0:
        if "mlflow_deployer" in result.stdout or "mlflow" in result.stdout:
            print("  ✅ MLflow model deployer found")
        else:
            print("  ❌ MLflow model deployer not registered")
            print("  💡 Run: zenml model-deployer register mlflow_deployer --flavor=mlflow")
            return False
    
    return True


def check_active_stack():
    """Check the active stack configuration."""
    print("\n🔍 Checking active stack...")
    
    result = run_command("zenml stack describe")
    if result.returncode == 0:
        stack_info = result.stdout
        print("  📋 Active stack details:")
        
        # Check for MLflow components
        if "mlflow" in stack_info.lower():
            print("  ✅ Stack includes MLflow components")
            
            # Parse stack info to show details
            lines = stack_info.split('\n')
            for line in lines:
                if 'experiment_tracker' in line.lower() or 'model_deployer' in line.lower():
                    print(f"    {line.strip()}")
        else:
            print("  ⚠️ Stack may not include MLflow components")
            print("  💡 Consider running:")
            print("    zenml stack register anomaly_detection_stack \\")
            print("      -e mlflow_tracker -d mlflow_deployer -a default -o default --set")
        
        return True
    else:
        print("  ❌ Could not describe stack")
        return False


def check_mlflow_connection():
    """Test basic MLflow functionality."""
    print("\n🔍 Testing MLflow connection...")
    
    test_script = '''
import mlflow
import tempfile
import os

try:
    # Set tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Test basic MLflow operations
    with mlflow.start_run():
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_metric", 0.95)
    
    print("✅ MLflow basic operations work")
except Exception as e:
    print(f"❌ MLflow test failed: {e}")
    exit(1)
'''
    
    result = run_command(f"python -c '{test_script}'")
    if result.returncode == 0:
        print("  ✅ MLflow connection successful")
        return True
    else:
        print("  ❌ MLflow connection failed")
        print(f"  📝 Error: {result.stderr}")
        return False


def check_data_availability():
    """Check if required data files exist."""
    print("\n🔍 Checking data availability...")
    
    data_file = Path("data/kddcup.data.corrected")
    if data_file.exists():
        print(f"  ✅ Data file found: {data_file}")
        print(f"  📊 File size: {data_file.stat().st_size / (1024*1024):.1f} MB")
        return True
    else:
        print("  ❌ Data file not found")
        print("  💡 Copy data file:")
        print("    cp /workspaces/Anomaly-Detection/anomaly/data/kddcup.data.corrected data/")
        return False


def check_config_file():
    """Check if configuration file exists and is valid."""
    print("\n🔍 Checking configuration...")
    
    config_file = Path("config.yaml")
    if config_file.exists():
        print(f"  ✅ Configuration file found: {config_file}")
        
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check for key sections
            required_sections = ['data_ingestion', 'preprocessing', 'xgboost']
            missing_sections = []
            
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"  ⚠️ Missing config sections: {missing_sections}")
                return False
            else:
                print("  ✅ Configuration file is valid")
                return True
                
        except Exception as e:
            print(f"  ❌ Configuration file is invalid: {e}")
            return False
    else:
        print("  ❌ Configuration file not found")
        return False


def run_verification():
    """Run all verification checks."""
    print("🚀 Starting ZenML/MLflow Integration Verification")
    print("=" * 60)
    
    checks = [
        ("Python Imports", check_imports),
        ("ZenML Status", check_zenml_status),
        ("MLflow Integration", check_mlflow_integration),
        ("Stack Components", check_stack_components),
        ("Active Stack", check_active_stack),
        ("MLflow Connection", check_mlflow_connection),
        ("Data Availability", check_data_availability),
        ("Configuration", check_config_file),
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
            if not results[check_name]:
                all_passed = False
        except Exception as e:
            print(f"  ❌ Check failed with exception: {e}")
            results[check_name] = False
            all_passed = False
    
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:<10} {check_name}")
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED! System is ready for anomaly detection.")
        print("\n🚀 Next steps:")
        print("  1. Run training: python run_pipeline.py --config config.yaml")
        print("  2. Deploy models: python run_deployment.py --config config.yaml")
        print("  3. Test predictions: python sample_predict.py --config config.yaml")
        return True
    else:
        print("⚠️ SOME CHECKS FAILED. Please address the issues above.")
        print("\n💡 Quick fix commands:")
        print("  zenml integration install mlflow -y")
        print("  zenml experiment-tracker register mlflow_tracker --flavor=mlflow")
        print("  zenml model-deployer register mlflow_deployer --flavor=mlflow")
        print("  zenml stack register anomaly_detection_stack -e mlflow_tracker -d mlflow_deployer -a default -o default --set")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)