#!/usr/bin/env python3
"""
Start ZenML dashboard server that binds to all interfaces for remote access.
"""
import os
import sys
from zenml.zen_server.zen_server import ZenServer
from zenml.config.global_config import GlobalConfiguration
from zenml.zen_stores.sql_zen_store import SqlZenStore

def start_zenml_server_remote():
    """Start ZenML server accessible from remote machines."""
    
    # Configure the server to bind to all interfaces
    os.environ["ZENML_SERVER_HOST"] = "0.0.0.0"
    os.environ["ZENML_SERVER_PORT"] = "8237"
    
    print("🚀 Starting ZenML server accessible from remote machines...")
    print("📡 Server will be available at:")
    print("   Local: http://127.0.0.1:8237")
    print("   Remote: http://<your-server-ip>:8237")
    print("\n⚠️  Security Note: This server is accessible from any IP address.")
    print("   Only use this in secure, trusted networks.\n")
    
    try:
        # Initialize ZenML server
        server = ZenServer(
            config={
                "host": "0.0.0.0",
                "port": 8237,
                "debug": False,
                "reload": False,
            }
        )
        
        print("Starting server...")
        server.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("ZenML Remote Access Server")
    print("=" * 50)
    start_zenml_server_remote()