#!/usr/bin/env python3
"""
Revers-o: GroundedSAM + Perception Encoders Image Similarity Search Application
Main application file - now primarily handles initialization and UI launch.
This is the refactored version. To use it, rename this file to main.py.
"""

import argparse
import os
import sys
import socket # For port checking
import gradio as gr
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import torch

# Initialize perception_models import paths
from init_perception_models import setup_perception_models_imports
setup_perception_models_imports()

# Core application modules
from core.device import setup_device
from core.models import load_pe_model # This function will set globals in core.models
from core.database import cleanup_qdrant_connections
from gradio_ui.tabs import create_multi_mode_interface
# AppState is initialized within create_multi_mode_interface, which imports it from core.app_state

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Revers-o: GroundedSAM + Perception Encoders Image Similarity Search Application")
    parser.add_argument("--port", type=int, default=7860, help="Port to use for the Gradio interface")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing link (use with caution and if allowed by environment)")
    args = parser.parse_args()

    # Clean up any existing database connections with force option
    print(f"[STATUS] Running database cleanup and integrity checks...")
    try:
        import threading
        # Run cleanup in a thread with a timeout to prevent blocking indefinitely
        cleanup_thread = threading.Thread(target=cleanup_qdrant_connections, args=(True,))
        cleanup_thread.start()
        cleanup_thread.join(timeout=10) # Wait for 10 seconds max
        
        if cleanup_thread.is_alive():
            print(f"[WARNING] Database cleanup is taking too long - continuing with application startup")
        else:
            print(f"[STATUS] Database cleanup completed")
    except Exception as e:
        print(f"[ERROR] Exception during initial cleanup: {e}")


    # Setup device and load models
    print("🚀 Initializing Grounded SAM Region Search Application")
    print("=" * 50)

    # Setup device
    # This device_instance is passed to load_pe_model
    device_instance = setup_device()

    # Load PE models
    print("\n🧠 Loading Perception Encoder models...")
    try:
        # load_pe_model will set the global variables pe_model, pe_vit_model, preprocess, and device in core.models
        load_pe_model(device_instance)
        # After load_pe_model is called, we can import the model variables from core.models
        from core.models import pe_model, pe_vit_model, preprocess, device as model_device
        if pe_model is None or preprocess is None or model_device is None: # Basic check
             raise Exception("Essential models or device not loaded correctly by load_pe_model into core.models")
        print("✅ Models loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import from core.models after loading. Error: {e}")
        print(f"   This might indicate an issue with the refactoring or sys.path.")
        print(f"   Current sys.path: {sys.path}")
        return
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("   Make sure you've run setup.py first and perception_models are available!")
        import traceback # Import for more detailed error
        traceback.print_exc()
        return

    print("\n🌐 Launching multi-mode interface...")
    # Pass the loaded models and device to the UI creation function
    # AppState is managed within create_multi_mode_interface
    demo = create_multi_mode_interface(pe_model, pe_vit_model, preprocess, model_device)
    print("🚀 Interface ready! Opening in browser...")

    # Try to launch with different ports to avoid port conflicts
    def is_port_in_use(port_to_check):
        # Check if port is available on 127.0.0.1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_local:
            local_free = s_local.connect_ex(('127.0.0.1', port_to_check)) != 0
        # If sharing, also check if port is available on 0.0.0.0
        if args.share: # This check should be outside the with block for s_local
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_all:
                all_free = s_all.connect_ex(('0.0.0.0', port_to_check)) != 0
            return local_free and all_free # Port must be free on both for sharing
        return local_free # If not sharing, only local matters

    launched = False
    max_port_attempts = 10

    for i in range(max_port_attempts):
        current_port = args.port + i

        # Check port availability before attempting launch
        if is_port_in_use(current_port):
            if i == 0 and current_port == args.port: # Only print for the initial specific port
                 print(f"Port {args.port} specified by --port is in use.")
            else:
                 print(f"Port {current_port} is already in use.")
            continue # Try next port

        try:
            print(f"Trying to start server on port {current_port}...")
            server_name = "0.0.0.0" if args.share else "127.0.0.1"
            demo.launch(share=args.share, server_name=server_name, server_port=current_port)
            launched = True
            break
        except OSError as e:
            if "Address already in use" in str(e) or "EADDRINUSE" in str(e):
                 print(f"Port {current_port} is already in use (caught by launch). Trying next port...")
            else:
                print(f"Failed to start on port {current_port} due to OSError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while launching on port {current_port}: {e}")

    if not launched:
        print(f"All attempted ports from {args.port} to {args.port + max_port_attempts -1} are in use or failed.")
        print("Attempting to launch on a Gradio-selected port...")
        try:
            server_name = "0.0.0.0" if args.share else "127.0.0.1"
            demo.launch(share=args.share, server_name=server_name)
        except Exception as e:
            print(f"❌ Failed to launch Gradio interface: {e}")
            print("   Please check if another instance is running or if there are port conflicts.")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
