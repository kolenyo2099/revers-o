#!/usr/bin/env python3
"""
Simple Revers-o: Streamlined Visual Investigation Tool
GroundedSAM + PE-Core-L14-336 + Simple Vector Search

Simplified based on PE research:
- Single model: PE-Core-L14-336 (optimal balance)
- Single layer: 24 (PE research optimal for L14)
- Core pipeline: Image → GroundedSAM → PE embeddings → Vector search
- Simple UI: Build database, search similar regions
"""

# main.py - Entry point for the Simple Revers-o application.

# Import the UI creation function
from ui import create_simple_interface
import os

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Simple Revers-o from main.py...")

    # The SimpleReverso instance is created within ui.py
    # The create_simple_interface function (from ui.py) builds the Gradio app
    
    # Create the interface
    demo = create_simple_interface()

    # Launch the interface
    demo.launch(
        server_name="127.0.0.1",  # Use localhost
        server_port=None,         # Let Gradio find an available port
        share=False,              # Don't create public URL by default
        show_error=True,          # Show detailed error messages
        show_api=False,           # Hide API documentation
        favicon_path="favicon.ico" if os.path.exists("favicon.ico") else None
    )