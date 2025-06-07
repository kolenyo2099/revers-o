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

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Simple Revers-o from main.py...")

    # The SimpleReverso instance is created within ui.py
    # The create_simple_interface function (from ui.py) builds the Gradio app
    
    # Create and launch interface
    # No need to check if create_simple_interface exists if the import succeeded.
    # If it didn't, an ImportError would have already occurred.
    demo = create_simple_interface()
    
    print("🌐 Launching Gradio interface...")
    demo.launch(
        share=False, # Set to True to share publicly (requires internet access for Gradio)
        server_name="127.0.0.1", # Listen on localhost
        server_port=7860,        # Standard Gradio port
        show_error=True,         # Show errors in the browser console
        # debug=True             # Uncomment for Gradio's debug mode
    )