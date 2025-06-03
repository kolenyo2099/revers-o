#!/usr/bin/env python3
"""
Perception Models Import Initialization
This script handles the proper setup of import paths for the perception_models package.
Run this before importing from perception_models or include its functionality in your startup.
"""

import os
import sys
import importlib.util
from pathlib import Path
import types

def setup_perception_models_imports():
    """
    Set up the import paths for perception_models to work correctly
    without modifying the original code.
    """
    # Add perception_models to path if not already there
    perception_models_path = os.path.abspath('./perception_models')
    if perception_models_path not in sys.path:
        sys.path.append(perception_models_path)
    
    # Create a hook to handle perception_models imports correctly
    # This approach avoids modifying the original source files
    class PatchedLoader:
        def find_module(self, fullname, path=None):
            if fullname.startswith('core.vision_encoder'):
                return self
            return None
        
        def load_module(self, fullname):
            # If already loaded, return it
            if fullname in sys.modules:
                return sys.modules[fullname]
            
            # Map to the correct perception_models path
            mapped_name = fullname.replace('core.', 'perception_models.core.')
            
            # Try to import the mapped module
            try:
                module = importlib.import_module(mapped_name)
                # Create an alias in sys.modules so future imports work
                sys.modules[fullname] = module
                return module
            except ImportError as e:
                raise ImportError(f"Error importing {mapped_name}: {e}")
    
    # Install the import hook
    sys.meta_path.insert(0, PatchedLoader())
    
    print(f"✅ Perception Models import paths configured successfully")
    return True

if __name__ == "__main__":
    setup_perception_models_imports() 