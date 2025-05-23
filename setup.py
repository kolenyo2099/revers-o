#!/usr/bin/env python3
"""
Setup script for Revers-o: GroundedSAM + Perception Encoders Image Similarity Search
This handles the special setup steps that can't be done via requirements.txt
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors"""
    print(f"Running: {description or cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def setup_perception_models():
    """Clone and set up the perception_models repository"""
    
    # Check if already exists
    if os.path.exists('./perception_models'):
        print("✅ perception_models directory already exists")
        return True
    
    print("🔄 Cloning perception_models repository...")
    
    # Clone the repository
    if not run_command(
        "git clone https://github.com/facebookresearch/perception_models.git",
        "Cloning perception_models repository"
    ):
        return False
    
    # Modify requirements.txt to use eva-decord instead of decord (for Apple Silicon compatibility)
    requirements_file = './perception_models/requirements.txt'
    
    if os.path.exists(requirements_file):
        # Back up the original file
        shutil.copy(requirements_file, f"{requirements_file}.bak")
        
        # Read and modify the content
        with open(requirements_file, 'r') as file:
            content = file.read()
        
        # Replace decord with eva-decord for cross-platform compatibility
        new_content = content.replace('decord==0.6.0', 'eva-decord==0.6.1')
        
        # Handle other potential compatibility issues
        new_content = new_content.replace('decord>=0.6.0', 'eva-decord==0.6.1')
        new_content = new_content.replace('decord', 'eva-decord==0.6.1')
        
        with open(requirements_file, 'w') as file:
            file.write(new_content)
        
        print("✅ Modified requirements.txt to use eva-decord for cross-platform compatibility")
    
    # Install the perception_models package with better error handling
    original_dir = os.getcwd()
    try:
        os.chdir('./perception_models')
        
        # Try to install with pip first
        success = run_command(
            "pip install -e .",
            "Installing perception_models package"
        )
        
        # If that fails, try with specific Python executable
        if not success:
            import sys
            python_exec = sys.executable
            success = run_command(
                f"{python_exec} -m pip install -e .",
                "Installing perception_models package with explicit Python executable"
            )
            
    finally:
        os.chdir(original_dir)
    
    if success:
        print("✅ perception_models setup completed successfully")
    else:
        print("❌ Failed to install perception_models package")
        print("💡 Try running: pip install -e ./perception_models manually")
    
    return success

def create_directories():
    """Create necessary project directories"""
    directories = [
        "./image_retrieval_project",
        "./image_retrieval_project/qdrant_data",
        "./image_retrieval_project/images", 
        "./image_retrieval_project/checkpoints",
        "./my_images"  # Default folder for user images
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up Revers-o: GroundedSAM + Perception Encoders Image Similarity Search")
    print("=" * 50)
    
    # Step 1: Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Step 2: Set up perception_models
    print("\n🧠 Setting up perception_models...")
    if not setup_perception_models():
        print("❌ Setup failed at perception_models step")
        sys.exit(1)
    
    # Step 3: Clean up any existing lock files
    print("\n🧹 Cleaning up any existing lock files...")
    qdrant_data_path = "./image_retrieval_project/qdrant_data"
    for root, dirs, files in os.walk(qdrant_data_path):
        for file in files:
            if file == ".lock":
                lock_path = os.path.join(root, file)
                try:
                    os.remove(lock_path)
                    print(f"✅ Removed lock file: {lock_path}")
                except:
                    print(f"⚠️ Could not remove lock file: {lock_path}")
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your images to the './my_images' folder")
    print("2. Run: python main.py --build-database")
    print("3. Run: python main.py --interface")
    print("\nOr run both steps: python main.py --all")

if __name__ == "__main__":
    main()