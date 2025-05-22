#!/bin/bash
# troubleshoot.sh - Simple troubleshooting script for Grounded SAM Region Search

echo "======================================================================"
echo "🔍 Grounded SAM Region Search - Troubleshooting"
echo "======================================================================"

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

# Function to print check results
print_check() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[PROBLEM]${NC} $1"
}

# Function to print fix suggestions
print_fix() {
    echo -e "  ${GREEN}→${NC} $1"
}

# Check if running with virtual environment
print_section "Environment Check"

if [[ -d ".venv" ]]; then
    print_success "Virtual environment exists"
    
    # Try to activate the environment
    if source .venv/bin/activate 2>/dev/null; then
        print_success "Virtual environment can be activated"
    else
        print_error "Cannot activate virtual environment"
        print_fix "Try running: rm -rf .venv && ./easy_setup.sh"
    fi
else
    print_error "Virtual environment not found"
    print_fix "Run ./easy_setup.sh to create the environment"
fi

# Check Python version in virtual environment
print_check "Checking Python version..."
if command -v python &> /dev/null; then
    python_version=$(python --version 2>&1)
    print_success "Python version: $python_version"
else
    print_error "Python not found in environment"
    print_fix "Run ./easy_setup.sh to set up the environment correctly"
fi

# Check critical dependencies
print_section "Dependency Check"

dependencies=("torch" "numpy" "gradio" "opencv-python" "autodistill-grounded-sam" "qdrant-client")

for dep in "${dependencies[@]}"; do
    if pip show $dep &> /dev/null; then
        version=$(pip show $dep | grep Version | cut -d' ' -f2)
        print_success "$dep is installed (version: $version)"
    else
        print_error "$dep is not installed"
        print_fix "Run ./easy_setup.sh again to install all dependencies"
    fi
done

# Check if perception_models exists
print_section "External Dependencies Check"

if [[ -d "perception_models" ]]; then
    print_success "Perception models folder exists"
else
    print_error "Perception models folder not found"
    print_fix "Run ./easy_setup.sh again or manually clone:"
    print_fix "git clone https://github.com/facebookresearch/perception.git perception_models"
fi

# Check folder structure
print_section "Folder Structure Check"

folders=("my_images" "image_retrieval_project/qdrant_data" "image_retrieval_project/checkpoints")

for folder in "${folders[@]}"; do
    if [[ -d "$folder" ]]; then
        print_success "Folder exists: $folder"
    else
        print_error "Missing folder: $folder"
        print_fix "Run: mkdir -p $folder"
    fi
done

# Check if main.py exists
print_section "Application Files Check"

if [[ -f "main.py" ]]; then
    print_success "Main application file exists: main.py"
else
    print_error "Main application file not found: main.py"
    print_fix "Make sure you're in the correct directory"
    print_fix "The application may not be installed correctly"
fi

# Check hardware and acceleration
print_section "Hardware Acceleration Check"

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check for Apple Silicon
    if [[ $(uname -m) == 'arm64' ]]; then
        print_success "Running on Apple Silicon"
        
        # Try to import torch and check MPS availability
        mps_available=$(python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null)
        
        if [[ "$mps_available" == "True" ]]; then
            print_success "MPS acceleration is available"
        else
            print_warning "MPS acceleration is not available"
            print_fix "Try reinstalling PyTorch with: pip install torch torchvision torchaudio"
        fi
    else
        print_success "Running on Intel Mac"
    fi
# Check for NVIDIA GPU
elif command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
    
    if [[ -n "$gpu_info" ]]; then
        print_success "NVIDIA GPU detected: $gpu_info"
        
        # Check CUDA availability
        cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        
        if [[ "$cuda_available" == "True" ]]; then
            print_success "CUDA acceleration is available"
        else
            print_warning "CUDA acceleration is not available"
            print_fix "Try reinstalling PyTorch with: pip install torch torchvision torchaudio"
        fi
    else
        print_warning "NVIDIA tools found but no GPU detected"
    fi
else
    print_success "Running in CPU-only mode"
fi

# Database Check
print_section "Database Check"

if [[ -d "image_retrieval_project/qdrant_data/collections" ]]; then
    collections=$(ls image_retrieval_project/qdrant_data/collections 2>/dev/null)
    
    if [[ -n "$collections" ]]; then
        print_success "Database collections found: $collections"
    else
        print_warning "No database collections found"
        print_fix "This is normal if you haven't processed any images yet"
    fi
    
    # Check for lock file
    if [[ -f "image_retrieval_project/qdrant_data/.lock" ]]; then
        print_warning "Database lock file found"
        print_fix "If the application crashed, you may need to remove the lock file:"
        print_fix "rm image_retrieval_project/qdrant_data/.lock"
    fi
else
    print_warning "Database directory not initialized"
    print_fix "This is normal if you haven't processed any images yet"
fi

# Summary and next steps
print_section "Summary"

echo "If you're experiencing issues:"
echo "1. Try running ./easy_setup.sh again to reinstall dependencies"
echo "2. Make sure you have images in the 'my_images' folder"
echo "3. Check that you have enough disk space and memory"
echo "4. For more detailed help, visit the project GitHub page"
echo ""
echo "You can start the application with: ./run.sh" 