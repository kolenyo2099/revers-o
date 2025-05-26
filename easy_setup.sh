#!/bin/bash
# easy_setup.sh - Simple setup script for Revers-o: GroundedSAM + Perception Encoders Image Similarity Search

echo "======================================================================"
echo "🚀 Revers-o: GroundedSAM + Perception Encoders Image Similarity Search - Easy Setup"
echo "======================================================================"

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print progress information
print_progress() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
print_progress "Checking for Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH."
    echo "Please install Python 3.8 or newer from https://www.python.org/downloads/"
    exit 1
fi

# Check if UV is installed
print_progress "Checking for UV package manager..."
if ! command -v uv &> /dev/null; then
    print_warning "UV package manager not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.zshrc
    if ! command -v uv &> /dev/null; then
        print_error "Failed to install UV. Please install it manually:"
        print_error "curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    else
        print_success "UV installed successfully"
    fi
else
    print_success "UV package manager found"
fi

# Detect hardware platform
print_progress "Detecting your hardware..."
PLATFORM="cpu"  # Default to CPU

# Check for macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check for Apple Silicon
    if [[ $(uname -m) == 'arm64' ]]; then
        print_success "Detected Apple Silicon (M1/M2) Mac"
        PLATFORM="apple"
    else
        print_success "Detected Intel Mac"
        # Check for NVIDIA GPU on Intel Mac (rare but possible)
        if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "NVIDIA"; then
            print_success "Detected NVIDIA GPU"
            PLATFORM="nvidia"
        fi
    fi
else
    # Check for NVIDIA GPU on Linux/Windows
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi -L &> /dev/null; then
            print_success "Detected NVIDIA GPU"
            PLATFORM="nvidia"
        fi
    fi
fi

if [ "$PLATFORM" == "cpu" ]; then
    print_success "Using CPU-only configuration"
fi

# Create directories
print_progress "Setting up project directories..."
mkdir -p my_images
mkdir -p image_retrieval_project/qdrant_data
mkdir -p image_retrieval_project/checkpoints

# Create virtual environment with Python 3.11
print_progress "Creating Python virtual environment with Python 3.11..."
if [ -d ".venv" ]; then
    print_warning "Existing virtual environment found. Removing..."
    rm -rf .venv
fi

uv venv -p 3.11 .venv
if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment with Python 3.11."
    print_error "Make sure Python 3.11 is installed on your system."
    exit 1
fi

# Activate virtual environment
print_progress "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip if needed
print_progress "Upgrading pip..."
uv pip install --upgrade pip

# Install dependencies from requirements.txt
print_progress "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Install yt-dlp for URL video downloads
print_progress "Installing yt-dlp for URL video downloads..."
uv pip install "yt-dlp>=2024.1.1"
if [ $? -eq 0 ]; then
    print_success "yt-dlp installed successfully - URL video downloads enabled"
else
    print_warning "Failed to install yt-dlp - URL video downloads may not work"
fi

# Install ffmpeg (required by yt-dlp for video processing)
print_progress "Installing ffmpeg (required for video processing)..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use Homebrew if available
    if command -v brew &> /dev/null; then
        brew install ffmpeg
        if [ $? -eq 0 ]; then
            print_success "ffmpeg installed successfully via Homebrew"
        else
            print_warning "Failed to install ffmpeg via Homebrew. Please install manually:"
            print_warning "brew install ffmpeg"
        fi
    else
        print_warning "Homebrew not found. Please install ffmpeg manually:"
        print_warning "1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        print_warning "2. Install ffmpeg: brew install ffmpeg"
    fi
else
    # Linux - try apt-get, yum, or pacman
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
        if [ $? -eq 0 ]; then
            print_success "ffmpeg installed successfully via apt-get"
        else
            print_warning "Failed to install ffmpeg via apt-get. Please install manually."
        fi
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
        if [ $? -eq 0 ]; then
            print_success "ffmpeg installed successfully via yum"
        else
            print_warning "Failed to install ffmpeg via yum. Please install manually."
        fi
    elif command -v pacman &> /dev/null; then
        sudo pacman -S ffmpeg
        if [ $? -eq 0 ]; then
            print_success "ffmpeg installed successfully via pacman"
        else
            print_warning "Failed to install ffmpeg via pacman. Please install manually."
        fi
    else
        print_warning "No supported package manager found. Please install ffmpeg manually:"
        print_warning "Ubuntu/Debian: sudo apt-get install ffmpeg"
        print_warning "CentOS/RHEL: sudo yum install ffmpeg"
        print_warning "Arch: sudo pacman -S ffmpeg"
    fi
fi

# Platform-specific installations
if [ "$PLATFORM" == "apple" ]; then
    print_progress "Installing Apple Silicon specific packages with MPS support..."
    uv pip install torch torchvision torchaudio
    
    # Set environment variables for MPS
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    
elif [ "$PLATFORM" == "nvidia" ]; then
    print_progress "Installing NVIDIA GPU specific packages..."
    uv pip install torch torchvision torchaudio
    
else
    print_progress "Installing CPU specific packages..."
    uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
fi

# Clone perception_models repository
print_progress "Setting up Facebook Perception Models repository..."
if [ -d "perception_models" ]; then
    print_warning "Existing perception_models directory found. Removing..."
    rm -rf perception_models
fi

git clone https://github.com/facebookresearch/perception_models.git perception_models
if [ $? -ne 0 ]; then
    print_error "Failed to clone perception_models repository."
    print_error "Make sure git is installed and you have internet connectivity."
    exit 1
else
    print_success "Perception models repository cloned successfully"
fi

# Install dependencies from perception_models repository
print_progress "Installing dependencies from perception_models repository..."
if [ -f "perception_models/requirements.txt" ]; then
    print_success "Dependencies for perception_models will be installed via main requirements.txt"
else
    print_warning "perception_models requirements.txt not found, skipping extra dependencies"
fi

# The import paths in main.py are now assumed to be correct for the ./perception_models structure.
# Removing sed commands that modify them, as they might be fragile.
# sed -i.bak 's|sys.path.append(os.path.join(os.path.dirname(__file__), '\''perception_models'\''))|sys.path.append('\''./perception_models'\'')|g' main.py
# sed -i.bak 's|import perception.models.vision_encoder.pe as pe|import core.vision_encoder.pe as pe|g' main.py
# sed -i.bak 's|import perception.models.vision_encoder.transforms as transforms|import core.vision_encoder.transforms as transforms|g' main.py
# rm -f main.py.bak

# Create a simple run script
print_progress "Creating run script..."
cat > run.sh << 'EOF'
#!/bin/bash
# Simple run script for Revers-o: GroundedSAM + Perception Encoders Image Similarity Search

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[STARTUP]${NC} Activating environment..."
source .venv/bin/activate

echo -e "${BLUE}[STARTUP]${NC} Starting Revers-o: GroundedSAM + Perception Encoders Image Similarity Search..."

# Set environment variables
if [[ "$(uname -m)" == "arm64" && "$OSTYPE" == "darwin"* ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# Run the application
python main.py

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Application exited with an error."
    echo "If you're experiencing issues, try running easy_setup.sh again."
    echo "For more help, check the troubleshooting section in the README."
else
    echo -e "${GREEN}[SUCCESS]${NC} Application closed successfully."
fi
EOF

chmod +x run.sh

print_success "Setup completed successfully! 🎉"
echo ""
echo "Getting started:"
echo "----------------"
echo "1. Add your images to the 'my_images' folder"
echo "2. Run './run.sh' to start the application"
echo ""
echo "✨ Video Processing Features:"
echo "- Extract keyframes from local videos using the 'Extract Images from Video' section"
echo "- Download videos from URLs (YouTube, Twitter, Facebook, Instagram, etc.)"
echo "- Supports .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v formats"
echo "- Uses intelligent scene detection and keyframe extraction"
echo ""
echo "The application will open in your browser automatically."
echo "Enjoy using Revers-o!" 