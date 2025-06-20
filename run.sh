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
    echo "If you're experiencing issues, try running setup.sh again."
    echo "For more help, check the troubleshooting section in the README."
else
    echo -e "${GREEN}[SUCCESS]${NC} Application closed successfully."
fi
