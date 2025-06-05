#!/bin/bash
# start.sh - Quick start script for Revers-o

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Revers-o Quick Start${NC}"
echo "=================================="

# Check if this is the first run (no virtual environment)
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚡ First time setup detected...${NC}"
    echo "Setting up your environment (this may take a few minutes)..."
    echo ""
    
    # Make setup script executable
    chmod +x setup.sh
    
    # Run setup
    ./setup.sh
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Setup failed!${NC}"
        echo "Please check the error messages above and try again."
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✅ Setup completed!${NC}"
    echo ""
fi

# Check if run script exists and is executable
if [ ! -f "run.sh" ]; then
    echo -e "${RED}❌ run.sh not found!${NC}"
    echo "Please run ./setup.sh first."
    exit 1
fi

if [ ! -x "run.sh" ]; then
    chmod +x run.sh
fi

# Check if my_images directory has content
if [ -d "my_images" ] && [ -z "$(ls -A my_images)" ]; then
    echo -e "${YELLOW}📁 Note: 'my_images' folder is empty${NC}"
    echo "   Add some images to get started with similarity search!"
    echo ""
fi

# Start the application
echo -e "${BLUE}🎯 Starting Revers-o application...${NC}"
echo ""
./run.sh "$@"
