@echo off
echo ======================================================================
echo 🚀 Revers-o: GroundedSAM + Perception Encoders Image Similarity Search - Easy Setup
echo ======================================================================

:: Check if Python is installed
echo [SETUP] Checking for Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or newer from https://www.python.org/downloads/
    exit /b 1
)

:: Check if git is installed
echo [SETUP] Checking for git installation...
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is required for cloning perception models repository.
    echo Please install git from https://git-scm.com/
    exit /b 1
)

:: Check if UV is installed
echo [SETUP] Checking for UV package manager...
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] UV package manager not found. Installing UV...
    powershell -Command "Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -OutFile install_uv.ps1"
    powershell -ExecutionPolicy Bypass -File install_uv.ps1
    del install_uv.ps1
    
    :: Refresh environment variables
    call refreshenv
    
    uv --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install UV. Please install it manually:
        echo curl -LsSf https://astral.sh/uv/install.ps1 | powershell
        exit /b 1
    ) else (
        echo [SUCCESS] UV installed successfully
    )
) else (
    echo [SUCCESS] UV package manager found
)

:: Detect hardware platform
echo [SETUP] Detecting your hardware...
set PLATFORM=cpu

:: Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Detected NVIDIA GPU
    set PLATFORM=nvidia
) else (
    echo [SUCCESS] Using CPU-only configuration
)

:: Create directories
echo [SETUP] Setting up project directories...
if not exist my_images mkdir my_images
if not exist data mkdir data
if not exist models mkdir models
if not exist checkpoints mkdir checkpoints

:: Create virtual environment with Python 3.11
echo [SETUP] Creating Python virtual environment with Python 3.11...
if exist .venv (
    echo [WARNING] Existing virtual environment found. Removing...
    rmdir /s /q .venv
)

uv venv -p 3.11 .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment with Python 3.11.
    echo Make sure Python 3.11 is installed on your system.
    exit /b 1
)

:: Activate virtual environment
echo [SETUP] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Upgrade pip if needed
echo [SETUP] Upgrading pip...
uv pip install --upgrade pip

:: Install dependencies from requirements.txt
echo [SETUP] Installing dependencies from requirements.txt...
if exist requirements.txt (
    uv pip install -r requirements.txt
) else (
    echo [WARNING] No requirements.txt found, installing basic dependencies...
    uv pip install torch torchvision torchaudio transformers gradio pillow numpy opencv-python
)

:: Install yt-dlp for URL video downloads
echo [SETUP] Installing yt-dlp for URL video downloads...
uv pip install "yt-dlp>=2024.1.1"
if %errorlevel% equ 0 (
    echo [SUCCESS] yt-dlp installed successfully - URL video downloads enabled
) else (
    echo [WARNING] Failed to install yt-dlp - URL video downloads may not work
)

:: Install ffmpeg
echo [SETUP] Installing ffmpeg (required for video processing)...
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ffmpeg not found. Please install it manually:
    echo 1. Download from https://ffmpeg.org/download.html
    echo 2. Add to PATH
)

:: Clone perception models repository
echo [SETUP] Setting up perception models...
if not exist perception_models (
    git clone https://github.com/facebookresearch/perception_models.git
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to clone perception models repository.
        exit /b 1
    )
)

echo [SUCCESS] Setup completed successfully!
echo.
echo To start the application, run: run.bat 