@echo off
echo ======================================================================
echo 🚀 Revers-o: GroundedSAM + Perception Encoders Image Similarity Search - Easy Setup (Windows)
echo ======================================================================
echo.

:: Check for Python
echo [SETUP] Checking for Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8 or newer from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [SUCCESS] Found Python %PYTHON_VERSION%

:: Detect hardware
echo [SETUP] Detecting your hardware...
set PLATFORM=cpu

:: Check for NVIDIA GPU
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    nvidia-smi >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [SUCCESS] Detected NVIDIA GPU
        set PLATFORM=nvidia
    )
) else (
    echo [SUCCESS] Using CPU-only configuration
)

:: Create directories
echo [SETUP] Setting up project directories...
if not exist my_images mkdir my_images
if not exist image_retrieval_project\qdrant_data mkdir image_retrieval_project\qdrant_data
if not exist image_retrieval_project\checkpoints mkdir image_retrieval_project\checkpoints

:: Create virtual environment
echo [SETUP] Creating Python virtual environment...
python -m venv .venv

:: Activate virtual environment
echo [SETUP] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Upgrade pip
echo [SETUP] Upgrading pip...
python -m pip install --upgrade pip

:: Install base requirements
echo [SETUP] Installing dependencies from requirements.txt...
pip install -r requirements.txt

:: Platform-specific installations
if "%PLATFORM%"=="nvidia" (
    echo [SETUP] Installing NVIDIA GPU specific packages...
    pip install torch torchvision torchaudio
) else (
    echo [SETUP] Installing CPU specific packages...
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
)

# Install Grounded SAM and other dependencies
# These are now in requirements.txt
# echo [SETUP] Installing Grounded SAM and other ML dependencies...
# pip install autodistill autodistill-grounded-sam

:: Clone perception_models if not already present
if not exist perception_models (
    echo [SETUP] Cloning Facebook Perception Models repository...
    git clone https://github.com/facebookresearch/perception_models.git perception_models
)

:: Create run script
echo [SETUP] Creating run script...
(
echo @echo off
echo echo [STARTUP] Activating environment...
echo call .venv\Scripts\activate.bat
echo.
echo echo [STARTUP] Starting Revers-o: GroundedSAM + Perception Encoders Image Similarity Search...
echo.
echo python main.py
echo.
echo if %%ERRORLEVEL%% NEQ 0 (
echo     echo [ERROR] Application exited with an error.
echo     echo If you're experiencing issues, try running easy_setup.bat again.
echo     echo For more help, check the troubleshooting section in the README.
echo ) else (
echo     echo [SUCCESS] Application closed successfully.
echo )
echo.
echo pause
) > run.bat

echo [SUCCESS] Setup completed successfully! 🎉
echo.
echo Getting started:
echo ----------------
echo 1. Add your images to the 'my_images' folder
echo 2. Run 'run.bat' to start the application
echo.
echo The application will open in your browser automatically.
echo Enjoy using Revers-o: GroundedSAM + Perception Encoders Image Similarity Search!

pause 