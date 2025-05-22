@echo off
echo ======================================================================
echo 🔍 Grounded SAM Region Search - Troubleshooting (Windows)
echo ======================================================================
echo.

:: Function to print section headers (simulated)
echo ==== Environment Check ====
echo.

:: Check if virtual environment exists
if exist .venv (
    echo [OK] Virtual environment exists
    
    :: Try to activate the environment
    call .venv\Scripts\activate.bat >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Virtual environment can be activated
    ) else (
        echo [PROBLEM] Cannot activate virtual environment
        echo   → Try running: rmdir /s /q .venv ^& easy_setup.bat
    )
) else (
    echo [PROBLEM] Virtual environment not found
    echo   → Run easy_setup.bat to create the environment
)

:: Check Python version in virtual environment
echo [CHECK] Checking Python version...
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo [OK] %PYTHON_VERSION%
) else (
    echo [PROBLEM] Python not found in environment
    echo   → Run easy_setup.bat to set up the environment correctly
)

echo.
echo ==== Dependency Check ====
echo.

:: Check critical dependencies
call .venv\Scripts\activate.bat >nul 2>&1

python -c "import torch; print('torch: ' + torch.__version__)" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] torch is installed
) else (
    echo [PROBLEM] torch is not installed
    echo   → Run easy_setup.bat again to install all dependencies
)

python -c "import numpy; print('numpy: ' + numpy.__version__)" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] numpy is installed
) else (
    echo [PROBLEM] numpy is not installed
    echo   → Run easy_setup.bat again to install all dependencies
)

python -c "import gradio; print('gradio: ' + gradio.__version__)" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] gradio is installed
) else (
    echo [PROBLEM] gradio is not installed
    echo   → Run easy_setup.bat again to install all dependencies
)

python -c "import qdrant_client; print('qdrant-client is installed')" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] qdrant-client is installed
) else (
    echo [PROBLEM] qdrant-client is not installed
    echo   → Run easy_setup.bat again to install all dependencies
)

echo.
echo ==== External Dependencies Check ====
echo.

:: Check if perception_models exists
if exist perception_models (
    echo [OK] Perception models folder exists
) else (
    echo [PROBLEM] Perception models folder not found
    echo   → Run easy_setup.bat again or manually clone:
    echo   → git clone https://github.com/facebookresearch/perception.git perception_models
)

echo.
echo ==== Folder Structure Check ====
echo.

:: Check folder structure
if exist my_images (
    echo [OK] Folder exists: my_images
) else (
    echo [PROBLEM] Missing folder: my_images
    echo   → Run: mkdir my_images
)

if exist image_retrieval_project\qdrant_data (
    echo [OK] Folder exists: image_retrieval_project\qdrant_data
) else (
    echo [PROBLEM] Missing folder: image_retrieval_project\qdrant_data
    echo   → Run: mkdir image_retrieval_project\qdrant_data
)

if exist image_retrieval_project\checkpoints (
    echo [OK] Folder exists: image_retrieval_project\checkpoints
) else (
    echo [PROBLEM] Missing folder: image_retrieval_project\checkpoints
    echo   → Run: mkdir image_retrieval_project\checkpoints
)

echo.
echo ==== Application Files Check ====
echo.

:: Check if main.py exists
if exist main.py (
    echo [OK] Main application file exists: main.py
) else (
    echo [PROBLEM] Main application file not found: main.py
    echo   → Make sure you're in the correct directory
    echo   → The application may not be installed correctly
)

echo.
echo ==== Hardware Acceleration Check ====
echo.

:: Check for NVIDIA GPU
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv,noheader 2^>nul') do set GPU_INFO=%%i
    if not "%GPU_INFO%"=="" (
        echo [OK] NVIDIA GPU detected: %GPU_INFO%
        
        :: Check CUDA availability
        python -c "import torch; print(torch.cuda.is_available())" > cuda_check.txt 2>nul
        set /p CUDA_AVAILABLE=<cuda_check.txt
        del cuda_check.txt
        
        if "%CUDA_AVAILABLE%"=="True" (
            echo [OK] CUDA acceleration is available
        ) else (
            echo [WARNING] CUDA acceleration is not available
            echo   → Try reinstalling PyTorch with: pip install torch torchvision torchaudio
        )
    ) else (
        echo [WARNING] NVIDIA tools found but no GPU detected
    )
) else (
    echo [OK] Running in CPU-only mode
)

echo.
echo ==== Database Check ====
echo.

:: Check for database collections
if exist image_retrieval_project\qdrant_data\collections (
    dir /b image_retrieval_project\qdrant_data\collections > collections.txt 2>nul
    set /p COLLECTIONS=<collections.txt
    del collections.txt
    
    if not "%COLLECTIONS%"=="" (
        echo [OK] Database collections found: %COLLECTIONS%
    ) else (
        echo [WARNING] No database collections found
        echo   → This is normal if you haven't processed any images yet
    )
    
    :: Check for lock file
    if exist image_retrieval_project\qdrant_data\.lock (
        echo [WARNING] Database lock file found
        echo   → If the application crashed, you may need to remove the lock file:
        echo   → del image_retrieval_project\qdrant_data\.lock
    )
) else (
    echo [WARNING] Database directory not initialized
    echo   → This is normal if you haven't processed any images yet
)

echo.
echo ==== Summary ====
echo.

echo If you're experiencing issues:
echo 1. Try running easy_setup.bat again to reinstall dependencies
echo 2. Make sure you have images in the 'my_images' folder
echo 3. Check that you have enough disk space and memory
echo 4. For more detailed help, visit the project GitHub page
echo.
echo You can start the application with: run.bat

pause 