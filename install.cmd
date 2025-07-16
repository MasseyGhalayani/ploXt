@echo off
:: This script sets up a Python virtual environment for the project
:: and installs all necessary dependencies.
:: It is designed for Windows with an NVIDIA GPU and CUDA 11.7 already installed. tested for Python 3.9.23

echo.
echo =================================================
echo      Chart Data Extractor Installation Script
echo =================================================
echo.

:: --- 1. Create Virtual Environment ---
echo Creating Python virtual environment in 'venv'...
python -m venv venv
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to create virtual environment.
    echo Please ensure you have Python 3 installed and available in your system's PATH.
    goto :error
)

:: --- 2. Activate Environment ---
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: --- 3. Upgrade Pip ---
echo Upgrading pip...
python -m pip install --upgrade pip

:: --- 4. Install Core Deep Learning Libraries ---
echo.
echo Installing PyTorch for CUDA 11.7...
echo This command is specific to PyTorch 1.13.1 and CUDA 11.7.
echo For other versions, visit https://pytorch.org/get-started/previous-versions/
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install PyTorch.
    echo Please check your CUDA setup, network connection, and ensure you are running this script as an administrator if needed.
    goto :error
)

echo.
echo Installing MMCV...
pip install openmim
:: We install mmcv-full this way to ensure it's compiled correctly.
mim install mmcv-full==1.7.2
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install mmcv-full.
    echo Please check your openmim installation and that Visual C++ Build Tools are installed.
    goto :error
)

:: --- 5. Install MMDetection (Editable) ---
if exist "mmdetection\" (
    echo.
    echo Installing mmdetection in editable mode...
    pip install -e mmdetection
) else (
    echo.
    echo WARNING: 'mmdetection' directory not found. Skipping editable install.
    echo The line segmentation model may not work.
)



:: --- 6. Install Remaining Dependencies ---
echo.
echo Installing other dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install packages from requirements.txt.
    goto :error
)


echo.
echo.
echo =============================
echo --- Installation Complete ---
echo =============================
echo.
echo To use the application, activate the environment by running:
echo     .\venv\Scripts\activate
echo.
goto :eof

:error
echo.
echo Installation failed. Please check the error messages above.

:eof
pause