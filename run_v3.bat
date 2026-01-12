@echo off
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  SmartVehicle Intelligence System v3.0 - Enterprise Edition  ║
echo ║  Starting with ALL 35 Features...                            ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8-3.11
    pause
    exit /b 1
)
echo.
echo Checking dependencies...
python -c "import PySide6, cv2, numpy, ultralytics, mediapipe, easyocr" 2>nul
if errorlevel 1 (
    echo.
    echo Some dependencies missing. Installing now...
    echo This may take 5-10 minutes...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency installation failed!
        echo Please run manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
)
echo.
echo ✓ All dependencies installed
echo.
echo ═══════════════════════════════════════════════════════════════
echo  Launching SmartVehicle Intelligence System v3.0...
echo ═══════════════════════════════════════════════════════════════
echo.
python main_v3.py
if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start!
    echo Check the error messages above.
    pause
    exit /b 1
)
echo.
echo System stopped.
pause
