@echo off
echo ========================================
echo SmartVehicle Intelligence System v2.0
echo TNT Surveillance PMO Engineer Demo
echo ========================================
echo.

REM Check Python installation
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install PySide6 opencv-python ultralytics mediapipe numpy pyttsx3

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup complete! Starting application...
echo ========================================
echo.

python main.py

pause
