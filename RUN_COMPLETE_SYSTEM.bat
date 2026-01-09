@echo off
REM Smart Vehicle Intelligence System - Complete Run Script
REM Trains models then runs the complete system

echo ================================================================================
echo    SMART VEHICLE INTELLIGENCE SYSTEM
echo    Complete Training + Execution Pipeline
echo ================================================================================
echo.
echo    Works with ALL vehicles: Cars, Trucks, Buses, Motorcycles, Bicycles
echo    Blind Spot Safety + Traffic Violation Enforcement
echo.
echo ================================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+ first.
    pause
    exit /b 1
)

echo [OK] Python detected
python --version
echo.

REM Create directories
if not exist "models" mkdir models
if not exist "violations" mkdir violations
if not exist "logs" mkdir logs
echo [OK] Directories ready
echo.

REM Check if models are trained
echo Checking trained models...
set MODELS_TRAINED=0

if exist "models\helmet_detector.pt" (
    echo   [OK] Helmet detector found
    set /a MODELS_TRAINED+=1
)
if exist "models\seatbelt_detector.pt" (
    echo   [OK] Seatbelt detector found
    set /a MODELS_TRAINED+=1
)
if exist "models\traffic_detector.pt" (
    echo   [OK] Traffic detector found
    set /a MODELS_TRAINED+=1
)

echo.
echo Models trained: %MODELS_TRAINED%/3
echo.

if %MODELS_TRAINED% LSS 3 (
    echo ================================================================================
    echo    MODELS NOT FULLY TRAINED
    echo ================================================================================
    echo.
    echo Some models are missing. Training is required.
    echo This will take approximately 2 hours on CPU.
    echo.
    set /p train_choice="Train models now? (Y/N): "

    if /i "%train_choice%"=="Y" (
        echo.
        echo Starting training pipeline...
        echo.
        python train_all_models.py

        if errorlevel 1 (
            echo.
            echo [ERROR] Training failed!
            pause
            exit /b 1
        )

        echo.
        echo [OK] Training complete!
        echo.
    ) else (
        echo.
        echo [WARNING] Running without custom models - will use base YOLO
        echo.
    )
)

echo ================================================================================
echo    RUNNING SYSTEM
echo ================================================================================
echo.
echo Select run mode:
echo   1. Quick Demo (no training required, instant start)
echo   2. Detection Engine Only (camera + detection)
echo   3. Dashboard Only (GUI)
echo   4. Full System (Engine + Dashboard - RECOMMENDED)
echo   5. Train Models Only
echo   6. Exit
echo.

set /p run_choice="Enter choice (1-6): "

if "%run_choice%"=="1" goto quick_demo
if "%run_choice%"=="2" goto engine_only
if "%run_choice%"=="3" goto dashboard_only
if "%run_choice%"=="4" goto full_system
if "%run_choice%"=="5" goto train_only
if "%run_choice%"=="6" goto end
goto invalid

:quick_demo
echo.
echo ================================================================================
echo    QUICK DEMO MODE
echo ================================================================================
echo.
echo Starting quick demo (no training required)...
echo Press 'q' to quit, 's' to save screenshot
echo.
python demo_quick.py
goto end

:engine_only
echo.
echo ================================================================================
echo    DETECTION ENGINE
echo ================================================================================
echo.
echo Vehicle type:
echo   1. Auto-detect (recommended)
echo   2. Car
echo   3. Truck
echo   4. Bus
echo   5. Motorcycle
echo.
set /p vehicle_type="Select vehicle type (1-5): "

set VEHICLE=auto
if "%vehicle_type%"=="2" set VEHICLE=car
if "%vehicle_type%"=="3" set VEHICLE=truck
if "%vehicle_type%"=="4" set VEHICLE=bus
if "%vehicle_type%"=="5" set VEHICLE=motorcycle

echo.
echo Starting detection engine...
echo Vehicle type: %VEHICLE%
echo Mode: Unified (Blind Spot + Traffic Violations)
echo.
echo Press Ctrl+C to stop
echo.

python unified_detection_engine.py --mode unified --vehicle-type %VEHICLE% --camera 0 --blind_spot True --traffic_violation True

goto end

:dashboard_only
echo.
echo ================================================================================
echo    DASHBOARD GUI
echo ================================================================================
echo.
echo Launching dashboard...
echo.

python unified_dashboard.py

goto end

:full_system
echo.
echo ================================================================================
echo    FULL SYSTEM MODE (RECOMMENDED)
echo ================================================================================
echo.
echo This will start:
echo   1. Detection Engine (background)
echo   2. Dashboard GUI (foreground)
echo.
echo Press any key to continue...
pause >nul

echo.
echo Starting detection engine in background...
start "Detection Engine" cmd /c python unified_detection_engine.py --mode unified --camera 0

echo Waiting 5 seconds for engine to initialize...
timeout /t 5 /nobreak >nul

echo.
echo Launching dashboard...
python unified_dashboard.py

echo.
echo Closing detection engine...
taskkill /FI "WINDOWTITLE eq Detection Engine*" /T /F >nul 2>&1

goto end

:train_only
echo.
echo ================================================================================
echo    TRAINING MODELS ONLY
echo ================================================================================
echo.
echo Starting training pipeline...
echo Estimated time: 2 hours on CPU
echo.

python train_all_models.py

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo [OK] Training complete!
pause
goto end

:invalid
echo.
echo [ERROR] Invalid choice!
pause
goto end

:end
echo.
echo ================================================================================
echo    SMART VEHICLE INTELLIGENCE SYSTEM
echo    Session ended
echo ================================================================================
echo.
echo Thank you for using Smart Vehicle Intelligence System!
echo.
pause

