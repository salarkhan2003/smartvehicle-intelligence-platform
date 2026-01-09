@echo off
REM SmartBus Intelligence Platform - Quick Start Script
REM Unified Blind Spot Safety + Traffic Enforcement

echo ========================================================================
echo    SMARTBUS INTELLIGENCE PLATFORM - QUICK START
echo    Unified Blind Spot Safety + Traffic Violation Enforcement
echo ========================================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+ first.
    pause
    exit /b 1
)

echo [1/6] Python detected:
python --version
echo.

REM Create models directory if not exists
if not exist "models" (
    echo [2/6] Creating models directory...
    mkdir models
)

REM Create violations directory if not exists
if not exist "violations" (
    echo Creating violations directory...
    mkdir violations
)

REM Create logs directory
if not exist "logs" (
    echo Creating logs directory...
    mkdir logs
)

echo [2/6] Directories ready
echo.

REM Check if models are trained
echo [3/6] Checking ML models...
if not exist "models\helmet_detector.pt" (
    echo     Helmet detector: NOT FOUND
    echo     Run: python train_yolo_helmet.py --generate
) else (
    echo     Helmet detector: OK
)

if not exist "models\seatbelt_detector.pt" (
    echo     Seatbelt detector: NOT FOUND
    echo     Run: python train_yolo_seatbelt.py
) else (
    echo     Seatbelt detector: OK
)

if not exist "models\traffic_detector.pt" (
    echo     Traffic detector: NOT FOUND
    echo     Run: python train_yolo_traffic.py
) else (
    echo     Traffic detector: OK
)

if not exist "yolov8n.pt" (
    echo     Base YOLO model: NOT FOUND - will download automatically
) else (
    echo     Base YOLO model: OK
)
echo.

REM Menu selection
echo [4/6] SELECT MODE:
echo ========================================================================
echo    1. Train All Models (First-time setup - 2 hours)
echo    2. Run Unified Detection Engine (Camera + Detection)
echo    3. Launch Dashboard Only (GUI)
echo    4. PMO Installation Guide Generator
echo    5. Run Complete Demo (Dashboard + Engine)
echo    6. Train Individual Model
echo ========================================================================
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto train_all
if "%choice%"=="2" goto run_engine
if "%choice%"=="3" goto run_dashboard
if "%choice%"=="4" goto pmo_install
if "%choice%"=="5" goto run_demo
if "%choice%"=="6" goto train_individual
goto invalid

:train_all
echo.
echo [5/6] Training all models...
echo This will take approximately 2 hours on CPU
echo ========================================================================
echo.
echo Training Helmet Detection Model...
python train_yolo_helmet.py --epochs 50 --generate
if errorlevel 1 (
    echo [ERROR] Helmet training failed
    pause
    exit /b 1
)
echo.
echo Training Seatbelt Detection Model...
python train_yolo_seatbelt.py --epochs 50
if errorlevel 1 (
    echo [ERROR] Seatbelt training failed
    pause
    exit /b 1
)
echo.
echo Training Traffic Detection Model...
python train_yolo_traffic.py --epochs 50
if errorlevel 1 (
    echo [ERROR] Traffic training failed
    pause
    exit /b 1
)
echo.
echo [6/6] All models trained successfully!
echo Models saved in: models\
pause
goto end

:run_engine
echo.
echo [5/6] Starting Unified Detection Engine...
echo ========================================================================
echo.
echo Mode: Unified (Blind Spot + Traffic Violations)
echo Camera: Default (ID 0)
echo Config: config.json
echo.
echo Press Ctrl+C to stop
echo ========================================================================
echo.
python unified_detection_engine.py --mode unified --camera 0 --config config.json
if errorlevel 1 (
    echo [ERROR] Detection engine failed
    pause
    exit /b 1
)
goto end

:run_dashboard
echo.
echo [5/6] Launching Unified Dashboard...
echo ========================================================================
echo.
python unified_dashboard.py
if errorlevel 1 (
    echo [ERROR] Dashboard failed to launch
    pause
    exit /b 1
)
goto end

:pmo_install
echo.
echo [5/6] PMO Installation Guide Generator
echo ========================================================================
echo.
set /p bus_id="Enter Bus ID (e.g., SGP1234): "
set /p depot="Enter Depot Name (e.g., Bedok): "
set /p installer="Enter Installer Name: "
echo.
echo Generating installation guide for Bus %bus_id%...
python pmo_installation.py --bus %bus_id% --depot %depot% --installer "%installer%" --generate-pdf
if errorlevel 1 (
    echo [ERROR] Installation guide generation failed
    pause
    exit /b 1
)
echo.
echo [6/6] Installation guide generated!
pause
goto end

:run_demo
echo.
echo [5/6] Starting Complete Demo System...
echo ========================================================================
echo.
echo This will start:
echo   1. Unified Detection Engine (background)
echo   2. Unified Dashboard (GUI)
echo.
echo Press any key to continue...
pause >nul
echo.
echo Starting detection engine in background...
start /B python unified_detection_engine.py --mode unified --camera 0 --config config.json
timeout /t 3 /nobreak >nul
echo.
echo Launching dashboard...
python unified_dashboard.py
if errorlevel 1 (
    echo [ERROR] Dashboard failed
    pause
    exit /b 1
)
goto end

:train_individual
echo.
echo SELECT MODEL TO TRAIN:
echo ========================================================================
echo    1. Helmet Detection (50 epochs, ~40 min)
echo    2. Seatbelt Detection (50 epochs, ~40 min)
echo    3. Traffic Scene Detection (50 epochs, ~50 min)
echo ========================================================================
echo.
set /p model_choice="Enter choice (1-3): "

if "%model_choice%"=="1" (
    echo.
    echo Training Helmet Detection Model...
    python train_yolo_helmet.py --epochs 50 --generate
) else if "%model_choice%"=="2" (
    echo.
    echo Training Seatbelt Detection Model...
    python train_yolo_seatbelt.py --epochs 50
) else if "%model_choice%"=="3" (
    echo.
    echo Training Traffic Scene Detection Model...
    python train_yolo_traffic.py --epochs 50
) else (
    echo Invalid choice
    pause
    goto end
)

if errorlevel 1 (
    echo [ERROR] Training failed
    pause
    exit /b 1
)
echo.
echo Training complete!
pause
goto end

:invalid
echo Invalid choice! Please select 1-6
pause
goto end

:end
echo.
echo ========================================================================
echo    SMARTBUS INTELLIGENCE PLATFORM
echo    Thank you for using our system!
echo ========================================================================
echo.
pause

