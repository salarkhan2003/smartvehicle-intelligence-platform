@echo off
echo.
echo 🚗 SmartVehicle Intelligence - React Landing Page
echo ================================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found! Please install Node.js 18+ from nodejs.org
    echo.
    pause
    exit /b 1
)

echo ✅ Node.js found
echo.

REM Install dependencies with legacy peer deps to fix conflicts
echo 📦 Installing dependencies (this may take a few minutes)...
echo.
npm install --legacy-peer-deps

if %errorlevel% neq 0 (
    echo.
    echo ❌ Installation failed! Trying alternative method...
    echo.
    npm install --force
)

echo.
echo ✅ Dependencies installed successfully!
echo.

REM Check if video files exist
if not exist "public\assets\AI VID.mp4" (
    echo ⚠️  Video file missing: public\assets\AI VID.mp4
    echo    Please copy your AI VID.mp4 to public\assets\ folder
    echo.
)

if not exist "public\assets\CAR VIDEO.mp4" (
    echo ⚠️  Video file missing: public\assets\CAR VIDEO.mp4
    echo    Please copy your CAR VIDEO.mp4 to public\assets\ folder
    echo.
)

echo 🚀 Starting development server...
echo.
echo 📝 Instructions:
echo    • Server will start on http://localhost:3000
echo    • Browser will open automatically
echo    • Press Ctrl+C to stop the server
echo    • Hot reload enabled for development
echo.

REM Start the development server
npm run dev

echo.
echo 👋 Thanks for using SmartVehicle Intelligence!
pause