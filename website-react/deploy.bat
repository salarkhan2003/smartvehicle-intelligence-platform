@echo off
echo 🚀 Deploying SIGHTLINE AI Platform to Vercel...

REM Check if git is initialized
if not exist ".git" (
    echo 📁 Initializing Git repository...
    git init
)

REM Add all files
echo 📦 Adding files to Git...
git add .

REM Commit changes
echo 💾 Committing changes...
git commit -m "Deploy: SIGHTLINE AI Platform %date% %time%"

REM Check if remote exists
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo ⚠️  No Git remote found. Please add your GitHub repository:
    echo    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    echo    Then run this script again.
    pause
    exit /b 1
)

REM Push to GitHub
echo 🔄 Pushing to GitHub...
git push origin main

REM Deploy to Vercel (if CLI is installed)
where vercel >nul 2>&1
if %errorlevel% == 0 (
    echo 🌐 Deploying to Vercel...
    vercel --prod
) else (
    echo ✅ Code pushed to GitHub!
    echo 🌐 Go to https://vercel.com to deploy your project
    echo    or install Vercel CLI: npm install -g vercel
)

echo 🎉 Deployment process completed!
pause