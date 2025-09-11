@echo off
REM ATLAS Startup Script for Windows (Conda Version)

echo Starting ATLAS Environmental DNA Analysis Platform...

REM Change to project root directory
cd /d "%~dp0.."

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Check if atlas conda environment exists
conda info --envs | findstr /C:"atlas " >nul
if %errorlevel% neq 0 (
    echo Error: 'atlas' conda environment not found
    echo Please create the environment first with:
    echo   conda env create -f environment.yml
    echo OR for CPU-only:
    echo   conda env create -f environment-cpu.yml
    pause
    exit /b 1
)

REM Install Flask dependencies if not already installed
echo Checking Flask dependencies in atlas environment...
conda run -n atlas pip install -q -r backend/requirements.txt

if %errorlevel% neq 0 (
    echo Error: Failed to install Flask dependencies
    pause
    exit /b 1
)

REM Start Flask backend using conda environment
echo Starting backend server with conda environment 'atlas'...
start "ATLAS Backend" conda run -n atlas python backend/app.py

REM Wait for backend to start
timeout /t 5 /nobreak >nul

echo.
echo ATLAS is now running!
echo.
echo Frontend: Opening in your default browser...
echo Backend API: http://localhost:5000
echo.
echo To stop the backend server, close the "ATLAS Backend" window
echo.
echo Note: Using conda environment 'atlas'
echo.

REM Open frontend in default browser
start "" "frontend\index.html"

pause
