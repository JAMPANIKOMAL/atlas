# ATLAS Startup Script for Windows (PowerShell Version)

Write-Host "Starting ATLAS Environmental DNA Analysis Platform..."

# Change to project root directory
Set-Location $PSScriptRoot\..

# Check if conda is available
if (!(Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Conda is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Anaconda or Miniconda first"
    Read-Host "Press Enter to continue"
    exit 1
}

# Check if atlas conda environment exists
$envs = conda info --envs
if (!($envs -match "atlas\s")) {
    Write-Host "Error: 'atlas' conda environment not found" -ForegroundColor Red
    Write-Host "Please create the environment first with:"
    Write-Host "  conda env create -f environment.yml"
    Write-Host "OR for CPU-only:"
    Write-Host "  conda env create -f environment-cpu.yml"
    Read-Host "Press Enter to continue"
    exit 1
}

# Install Flask dependencies if not already installed
Write-Host "Checking Flask dependencies in atlas environment..."
$result = & conda run -n atlas pip install -q -r backend/requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install Flask dependencies" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

# Start Flask backend using conda environment
Write-Host "Starting backend server with conda environment 'atlas'..."
Start-Process -WindowStyle Normal -FilePath "conda" -ArgumentList "run", "-n", "atlas", "python", "backend/app.py"

# Wait for backend to start
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "ATLAS is now running!" -ForegroundColor Green
Write-Host ""
Write-Host "Frontend: Opening in your default browser..."
Write-Host "Backend API: http://localhost:5000"
Write-Host ""
Write-Host "To stop the backend server, close the backend window"
Write-Host ""
Write-Host "Note: Using conda environment 'atlas'"
Write-Host ""

# Open frontend in default browser
Invoke-Item "frontend\index.html"

Read-Host "Press Enter to continue"
