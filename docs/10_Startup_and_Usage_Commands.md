# ATLAS Web Interface - Startup and Usage Commands

This document provides step-by-step commands to start and use the ATLAS web interface.

## System Startup Commands

### Windows Users

#### Option 1: Automated Startup (Recommended)
```powershell
# Navigate to ATLAS directory
cd C:\Users\YourName\path\to\atlas

# Run the PowerShell startup script
scripts\start_atlas.ps1

# The script will:
# - Check conda installation
# - Verify atlas environment
# - Install Flask dependencies
# - Start backend server
# - Open frontend in browser
```

#### Option 2: Manual Startup
```powershell
# 1. Check conda environments
conda info --envs

# 2. Install Flask dependencies (first time only)
conda run -n atlas pip install -r backend\requirements.txt

# 3. Start backend server
conda run -n atlas python backend\app.py

# 4. Open frontend (in new PowerShell terminal)
Invoke-Item frontend\index.html
```

### Linux/macOS Users

#### Option 1: Automated Startup (Recommended)
```bash
# Navigate to ATLAS directory
cd /path/to/atlas

# Make script executable (first time only)
chmod +x scripts/start_atlas.sh

# Run the startup script
scripts/start_atlas.sh

# The script will:
# - Check conda installation
# - Verify atlas environment  
# - Install Flask dependencies
# - Start backend server in background
# - Provide frontend access instructions
```

#### Option 2: Manual Startup
```bash
# 1. Check conda environments
conda info --envs

# 2. Install Flask dependencies (first time only)
conda run -n atlas pip install -r backend/requirements.txt

# 3. Start backend server
conda run -n atlas python backend/app.py &

# 4. Open frontend
open frontend/index.html  # macOS
xdg-open frontend/index.html  # Linux
```

## System Verification Commands

### Check Backend Health
```bash
# Test if backend is running
curl http://localhost:5000/api/health

# Expected response:
# {
#   "status": "healthy", 
#   "timestamp": "2025-09-11T...",
#   "version": "1.0.0"
# }
```

### Test Environment
```bash
# Run environment test script
conda run -n atlas python backend/test_environment.py

# Should show all green checkmarks
```

### Verify File Structure
```bash
# Check main directories exist
ls -la atlas/
# Should show: backend/, frontend/, src/, models/, docs/

# Check backend files
ls -la backend/
# Should show: app.py, requirements.txt, test_environment.py

# Check frontend files  
ls -la frontend/
# Should show: index.html, html/, js/, css/, assets/
```

## Frontend Usage Commands

### Access the Web Interface

#### Method 1: Direct File Access
```powershell
# Open main page
Invoke-Item frontend\index.html

# Or open analysis page directly
Invoke-Item frontend\html\analysis.html
```

#### Method 2: Local Web Server (Recommended)
```powershell
# Using Python's built-in server
cd frontend
python -m http.server 8080

# Then open: http://localhost:8080
```

#### Method 3: VS Code Live Server
```bash
# If using VS Code with Live Server extension
code frontend/index.html
# Right-click → "Open with Live Server"
```

### Frontend Navigation
1. **Home Page**: `frontend/index.html`
   - Project overview
   - Click "Start Analysis" button

2. **Analysis Page**: `frontend/html/analysis.html`
   - Upload interface
   - Analysis options
   - Results display

## Testing Upload and Analysis

### Create Test Data
```powershell
# Create a test FASTA file
@'
>Test_Sequence_1
ATCGATCGATCGATCGATCG
>Test_Sequence_2
GCTAGCTAGCTAGCTAGCTA
>Test_Sequence_3
TTTTAAAAAGGGGCCCCTTTT
'@ | Out-File -FilePath test_sample.fasta -Encoding ASCII
```

### Test via Command Line (API Testing)
```powershell
# 1. Upload file
curl -X POST -F "file=@test_sample.fasta" -F "analysis_mode=rapid" http://localhost:5000/api/upload

# Response will include job_id like:
# {"job_id": "12345-abcde-...", "message": "File uploaded successfully"}

# 2. Start analysis (replace JOB_ID with actual ID)
curl -X POST http://localhost:5000/api/analyze/JOB_ID

# 3. Check status
curl http://localhost:5000/api/status/JOB_ID

# 4. Get results when complete
curl http://localhost:5000/api/results/JOB_ID

# 5. Download files
curl -O http://localhost:5000/api/download/JOB_ID/results.json
curl -O http://localhost:5000/api/download/JOB_ID/classification_report.csv
```

### Test via Web Interface
1. **Upload Method 1 - File Upload**:
   - Open analysis page
   - Drag `test_sample.fasta` to upload area
   - Or click "Choose File" and select it
   - Select analysis mode: "Rapid Analysis"
   - Click "Start Analysis"

2. **Upload Method 2 - Text Input**:
   - Open analysis page
   - Paste FASTA sequences in text area:
     ```
     >Test_Sequence_1
     ATCGATCGATCGATCGATCG
     >Test_Sequence_2
     GCTAGCTAGCTAGCTAGCTA
     ```
   - Select analysis mode
   - Click "Start Analysis"

3. **Monitor Progress**:
   - Watch real-time progress indicators
   - Progress steps: Data Processing → AI Analysis → Results Generation

4. **View Results**:
   - Species identification table
   - Biodiversity metrics
   - Download CSV/JSON reports

## Maintenance Commands

### Stop the System
```powershell
# Windows - close the command window or press Ctrl+C
# If running as background process, find and stop:
Get-Process | Where-Object {$_.ProcessName -eq "python" -and $_.CommandLine -like "*backend/app.py*"} | Stop-Process

# Linux/macOS - if running in background:
kill $(cat atlas_backend.pid)  # if PID file exists
# or find and kill process:
pkill -f "python backend/app.py"
```

### Clean Up Temporary Files
```powershell
# Remove uploaded files
Remove-Item -Recurse -Force backend\uploads\*

# Remove result files  
Remove-Item -Recurse -Force backend\results\*

# Remove generated reports (if reports directory exists)
if (Test-Path reports) { Remove-Item -Recurse -Force reports\* }
```

### Update Dependencies
```powershell
# Update Flask dependencies
conda run -n atlas pip install -r backend\requirements.txt --upgrade

# Update conda environment
conda env update -f environment.yml
```

### Backup Results
```powershell
# Create backup of results
$date = Get-Date -Format "yyyyMMdd"
Compress-Archive -Path backend\results\* -DestinationPath "atlas_backup_$date.zip"

# Or copy to another location
Copy-Item -Recurse backend\results\ C:\path\to\backup\location\
```

## Troubleshooting Commands

### Common Issues

#### Backend Not Starting
```powershell
# Check conda environment
conda info --envs | findstr atlas

# Test environment
conda run -n atlas python -c "import flask; print('Flask OK')"

# Check port 5000
netstat -an | findstr 5000

# Kill process using port 5000
$process = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if ($process) { Stop-Process -Id $process.OwningProcess }
```

#### Frontend Not Loading
```powershell
# Check if files exist
Get-Item frontend\index.html
Get-Item frontend\html\analysis.html
Get-Item frontend\js\analysis.js

# Check browser console for errors
# Open Developer Tools (F12) → Console tab
```

#### Analysis Failing
```powershell
# Check model files
Get-ChildItem models\

# Test prediction script directly
conda run -n atlas python src\predict.py --input_fasta test_sample.fasta

# Check backend logs
# Look at terminal running backend for error messages
```

#### Upload Issues
```powershell
# Check file permissions and directory
Get-ChildItem backend\uploads\ | Format-List

# Check file format
Get-Content test_sample.fasta | Select-Object -First 5
```

## Performance Monitoring

### Check System Resources
```powershell
# CPU and memory usage
Get-Process | Where-Object {$_.ProcessName -eq "python"} | Select-Object ProcessName, CPU, WorkingSet

# Disk space
Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, @{Name="Size(GB)";Expression={[math]::Round($_.Size/1GB,2)}}, @{Name="FreeSpace(GB)";Expression={[math]::Round($_.FreeSpace/1GB,2)}}

# Network connections
netstat -an | findstr 5000
```

### Backend Performance
```powershell
# Test response time
Measure-Command { curl http://localhost:5000/api/health }

# Monitor requests (if needed)
# Check terminal running backend for log messages
```

## Quick Reference

### Essential Commands
```powershell
# Start system (Windows)
scripts\start_atlas.ps1

# Start system (Linux/macOS)  
scripts/start_atlas.sh

# Test backend  
curl http://localhost:5000/api/health

# Stop system
Ctrl+C                                    # In backend terminal
Stop-Process -Name python -Force         # Kill all Python processes

# Clean up
Remove-Item -Recurse -Force backend\uploads\*, backend\results\*
```

### Important URLs
- Backend Health: http://localhost:5000/api/health
- Frontend Home: file:///path/to/atlas/frontend/index.html
- Analysis Page: file:///path/to/atlas/frontend/html/analysis.html

### File Locations
- Uploads: `backend/uploads/`
- Results: `backend/results/`  
- Reports: `reports/`
- Logs: Terminal output (backend/app.py)
