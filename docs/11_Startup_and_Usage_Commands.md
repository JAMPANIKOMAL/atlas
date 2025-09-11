# ATLAS Web Interface - Startup and Usage Commands

This document provides step-by-step commands to start and use the ATLAS web interface.

## System Startup Commands

### Windows Users

#### Option 1: Automated Startup (Recommended)
```cmd
# Navigate to ATLAS directory
cd C:\Users\YourName\path\to\atlas

# Run the startup script
scripts\start_atlas.bat

# The script will:
# - Check conda installation
# - Verify atlas environment
# - Install Flask dependencies
# - Start backend server
# - Open frontend in browser
```

#### Option 2: Manual Startup
```cmd
# 1. Check conda environments
conda info --envs

# 2. Activate atlas environment  
conda activate atlas

# 3. Install Flask dependencies (first time only)
pip install -r backend\requirements.txt

# 4. Start backend server
python backend\app.py

# 5. Open frontend (in new terminal)
start frontend\index.html
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
```bash
# Open main page
open frontend/index.html

# Or open analysis page directly
open frontend/html/analysis.html
```

#### Method 2: Local Web Server (Recommended)
```bash
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
```bash
# Create a test FASTA file
cat > test_sample.fasta << EOF
>Test_Sequence_1
ATCGATCGATCGATCGATCG
>Test_Sequence_2
GCTAGCTAGCTAGCTAGCTA
>Test_Sequence_3
TTTTAAAAAGGGGCCCCTTTT
EOF
```

### Test via Command Line (API Testing)
```bash
# 1. Upload file
curl -X POST -F "file=@test_sample.fasta" -F "analysis_mode=rapid" \
     http://localhost:5000/api/upload

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
```bash
# Windows - close the command window or press Ctrl+C
# Linux/macOS - if running in background:
kill $(cat atlas_backend.pid)  # if PID file exists
# or find and kill process:
pkill -f "python backend/app.py"
```

### Clean Up Temporary Files
```bash
# Remove uploaded files
rm -rf backend/uploads/*

# Remove result files  
rm -rf backend/results/*

# Remove generated reports
rm -rf reports/*
```

### Update Dependencies
```bash
# Update Flask dependencies
conda run -n atlas pip install -r backend/requirements.txt --upgrade

# Update conda environment
conda env update -f environment.yml
```

### Backup Results
```bash
# Create backup of results
tar -czf atlas_backup_$(date +%Y%m%d).tar.gz backend/results/

# Or copy to another location
cp -r backend/results/ /path/to/backup/location/
```

## Troubleshooting Commands

### Common Issues

#### Backend Not Starting
```bash
# Check conda environment
conda info --envs | grep atlas

# Test environment
conda run -n atlas python -c "import flask; print('Flask OK')"

# Check port 5000
netstat -an | grep 5000  # Windows: netstat -an | findstr 5000

# Kill process using port 5000
sudo lsof -ti:5000 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :5000        # Windows - then kill PID
```

#### Frontend Not Loading
```bash
# Check if files exist
ls -la frontend/index.html
ls -la frontend/html/analysis.html
ls -la frontend/js/analysis.js

# Check browser console for errors
# Open Developer Tools (F12) → Console tab
```

#### Analysis Failing
```bash
# Check model files
ls -la models/

# Test prediction script directly
conda run -n atlas python src/predict.py --input_fasta test_sample.fasta

# Check backend logs
# Look at terminal running backend for error messages
```

#### Upload Issues
```bash
# Check file permissions
ls -la backend/uploads/
chmod 755 backend/uploads/  # if needed

# Check file format
file test_sample.fasta
head -5 test_sample.fasta
```

## Performance Monitoring

### Check System Resources
```bash
# CPU and memory usage
top  # Linux/macOS
tasklist  # Windows

# Disk space
df -h  # Linux/macOS  
dir   # Windows

# Network connections
netstat -an | grep 5000
```

### Backend Performance
```bash
# Test response time
time curl http://localhost:5000/api/health

# Monitor requests (if needed)
tail -f backend/app.log  # if logging enabled
```

## Quick Reference

### Essential Commands
```bash
# Start system
scripts\start_atlas.bat      # Windows
scripts/start_atlas.sh       # Linux/macOS

# Test backend  
curl http://localhost:5000/api/health

# Stop system
Ctrl+C                       # In backend terminal
kill $(cat atlas_backend.pid)  # Background process

# Clean up
rm -rf backend/uploads/* backend/results/*
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
