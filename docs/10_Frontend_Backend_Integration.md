# ATLAS Frontend & Backend Integration Guide

This document provides comprehensive instructions for setting up and running the ATLAS web interface with its backend API.

## Architecture Overview

ATLAS consists of three main components:
1. **Frontend**: HTML/CSS/JavaScript web interface
2. **Backend**: Flask REST API server 
3. **Analysis Pipeline**: Python-based bioinformatics workflow

```
+------------------+    HTTP/REST    +------------------+    Python    +------------------+
|   Frontend       | ---------------> |   Flask API      | ------------> | Analysis Engine  |
|   (Browser)      |                  |   Backend        |               | (src/predict.py) |
+------------------+                  +------------------+               +------------------+
```

## Quick Start Guide

### Prerequisites
- Conda environment manager (Anaconda/Miniconda)
- Modern web browser
- 'atlas' conda environment created

### Step 1: Verify Environment
```bash
# Check conda environments
conda info --envs

# Should show 'atlas' environment
```

### Step 2: Start the System
```bash
# Windows
scripts\start_atlas.bat

# Linux/macOS  
scripts/start_atlas.sh
```

### Step 3: Access the Interface
- Backend API: http://localhost:5000
- Frontend: Open `frontend/index.html` in your browser

## File Organization

```
atlas/
├── backend/                    # Flask API server
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt       # Flask dependencies
│   ├── test_environment.py    # Environment tester
│   ├── uploads/              # Uploaded files storage
│   └── results/              # Analysis results storage
├── frontend/                  # Web interface
│   ├── index.html            # Landing page
│   ├── html/analysis.html    # Analysis interface
│   ├── js/analysis.js        # Analysis functionality
│   ├── css/styles.css        # Styling
│   └── assets/               # Images, videos, fonts
├── scripts/                   # Startup and utility scripts
│   ├── start_atlas.bat       # Windows startup script
│   └── start_atlas.sh        # Linux/macOS startup script
├── tests/                     # Test files and data
├── src/                       # Analysis pipeline
│   └── predict.py            # Main prediction script
├── models/                    # AI models (16S, 18S, COI, ITS)
├── docs/                      # Documentation
└── environment.yml            # Conda environment definition
```

## Manual Setup (Alternative)

If the automated scripts don't work, follow these manual steps:

### 1. Environment Setup
```bash
# Activate conda environment
conda activate atlas

# Install Flask dependencies
conda run -n atlas pip install -r backend/requirements.txt
```

### 2. Start Backend Manually
```bash
# Navigate to project root
cd /path/to/atlas

# Start Flask server
conda run -n atlas python backend/app.py
```

### 3. Open Frontend
Open `frontend/index.html` in your web browser or serve it via a local server.

## API Endpoints

The Flask backend provides these REST endpoints:

### Health Check
```http
GET /api/health
```
Returns server status and version info.

### File Upload
```http
POST /api/upload
Content-Type: multipart/form-data

Parameters:
- file: FASTA file (optional)
- sequences: Text sequences (optional) 
- analysis_mode: rapid|comprehensive|custom
```

### Start Analysis
```http
POST /api/analyze/{job_id}
```

### Check Status  
```http
GET /api/status/{job_id}
```

### Get Results
```http
GET /api/results/{job_id}
```

### Download Files
```http
GET /api/download/{job_id}/{filename}
```

## Testing the System

### 1. Test Backend API
```bash
# Health check
curl http://localhost:5000/api/health

# Should return:
# {
#   "status": "healthy",
#   "timestamp": "2025-09-11T...",
#   "version": "1.0.0"
# }
```

### 2. Test File Upload
```bash
# Create test file
echo ">Test_Seq\nATCGATCGATCG" > test.fasta

# Upload file
curl -X POST -F "file=@test.fasta" -F "analysis_mode=rapid" \
     http://localhost:5000/api/upload

# Should return job_id
```

### 3. Test Analysis
```bash
# Replace JOB_ID with actual ID from upload
curl -X POST http://localhost:5000/api/analyze/JOB_ID

# Check status
curl http://localhost:5000/api/status/JOB_ID
```

### 4. Test Frontend
1. Open `frontend/index.html`
2. Click "Start Analysis"
3. Upload a FASTA file or paste sequences
4. Click "Start Analysis" button
5. Monitor progress and view results

## Frontend Features

### Upload Interface
- **Drag & Drop**: Drop FASTA files directly
- **File Browser**: Click to select files
- **Text Input**: Paste sequences directly
- **Validation**: Automatic FASTA format checking

### Analysis Options
- **Rapid**: Quick species identification
- **Comprehensive**: Detailed biodiversity analysis
- **Custom**: Advanced parameters (future)

### Results Display
- **Real-time Progress**: Live status updates
- **Species List**: Top identified species
- **Diversity Metrics**: Shannon, Simpson indices
- **Downloads**: CSV reports and JSON data

## Troubleshooting

### Backend Issues
```bash
# Check if backend is running
curl http://localhost:5000/api/health

# Check conda environment
conda run -n atlas python backend/test_environment.py

# Manual start
conda run -n atlas python backend/app.py
```

### Frontend Issues
- **CORS Errors**: Ensure backend is running on localhost:5000
- **Upload Fails**: Check file format (must be valid FASTA)
- **No Progress**: Check browser console for error messages

### Common Problems
1. **Port 5000 in use**: Stop other services using port 5000
2. **Models missing**: System works in demo mode without model files
3. **Permission errors**: Check file/folder permissions

## Security Notes

- System runs locally - no data sent to external servers
- Files are temporarily stored in `backend/uploads/`
- Results stored in `backend/results/`
- Automatic cleanup of old files (future feature)

## System Status Indicators

### Backend Status
- **HEALTHY**: API responds with 200 status
- **WARNING**: Slow responses or errors
- **ERROR**: Server not responding

### Frontend Status  
- **CONNECTED**: Successfully communicating with backend
- **LIMITED**: Some features unavailable
- **OFFLINE**: Cannot connect to backend

## Development Notes

### Adding New Features
1. Backend: Add endpoints to `backend/app.py`
2. Frontend: Update `frontend/js/analysis.js`
3. Test: Use curl commands or browser testing

### Model Integration
- Place model files in `models/` directory
- Update `src/predict.py` if needed
- System falls back to demo mode if models missing

## Next Steps

- [ ] Add user authentication
- [ ] Implement file cleanup scheduler  
- [ ] Add progress persistence
- [ ] Enhance error handling
- [ ] Add batch processing
- [ ] Implement result caching
